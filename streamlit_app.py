import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import io
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import pytesseract
import re
import imagehash
import uuid
import os
import openai
import base64

# Function to get OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Set OpenAI API key
try:
    openai.api_key = get_api_key()
except ValueError as e:
    st.error(str(e))

# Load the embedding model (cached to avoid reloading on every app refresh)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Initialize MinIO client for Cloudflare R2
def initialize_minio():
    try:
        return Minio(
            st.secrets["R2_ENDPOINT"].replace("https://", ""),
            access_key=st.secrets["R2_ACCESS_KEY"],
            secret_key=st.secrets["R2_SECRET_KEY"],
            secure=True
        )
    except KeyError as e:
        st.error(f"MinIO initialization failed: Missing secret key {e}")
        return None

minio_client = initialize_minio()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets["qdrant"]["url"],
    api_key=st.secrets["qdrant"]["api_key"]
)

# Load the reference header image from the assets folder and calculate its hash
reference_image_path = "assets/header_image.png"  # Adjust the path based on your repository structure
try:
    reference_image = Image.open(reference_image_path)
    reference_image_hash = imagehash.phash(reference_image)
except FileNotFoundError:
    st.error(f"Reference header image not found at {reference_image_path}. Please ensure it is available.")
    reference_image_hash = None

# Function to recreate the Qdrant collection
def recreate_qdrant_collection():
    try:
        qdrant_client.delete_collection(collection_name="manual_vectors")
    except Exception as e:
        st.warning(f"Failed to delete existing collection: {e}")

    # Create or recreate the Qdrant collection with proper configuration
    qdrant_client.create_collection(
        collection_name="manual_vectors",
        vectors_config=VectorParams(
            size=384,  # Size of the vector (embedding dimension)
            distance="Cosine"
        )
    )

# Function to extract text and images from PDFs, vectorize, and store in Qdrant
def vectorize_pdfs():
    if not minio_client:
        st.error("MinIO client not initialized.")
        return

    if reference_image_hash is None:
        st.error("Reference header image hash could not be calculated.")
        return
    
    # Get list of all PDFs from Cloudflare R2
    try:
        objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"], recursive=True)
        pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    except S3Error as e:
        st.error(f"Error listing PDF files from Cloudflare R2: {e}")
        return

    vectors = []
    extracted_images_count = 0

    # Process each PDF file
    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            # Extract text and images
            for page_num, page in enumerate(doc):
                text = page.get_text()
                sentences = re.split(r'(?<=[.!?])\s+', text)

                # Vectorize text chunks
                for sentence in sentences:
                    if len(sentence.strip()) > 0:
                        embedding = model.encode(sentence.strip()).tolist()
                        point_id = str(uuid.uuid4())  # Generate a unique UUID for each point
                        vectors.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "text",
                                "page": page_num + 1,
                                "content": sentence.strip(),
                                "file_name": pdf_file_name
                            }
                        ))

                # Extract and vectorize images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Calculate the hash of the image
                    image_hash = imagehash.phash(image)

                    # Skip the image if it matches the reference header image hash
                    if image_hash == reference_image_hash:
                        continue

                    # Perform OCR on the image
                    try:
                        image_text = pytesseract.image_to_string(image)
                    except Exception:
                        image_text = "OCR failed for this image"

                    # Get nearby text from the page
                    bbox = img[1]
                    try:
                        extended_rect = fitz.Rect(bbox).expand(50)
                        nearby_text = page.get_text("text", clip=extended_rect)
                    except Exception:
                        nearby_text = "No nearby text found"

                    # Vectorize image with metadata (nearby and OCR text)
                    metadata_text = f"Image OCR text: {image_text}. Nearby text: {nearby_text}."
                    embedding = model.encode(metadata_text).tolist()
                    point_id = str(uuid.uuid4())  # Generate a unique UUID for each point

                    # Convert image to base64 for storage
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    vectors.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "type": "image",
                            "page": page_num + 1,
                            "content": metadata_text,
                            "file_name": pdf_file_name,
                            "image_index": img_index,
                            "image_data": img_str
                        }
                    ))
                    extracted_images_count += 1

            doc.close()

        except S3Error as e:
            st.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            st.error(f"Error processing file {pdf_file_name}: {e}")

    # Recreate Qdrant collection and store the new vectors in smaller batches
    recreate_qdrant_collection()

    # Batch size to keep the request payload below the limit (adjust as necessary)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name="manual_vectors", points=batch)
        except Exception as e:
            st.error(f"Error upserting batch {i // batch_size}: {e}")

    st.success(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files, including {extracted_images_count} images.")

# Function to perform semantic search in Qdrant
def semantic_search(query, top_k=5):
    query_vector = model.encode(query).tolist()
    search_result = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k
    )
    return search_result

# Function to generate response using OpenAI
def generate_response(query, context, images):
    image_descriptions = [f"Image on page {img.payload['page']}: {img.payload['content']}" for img in images if img.payload['type'] == 'image']
    image_context = "\n".join(image_descriptions)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures."},
                {"role": "user", "content": f"Context: {context}\n\nImage Context: {image_context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return ""

# Streamlit UI
st.title('PropulsionPro: Vectorization and Query System')

# Vectorize PDFs button
if st.button("Vectorize PDFs"):
    with st.spinner("Vectorizing all PDFs from Cloudflare R2 and saving in Qdrant..."):
        vectorize_pdfs()
        st.success("All PDFs have been successfully vectorized and saved in Qdrant!")

# Chat interface
st.subheader("Chat with PropulsionPro")
user_query = st.text_input("Enter your query about maintenance or overhaul procedures:")

if user_query:
    # Perform semantic search
    search_results = semantic_search(user_query)
    
    # Prepare context for OpenAI
    context = "\n".join([result.payload['content'] for result in search_results])
    
    # Generate response
    response = generate_response(user_query, context, search_results)
    
    if response:
        st.write("Response:")
        st.write(response)
        
        # Display associated images
        st.write("Associated Images:")
        for result in search_results:
            if result.payload['type'] == 'image':
                image_data = result.payload.get('image_data')
                if image_data:
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                        image = image.convert("RGB")  # Convert to RGB to ensure compatibility
                        st.image(image, caption=f"Image from {result.payload['file_name']}, Page {result.payload['page']}")
                    except Exception as e:
                        st.warning(f"Failed to decode image data for {result.payload['file_name']}, Page {result.payload['page']}: {e}")
                else:
                    st.write(f"Image data not found for {result.payload['file_name']}, Page {result.payload['page']}")

st.sidebar.markdown("""
## How to use the system:
1. The reference header image is already stored in the Git repository.
2. Click the "Vectorize PDFs" button to vectorize all the available PDFs (if not done already).
3. Enter your query about maintenance or overhaul procedures in the chat interface.
4. The system will provide a detailed response along with associated images.
""")
