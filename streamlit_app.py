import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import io
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Filter, FieldCondition, MatchValue
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

                # Extract page heading (assuming it's the first line of text)
                page_heading = sentences[0] if sentences else ""

                # Vectorize text chunks
                for sentence in sentences:
                    if len(sentence.strip()) > 0:
                        embedding = model.encode(sentence.strip()).tolist()
                        point_id = str(uuid.uuid4())
                        vectors.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "text",
                                "page": page_num + 1,
                                "content": sentence.strip(),
                                "file_name": pdf_file_name,
                                "page_heading": page_heading
                            }
                        ))

                # Extract and vectorize images
                image_list = page.get_images(full=True)
                st.write(f"Found {len(image_list)} images on page {page_num + 1} of {pdf_file_name}")
                
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

                    # Get nearby text from the page (expanded)
                    bbox = img[1]
                    try:
                        extended_rect = fitz.Rect(bbox).expand(100)  # Increased expansion
                        nearby_text = page.get_text("text", clip=extended_rect)
                    except Exception:
                        nearby_text = "No nearby text found"

                    # Vectorize image with metadata (nearby text, OCR text, page heading)
                    metadata_text = f"Page {page_num + 1} - {page_heading}\nImage OCR text: {image_text}\nNearby text: {nearby_text}"
                    embedding = model.encode(metadata_text).tolist()
                    point_id = str(uuid.uuid4())

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
                            "image_data": img_str,
                            "page_heading": page_heading
                        }
                    ))
                    extracted_images_count += 1

                    # Debug: Print information about the extracted image
                    st.write(f"Extracted image {img_index + 1} from page {page_num + 1} of {pdf_file_name}")
                    st.write(f"Image OCR text: {image_text[:100]}...")
                    st.write(f"Nearby text: {nearby_text[:100]}...")
                    st.write(f"Sample of encoded image data: {img_str[:100]}")

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
            st.write(f"Successfully upserted batch {i // batch_size + 1}")
        except Exception as e:
            st.error(f"Error upserting batch {i // batch_size}: {e}")

    st.success(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files, including {extracted_images_count} images.")

# Function to perform semantic search in Qdrant
def semantic_search(query, top_k=5):
    query_vector = model.encode(query).tolist()
    
    # Search for text results
    text_results = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="text")
                )
            ]
        )
    )

    # Search for image results based on text results
    image_results = []
    for text_result in text_results:
        page_images = qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_vector,
            limit=2,  # Limit to 2 images per text result
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value="image")
                    ),
                    FieldCondition(
                        key="page",
                        match=MatchValue(value=text_result.payload["page"])
                    ),
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=text_result.payload["file_name"])
                    )
                ]
            )
        )
        image_results.extend(page_images)

    # Combine and deduplicate results
    all_results = text_results + image_results
    unique_results = list({r.id: r for r in all_results}.values())

    st.write(f"Semantic search returned {len(unique_results)} results ({len(text_results)} text, {len(image_results)} images).")
    st.write(f"Query: {query}")
    st.write("Search results:")
    for i, result in enumerate(unique_results):
        st.write(f"Result {i + 1}: {result.payload['type']} from file {result.payload['file_name']}, Page {result.payload['page']}")
        st.write(f"Content: {result.payload['content'][:100]}...")
        st.write(f"Score: {result.score}")

    return unique_results

# Function to generate response using OpenAI
def generate_response(query, context, images):
    image_descriptions = [f"Image on page {img.payload['page']} of {img.payload['file_name']}: {img.payload['content']}" for img in images if img.payload['type'] == 'image']
    image_context = "\n".join(image_descriptions)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures. When relevant, refer to the images provided in the context."},
                {"role": "user", "content": f"Context:\n{context}\n\nImage Context:\n{image_context}\n\nQuestion: {query}"}
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
    text_context = "\n".join([result.payload['content'] for result in search_results if result.payload['type'] == 'text'])
    image_results = [result for result in search_results if result.payload['type'] == 'image']
    
    # Generate response
    response = generate_response(user_query, text_context, image_results)
    
    if response:
        st.write("Response:")
        st.write(response)
        
        # Display associated images
        st.write("Associated Images:")
        st.write(f"Number of image results: {len(image_results)}")

        if len(image_results) == 0:
            st.write("No image results found. This could be due to:")
            st.write("1. No images were extracted during vectorization.")
            st.write("2. The semantic search didn't find relevant images for the query.")
            st.write("3. Issues with image metadata or labeling.")

        for result in image_results:
            image_data = result.payload.get('image_data')
            st.write(f"Image from {result.payload['file_name']}, Page {result.payload['page']}")
            st.write(f"Image content: {result.payload['content'][:100]}...")
            st.write(f"Length of image data: {len(image_data) if image_data else 'No data'}")
            if image_data:
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    image = image.convert("RGB")  # Convert to RGB to ensure compatibility
                    st.image(image, caption=f"Image from {result.payload['file_name']}, Page {result.payload['page']}")
                except Exception as e:
                    st.warning(f"Failed to decode image data: {str(e)}")
                    st.write(f"Image data (first 100 chars): {image_data[:100]}")
            else:
                st.write("Image data not found.")

st.sidebar.markdown("""
## How to use the system:
1. The reference header image is already stored in the Git repository.
2. Click the "Vectorize PDFs" button to vectorize all the available PDFs (if not done already).
3. Enter your query about maintenance or overhaul procedures in the chat interface.
4. The system will provide a detailed response along with associated images.
5. Review the response and any associated images for comprehensive information.

## Troubleshooting:
If you're not seeing images in the results:
- Ensure that the PDFs contain images.
- Check the vectorization output for any errors during image extraction.
- Verify that the semantic search is finding relevant image results.

For any persistent issues, please contact the system administrator.
""")

# Add a debug section
if st.checkbox("Show Debug Information"):
    st.subheader("Debug Information")
    st.write("Qdrant Collection Info:")
    try:
        collection_info = qdrant_client.get_collection("manual_vectors")
        st.json(collection_info.dict())
    except Exception as e:
        st.error(f"Failed to retrieve collection info: {str(e)}")

    st.write("Sample Vectors:")
    try:
        sample_vectors = qdrant_client.scroll(
            collection_name="manual_vectors",
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        st.json([point.dict() for point in sample_vectors[0]])
    except Exception as e:
        st.error(f"Failed to retrieve sample vectors: {str(e)}")

# Main execution
if __name__ == "__main__":
    st.set_page_config(page_title="PropulsionPro", page_icon="🚢", layout="wide")
    main()
