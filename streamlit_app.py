import streamlit as st
import io
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import imagehash
import uuid
import os
import openai
import base64
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Set page config
st.set_page_config(page_title="PropulsionPro", page_icon="🚢", layout="wide")

# Function definitions
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_phi2_model():
    model_id = "microsoft/phi-2"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer

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

def initialize_qdrant():
    try:
        return QdrantClient(
            url=st.secrets["qdrant"]["url"],
            api_key=st.secrets["qdrant"]["api_key"]
        )
    except KeyError as e:
        st.error(f"Qdrant initialization failed: Missing secret key {e}")
        return None

def get_nearby_text(page, rect, max_chars=500):
    words = page.get_text("words")
    nearby_words = [w[4] for w in words if fitz.Rect(w[:4]).intersects(rect)]
    nearby_text = " ".join(nearby_words)
    if len(nearby_text) > max_chars:
        nearby_text = nearby_text[:max_chars] + "..."
    return nearby_text

def extract_images_from_pdf(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        # Extract page heading (assuming it's the first line of text on the page)
        page_text = page.get_text()
        page_heading = page_text.split('\n')[0] if page_text else f"Page {page_num + 1}"
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get nearby text
            rect = page.get_image_bbox(img)
            nearby_text = get_nearby_text(page, rect)
            
            images.append({
                "page_num": page_num + 1,
                "image_index": img_index + 1,
                "image": image,
                "nearby_text": nearby_text,
                "page_heading": page_heading
            })
    doc.close()
    return images

def process_with_phi2(model, tokenizer, text, max_length=500):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length, temperature=0.7, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def vectorize_pdfs():
    if not minio_client:
        st.error("MinIO client not initialized.")
        return
    if reference_image_hash is None:
        st.error("Reference header image hash could not be calculated.")
        return
    
    try:
        objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"], recursive=True)
        pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    except S3Error as e:
        st.error(f"Error listing PDF files from Cloudflare R2: {e}")
        return

    vectors = []
    total_images = 0
    phi2_model, phi2_tokenizer = load_phi2_model()

    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            
            extracted_images = extract_images_from_pdf(pdf_content)
            total_images += len(extracted_images)

            doc = fitz.open(stream=pdf_content, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_vector = process_with_phi2(phi2_model, phi2_tokenizer, f"Summarize this text: {text}")
                vectors.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=model.encode(text_vector).tolist(),
                    payload={
                        "type": "text",
                        "page": page_num + 1,
                        "content": text,
                        "file_name": pdf_file_name,
                    }
                ))

            for img_data in extracted_images:
                image_hash = imagehash.average_hash(img_data["image"])
                if image_hash == reference_image_hash:
                    continue

                metadata_text = f"Page {img_data['page_num']}, Image {img_data['image_index']}\n"
                metadata_text += f"Page Heading: {img_data['page_heading']}\n"
                metadata_text += f"Document: {pdf_file_name}\n"
                metadata_text += f"Nearby Text: {img_data['nearby_text']}\n"
                metadata_text += f"Image Hash: {str(image_hash)}"

                image_vector = process_with_phi2(phi2_model, phi2_tokenizer, f"Describe this image and its context: {metadata_text}")

                buffered = io.BytesIO()
                img_data["image"].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                if len(img_str) < 1000:
                    st.warning(f"Skipping small image: Page {img_data['page_num']}, Image {img_data['image_index']} from {pdf_file_name}")
                    continue

                vectors.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=model.encode(image_vector).tolist(),
                    payload={
                        "type": "image",
                        "page": img_data['page_num'],
                        "image_index": img_data['image_index'],
                        "content": metadata_text,
                        "file_name": pdf_file_name,
                        "image_data": img_str,
                        "image_hash": str(image_hash),
                        "page_heading": img_data['page_heading']
                    }
                ))

            doc.close()

        except S3Error as e:
            st.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            st.error(f"Error processing file {pdf_file_name}: {str(e)}")

    qdrant_client.recreate_collection(
        collection_name="manual_vectors",
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance="Cosine")
    )

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name="manual_vectors", points=batch)
            st.write(f"Successfully upserted batch {i // batch_size + 1}")
        except Exception as e:
            st.error(f"Error upserting batch {i // batch_size}: {e}")

    st.success(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files, including {total_images} images.")

def semantic_search(query, top_k=10):
    query_vector = model.encode(query).tolist()
    results = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k
    )
    
    st.write(f"Semantic search returned {len(results)} results.")
    st.write(f"Query: {query}")
    st.write("Search results:")
    for i, result in enumerate(results):
        if result.payload['type'] == 'text':
            st.write(f"Result {i + 1}: text from file {result.payload['file_name']}, Page {result.payload['page']}")
            st.write(f"Content: {result.payload['content'][:100]}...")
        else:
            st.write(f"Result {i + 1}: image from file {result.payload['file_name']}, Page {result.payload['page']}, Image {result.payload['image_index']}")
            st.write(f"Page Heading: {result.payload['page_heading']}")
            st.write(f"Content: {result.payload['content'][:100]}...")
        st.write(f"Score: {result.score}")

    return results

def generate_response(query, context, images):
    image_descriptions = []
    for img in images:
        if img.payload['type'] == 'image':
            desc = f"Image on page {img.payload['page']} of {img.payload['file_name']}, Image {img.payload['image_index']}:\n"
            desc += f"Page Heading: {img.payload['page_heading']}\n"
            desc += f"Content: {img.payload['content'][:500]}..."  # Truncate if too long
            image_descriptions.append(desc)
    
    image_context = "\n\n".join(image_descriptions)

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

def display_image(image_data, caption):
    try:
        decoded_image = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded_image))
        image = image.convert("RGB")
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to display image: {str(e)}")

# Main Streamlit UI
st.title('PropulsionPro: Vectorization and Query System')

# Initialize components
try:
    openai.api_key = get_api_key()
except ValueError as e:
    st.error(str(e))

model = load_sentence_transformer_model()
minio_client = initialize_minio()
qdrant_client = initialize_qdrant()

# Reference image hash
reference_image_path = "assets/header_image.png"
try:
    reference_image = Image.open(reference_image_path)
    reference_image_hash = imagehash.average_hash(reference_image)
except FileNotFoundError:
    st.error(f"Reference header image not found at {reference_image_path}. Please ensure it is available.")
    reference_image_hash = None

if st.button("Vectorize PDFs"):
    with st.spinner("Vectorizing all PDFs from Cloudflare R2 and saving in Qdrant..."):
        vectorize_pdfs()
        st.success("All PDFs have been successfully vectorized and saved in Qdrant!")

st.subheader("Chat with PropulsionPro")
user_query = st.text_input("Enter your query about maintenance or overhaul procedures:")

if user_query:
    search_results = semantic_search(user_query)
    
    text_context = "\n".join([result.payload['content'] for result in search_results if result.payload['type'] == 'text'])
    image_results = [result for result in search_results if result.payload['type'] == 'image']
    
    response = generate_response(user_query, text_context, image_results)
    
    if response:
        st.write("Response:")
        st.write(response)
        
        st.write("Associated Images:")
        for i, result in enumerate(image_results):
            image_data = result.payload.get('image_data')
            st.write(f"Image {i+1} from {result.payload['file_name']}, Page {result.payload['page']}, Image {result.payload['image_index']}")
            st.write(f"Page Heading: {result.payload['page_heading']}")
            if image_data:
                display_image(image_data, f"Image from {result.payload['file_name']}, Page {result.payload['page']}, Image {result.payload['image_index']}")
            else:
                st.write("Image data not found.")
            st.write(f"Context: {result.payload['content'][:500]}...")  # Display truncated context
            st.write("---")

st.sidebar.markdown("""
## How to use the system:
1. The reference header image is already stored in the Git repository.
2. Click the "Vectorize PDFs" button to vectorize all the available PDFs (if not done already).
3. Enter your query about maintenance or overhaul procedures in the chat interface.
4. The system will provide a detailed response along with associated images.
5. Review the response and any associated images for comprehensive information.

For any persistent issues, contact the system administrator.
""")
