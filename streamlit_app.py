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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
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

def load_phi3_model():
    model_id = "microsoft/Phi-3-vision-128k-instruct"

    # Load the model configuration with trust_remote_code explicitly set to True
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True  # Explicitly allow custom code execution
    )
    
    # Specify the device explicitly to avoid GPU-related issues
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model with the modified configuration and trust_remote_code enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True  # Allow custom code from the model repository
    ).to(device)
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True  # Ensure processor also allows custom code
    )

    return model, processor

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

def extract_images_from_pdf(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            image = Image.open(io.BytesIO(image_bytes))
            images.append((f"Page {page_num + 1}, Image {img_index + 1}", image))

    doc.close()
    return images

def process_with_phi3(model, processor, image, prompt):
    prompt_template = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n" if image else f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    inputs = processor(prompt_template, [image] if image else None, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=False)
    
    output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

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

    phi3_model, phi3_processor = load_phi3_model()

    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            
            # Extract images using PyMuPDF
            extracted_images = extract_images_from_pdf(pdf_content)
            total_images += len(extracted_images)

            # Process text
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_vector = process_with_phi3(phi3_model, phi3_processor, None, f"Summarize this text: {text}")
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

            # Process images
            for img_info, image in extracted_images:
                image_hash = imagehash.average_hash(image)
                
                if image_hash == reference_image_hash:
                    continue

                metadata_text = f"{img_info}\n"
                metadata_text += f"Document Section: {pdf_file_name.split('_')[0]}\n"
                metadata_text += f"Image Hash: {str(image_hash)}"

                image_vector = process_with_phi3(phi3_model, phi3_processor, image, "Describe this image in detail.")

                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                if len(img_str) < 1000:  # Adjust this threshold as needed
                    st.warning(f"Skipping small image: {img_info} from {pdf_file_name}")
                    continue

                vectors.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=model.encode(image_vector).tolist(),
                    payload={
                        "type": "image",
                        "page": int(img_info.split(',')[0].split()[-1]),
                        "content": metadata_text,
                        "file_name": pdf_file_name,
                        "image_data": img_str,
                        "image_hash": str(image_hash)
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
        st.write(f"Result {i + 1}: {result.payload['type']} from file {result.payload['file_name']}, Page {result.payload['page']}")
        st.write(f"Content: {result.payload['content'][:100]}...")
        st.write(f"Score: {result.score}")

    return results

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
            st.write(f"Image {i+1} from {result.payload['file_name']}, Page {result.payload['page']}")
            if image_data:
                display_image(image_data, f"Image from {result.payload['file_name']}, Page {result.payload['page']}")
            else:
                st.write("Image data not found.")
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
