import streamlit as st
import fitz  # PyMuPDF
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import re
import uuid
import cv2
import numpy as np
import openai
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error(f"MinIO initialization failed: Missing secret key {e}")
        return None

minio_client = initialize_minio()

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=st.secrets["qdrant"]["url"],
        api_key=st.secrets["qdrant"]["api_key"]
    )
except KeyError as e:
    logger.error(f"Qdrant client initialization failed: Missing secret key {e}")
    qdrant_client = None

# Function to recreate the Qdrant collection
def recreate_qdrant_collection():
    if not qdrant_client:
        logger.error("Qdrant client not initialized.")
        return False
    try:
        qdrant_client.delete_collection(collection_name="manual_vectors")
        logger.info("Existing collection deleted.")
    except Exception as e:
        logger.warning(f"Failed to delete existing collection: {e}")

    try:
        qdrant_client.create_collection(
            collection_name="manual_vectors",
            vectors_config=VectorParams(size=384, distance="Cosine")
        )
        logger.info("New collection created.")
        return True
    except Exception as e:
        logger.error(f"Failed to create new collection: {e}")
        return False

def extract_images_from_page(page, page_num):
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 100 and h > 100:
            roi = img_np[y:y+h, x:x+w]
            pil_img = Image.fromarray(roi)
            images.append((f"Page {page_num + 1}, Image {i + 1}", pil_img, (x, y, w, h)))
    
    return images

def vectorize_pdfs():
    if not minio_client or not qdrant_client:
        logger.error("MinIO or Qdrant client not initialized.")
        return False

    try:
        objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"], recursive=True)
        pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    except S3Error as e:
        logger.error(f"Error listing PDF files from Cloudflare R2: {e}")
        return False

    vectors = []
    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1} of {pdf_file_name}...")
                page = doc[page_num]
                
                # Process text
                text = page.get_text()
                sentences = re.split(r'(?<=[.!?])\s+', text)
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
                                "file_name": pdf_file_name
                            }
                        ))

                # Process images
                images = extract_images_from_page(page, page_num)
                for img_name, img, bbox in images:
                    x, y, w, h = bbox
                    try:
                        extended_rect = fitz.Rect(x/2, y/2, (x+w)/2, (y+h)/2).extend(50)
                        nearby_text = page.get_text("text", clip=extended_rect)
                    except Exception:
                        nearby_text = "No nearby text found"

                    metadata_text = f"Image metadata: {nearby_text}"
                    embedding = model.encode(metadata_text).tolist()
                    point_id = str(uuid.uuid4())
                    vectors.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "type": "image",
                            "page": page_num + 1,
                            "content": metadata_text,
                            "file_name": pdf_file_name,
                            "image_name": img_name
                        }
                    ))

            doc.close()

        except S3Error as e:
            logger.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            logger.error(f"Error processing file {pdf_file_name}: {e}")

    if not recreate_qdrant_collection():
        return False

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name="manual_vectors", points=batch)
        except Exception as e:
            logger.error(f"Error upserting batch {i // batch_size}: {e}")
            return False

    logger.info(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files.")
    return True

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        logger.error("API key not found. Set OPENAI_API_KEY as an environment variable.")
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

def rag_pipeline(question):
    if not qdrant_client:
        logger.error("Qdrant client not initialized.")
        return "Error: Qdrant client not initialized.", []

    query_embedding = model.encode(question).tolist()
    search_result = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_embedding,
        limit=10
    )
    
    context = ""
    relevant_images = []
    for result in search_result:
        payload = result.payload
        if payload["type"] == "text":
            context += f"From {payload['file_name']}, page {payload['page']}: {payload['content']}\n"
        elif payload["type"] == "image":
            relevant_images.append(payload)
            context += f"Image found in {payload['file_name']}, page {payload['page']}: {payload['content']}\n"

    logger.info(f"Context being sent to LLM:\n{context}")

    try:
        openai.api_key = get_api_key()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based solely on the provided context. If the context doesn't contain relevant information, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question using only the information provided in the context above. If the context doesn't contain relevant information to answer the question, please state that."}
            ]
        )
        
        answer = response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        answer = "Sorry, there was an error processing your question."

    return answer, relevant_images

# Streamlit UI
st.title('Advanced PDF Extractor and Vectorizer')

if st.button("Process PDFs from Cloudflare R2"):
    with st.spinner("Processing PDFs from Cloudflare R2, extracting content, and saving in Qdrant..."):
        if vectorize_pdfs():
            st.success("All PDFs have been successfully processed and vectors saved in Qdrant!")
        else:
            st.error("There was an error processing the PDFs. Please check the logs for more information.")

st.sidebar.markdown("""
## How to use the system:
1. Click the "Process PDFs from Cloudflare R2" button to start processing all available PDFs in Cloudflare R2.
2. The system will extract text and images from each PDF using advanced image processing techniques.
3. Text will be split into sentences and vectorized.
4. Images will be extracted using contour detection and vectorized along with their nearby text as metadata.
5. All vectors will be stored in Qdrant, replacing any existing vectors.
6. You'll see a success message when the process is complete.
7. Use the question answering section below to query the processed documents.
""")

# RAG Pipeline User Interface
st.title('RAG Pipeline: Question Answering')
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Fetching answer..."):
            answer, images = rag_pipeline(question)
            st.write("**Answer:**", answer)
            
            if images:
                st.write("**Relevant Images:**")
                for img_data in images:
                    st.write(f"- {img_data['image_name']} from {img_data['file_name']}, page {img_data['page']}")
            
            # Display debug information
            with st.expander("Debug Information"):
                st.write(f"Number of relevant images: {len(images)}")
                st.write("Context used (first 500 characters):")
                st.write(answer[:500] + "..." if len(answer) > 500 else answer)
    else:
        st.error("Please enter a question.")
