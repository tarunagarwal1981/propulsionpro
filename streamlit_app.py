import streamlit as st
import fitz  # PyMuPDF
import io
import os
from minio import Minio
from minio.error import S3Error
from PIL import Image
import openai
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pytesseract
from io import BytesIO

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

# Lazy load SentenceTransformer model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Initialize MinIO client (for R2)
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

if minio_client:
    try:
        BUCKET_NAME = st.secrets["R2_BUCKET_NAME"]
    except KeyError:
        st.error("MinIO initialization failed: Missing bucket name")

# Function to get PDF from R2
def get_pdf_from_r2(file_name):
    try:
        response = minio_client.get_object(BUCKET_NAME, file_name)
        return response.read()
    except S3Error as exc:
        st.error(f"Error downloading file from R2: {exc}")
        return None

# Extract text and images from PDF
def extract_content_with_metadata(pdf_content):
    text_content = []
    images = []
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_content.append({
                'page': page_num + 1,
                'content': text
            })
            
            # Get all images on the page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Get image rectangle (bbox) directly from the image info
                bbox = img[1]
                
                # Extract text near the image
                try:
                    extended_rect = fitz.Rect(bbox).extend(50)
                    nearby_text = page.get_text("text", clip=extended_rect)
                except Exception:
                    nearby_text = page.get_text("text")
                
                # Use OCR to extract text from the image
                try:
                    image_text = pytesseract.image_to_string(image)
                except Exception:
                    image_text = "OCR failed for this image"
                
                images.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'image': image,
                    'nearby_text': nearby_text,
                    'image_text': image_text
                })
    except Exception as e:
        st.error(f"Failed to extract content: {e}")
    return text_content, images

# Chunk text into smaller parts
def chunk_text_with_metadata(text_content, chunk_size=500):
    chunks = []
    for page_data in text_content:
        page_num = page_data['page']
        text = page_data['content']
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        for sentence in sentences:
            if current_size + len(sentence) <= chunk_size:
                current_chunk.append(sentence)
                current_size += len(sentence)
            else:
                chunks.append({
                    'page': page_num,
                    'content': ' '.join(current_chunk)
                })
                current_chunk = [sentence]
                current_size = len(sentence)
        if current_chunk:
            chunks.append({
                'page': page_num,
                'content': ' '.join(current_chunk)
            })
    return chunks

# Vectorize chunks
def vectorize_chunks(chunks):
    if model is not None:
        return [model.encode(chunk['content']) for chunk in chunks]
    return []

# Find most relevant chunk based on query
def find_most_relevant_chunk(query, chunks, chunk_vectors):
    query_vector = model.encode(query)
    similarities = cosine_similarity([query_vector], chunk_vectors)[0]
    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index], similarities[most_relevant_index]

# Generate an answer using OpenAI
def answer_query(query, context, images):
    image_descriptions = [f"Image on page {img['page']}: {img['image_text']}" for img in images if img['page'] == context['page']]
    image_context = "\n".join(image_descriptions)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures."},
                {"role": "user", "content": f"Context: {context['content']}\n\nImage Context: {image_context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return ""

# Streamlit UI
st.title('PropulsionPro: Marine Engine Manual Query System')

if minio_client:
    try:
        objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
        pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    except S3Error as exc:
        st.error(f"Error listing objects in R2: {exc}")
        pdf_file_names = []

    selected_file = st.selectbox('Select a PDF manual:', pdf_file_names)

    if selected_file:
        pdf_content = get_pdf_from_r2(selected_file)
        if pdf_content:
            st.success(f'Manual "{selected_file}" successfully loaded from R2!')
            text_content, images = extract_content_with_metadata(pdf_content)
            chunks = chunk_text_with_metadata(text_content)
            chunk_vectors = vectorize_chunks(chunks)

            query = st.text_input('Enter your maintenance query:')
            if query:
                with st.spinner('Processing your query...'):
                    relevant_chunk, similarity_score = find_most_relevant_chunk(query, chunks, chunk_vectors)
                    response = answer_query(query, relevant_chunk, images)
                st.subheader('Answer:')
                st.write(response)
                st.write(f"Confidence: {similarity_score:.2f}")

                st.subheader('Related Images:')
                relevant_images = [img for img in images if img['page'] == relevant_chunk['page']]
                for img in relevant_images:
                    st.image(img['image'], caption=f"Image on page {img['page']}", use_column_width=True)
                    st.write(f"Nearby text: {img['nearby_text'][:200]}...")
                    st.write(f"Image text: {img['image_text']}")

st.sidebar.markdown("""
## How to use PropulsionPro:
1. Select a PDF manual from the dropdown.
2. Wait for the manual to load and process.
3. Enter your maintenance query in the search box.
4. View the answer, confidence score, and related images below.
""")
