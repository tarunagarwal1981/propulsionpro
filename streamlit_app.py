import streamlit as st
import io
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
import torch
from pdf2image import convert_from_bytes

# Streamlit configuration
st.set_page_config(page_title="PropulsionPro", page_icon="🚢", layout="wide")

# Function to get API keys from environment variables
def get_api_key(key_name):
    api_key = os.getenv(key_name)
    if api_key is None:
        raise ValueError(f"{key_name} not found. Set it as an environment variable.")
    return api_key

# Initialize components
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_qdrant():
    try:
        qdrant_url = get_api_key('QDRANT_URL')
        qdrant_api_key = get_api_key('QDRANT_API_KEY')
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    except ValueError as e:
        st.error(f"Qdrant initialization failed: {e}")
        return None

# PDF processing functions
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text

def extract_images_from_pdf(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    return images

# Vector operations
def vectorize_text(model, text):
    return model.encode(text).tolist()

def save_to_qdrant(qdrant_client, collection_name, vectors):
    if not qdrant_client.get_collection(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(vectors[0].vector), distance="Cosine")
        )
    qdrant_client.upsert(collection_name=collection_name, points=vectors)

# RAG pipeline
def semantic_search(qdrant_client, collection_name, query_vector, top_k=5):
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

def generate_response(query, context):
    openai.api_key = get_api_key('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message['content']

# Streamlit UI
st.title('PropulsionPro: Marine Engine Maintenance Assistant')

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Process the PDF
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        images = extract_images_from_pdf(uploaded_file)

        # Vectorize text
        model = load_sentence_transformer_model()
        text_vector = vectorize_text(model, text)

        # Save to Qdrant
        qdrant_client = initialize_qdrant()
        if qdrant_client:
            vector_id = str(uuid.uuid4())
            point = PointStruct(
                id=vector_id,
                vector=text_vector,
                payload={"text": text, "file_name": uploaded_file.name}
            )
            save_to_qdrant(qdrant_client, "manual_vectors", [point])
            st.success("PDF processed and saved to vector database!")

    # Display extracted images
    st.subheader("Extracted Images")
    for i, img in enumerate(images):
        st.image(img, caption=f"Page {i+1}", use_column_width=True)

# User query
user_query = st.text_input("Ask a question about marine engine maintenance:")

if user_query:
    with st.spinner("Searching for relevant information..."):
        query_vector = vectorize_text(model, user_query)
        search_results = semantic_search(qdrant_client, "manual_vectors", query_vector)

        context = "\n".join([result.payload['text'] for result in search_results])
        response = generate_response(user_query, context)

        st.subheader("Response:")
        st.write(response)

        st.subheader("Relevant Images:")
        for result in search_results:
            if 'image' in result.payload:
                st.image(result.payload['image'], caption="Relevant image", use_column_width=True)

# Sidebar with instructions
st.sidebar.title("How to use PropulsionPro")
st.sidebar.markdown("""
1. Upload a PDF manual using the file uploader.
2. Wait for the PDF to be processed and saved to the vector database.
3. Ask a question about marine engine maintenance in the text input field.
4. Review the AI-generated response and any relevant images.
""")
