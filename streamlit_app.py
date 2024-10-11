import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import io
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Filter, FieldCondition, MatchValue
import re
import imagehash
import uuid
import os
import openai
import base64

# Set page config at the very beginning
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
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

def recreate_qdrant_collection():
    try:
        qdrant_client.delete_collection(collection_name="manual_vectors")
    except Exception as e:
        st.warning(f"Failed to delete existing collection: {e}")

    qdrant_client.create_collection(
        collection_name="manual_vectors",
        vectors_config=VectorParams(
            size=384,
            distance="Cosine"
        )
    )

def vectorize_pdfs():
    # ... (previous vectorize_pdfs code remains the same)
    # Make sure to keep the improvements for image metadata and logging

def semantic_search(query, top_k=10):
    # ... (previous semantic_search code remains the same)

def generate_response(query, context, images):
    # ... (previous generate_response code remains the same)

def display_image(image_data, caption):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = image.convert("RGB")
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.warning(f"Failed to display image: {str(e)}")
        st.write(f"Image data (first 100 chars): {image_data[:100]}")

# Main function
def main():
    st.title('PropulsionPro: Vectorization and Query System')

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
            st.write(f"Number of image results: {len(image_results)}")

            if len(image_results) == 0:
                st.write("No image results found. This could be due to:")
                st.write("1. No images were extracted during vectorization.")
                st.write("2. The semantic search didn't find relevant images for the query.")
                st.write("3. Issues with image metadata or labeling.")

            for i, result in enumerate(image_results):
                image_data = result.payload.get('image_data')
                st.write(f"Image {i+1} from {result.payload['file_name']}, Page {result.payload['page']}")
                st.write(f"Image content: {result.payload['content'][:100]}...")
                st.write(f"Image hash: {result.payload.get('image_hash', 'N/A')}")
                st.write(f"Length of image data: {len(image_data) if image_data else 'No data'}")
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

    ## Troubleshooting:
    If you're not seeing images in the results:
    - Ensure that the PDFs contain images.
    - Check the vectorization output for any errors during image extraction.
    - Verify that the semantic search is finding relevant images for the query.

    For any persistent issues, contact the system administrator.
    """)

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

# Initialization
try:
    openai.api_key = get_api_key()
except ValueError as e:
    st.error(str(e))

model = load_model()
minio_client = initialize_minio()
qdrant_client = QdrantClient(
    url=st.secrets["qdrant"]["url"],
    api_key=st.secrets["qdrant"]["api_key"]
)

reference_image_path = "assets/header_image.png"
try:
    reference_image = Image.open(reference_image_path)
    reference_image_hash = imagehash.phash(reference_image)
except FileNotFoundError:
    st.error(f"Reference header image not found at {reference_image_path}. Please ensure it is available.")
    reference_image_hash = None

# Run the main function
if __name__ == "__main__":
    main()

# Add a footer
st.markdown("""
---
Created by Your Company Name | © 2023 All Rights Reserved
""")

# Optional: Add some CSS to improve the app's appearance
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.stButton>button {
    width: 100%;
}
.streamlit-expanderHeader {
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# End of the application
