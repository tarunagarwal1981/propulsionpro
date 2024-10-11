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
    extracted_images_count = 0

    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract and process text
                text = page.get_text()
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if len(sentence.strip()) > 0:
                        embedding = model.encode(sentence.strip()).tolist()
                        vectors.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "type": "text",
                                "page": page_num + 1,
                                "content": sentence.strip(),
                                "file_name": pdf_file_name,
                            }
                        ))

                # Extract and process images
                image_list = page.get_images(full=True)
                st.write(f"Found {len(image_list)} images on page {page_num + 1} of {pdf_file_name}")
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Skip reference header image
                    image_hash = imagehash.phash(image)
                    if image_hash == reference_image_hash:
                        continue

                    # Create rich metadata for the image
                    metadata_text = f"Page {page_num + 1}\n"
                    metadata_text += f"Page Content: {text[:500]}..."  # Include more context
                    metadata_text += f"\nImage Hash: {str(image_hash)}"

                    # Create image vector
                    image_vector = model.encode(metadata_text).tolist()

                    # Store image data and vector
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    vectors.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=image_vector,
                        payload={
                            "type": "image",
                            "page": page_num + 1,
                            "content": metadata_text,
                            "file_name": pdf_file_name,
                            "image_data": img_str,
                            "image_hash": str(image_hash)
                        }
                    ))
                    extracted_images_count += 1

                st.write(f"Processed page {page_num + 1} of {pdf_file_name}")

            doc.close()

        except S3Error as e:
            st.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            st.error(f"Error processing file {pdf_file_name}: {str(e)}")

    recreate_qdrant_collection()

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            qdrant_client.upsert(collection_name="manual_vectors", points=batch)
            st.write(f"Successfully upserted batch {i // batch_size + 1}")
        except Exception as e:
            st.error(f"Error upserting batch {i // batch_size}: {e}")

    st.success(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files, including {extracted_images_count} images.")

def semantic_search(query, top_k=10):
    query_vector = model.encode(query).tolist()
    
    # Perform a single search for both text and images
    results = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k
    )
    
    # Separate text and image results
    text_results = [r for r in results if r.payload['type'] == 'text']
    image_results = [r for r in results if r.payload['type'] == 'image']
    
    st.write(f"Semantic search returned {len(results)} results ({len(text_results)} text, {len(image_results)} images).")
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

            for result in image_results:
                image_data = result.payload.get('image_data')
                st.write(f"Image from {result.payload['file_name']}, Page {result.payload['page']}")
                st.write(f"Image content: {result.payload['content'][:100]}...")
                st.write(f"Image hash: {result.payload.get('image_hash', 'N/A')}")
                st.write(f"Length of image data: {len(image_data) if image_data else 'No data'}")
                if image_data:
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                        image = image.convert("RGB")
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
    - Verify that the semantic search is finding relevant images for the query.

    For any persistent issues, please contact the system administrator.
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
