import streamlit as st
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
import fitz  # PyMuPDF for PDF handling
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def initialize_qdrant():
    try:
        return QdrantClient(
            url=st.secrets["qdrant"]["url"],
            api_key=st.secrets["qdrant"]["api_key"]
        )
    except KeyError as e:
        st.error(f"Qdrant initialization failed: Missing secret key {e}")
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
    total_images = 0

    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            
            # Load PDF using PyMuPDF (fitz)
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
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

                # Extract and process images using PyMuPDF
                image_list = page.get_images(full=True)
                total_images += len(image_list)
                st.write(f"Found {len(image_list)} images on page {page_num + 1} of {pdf_file_name}")
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Use a more robust hashing method
                    image_hash = imagehash.average_hash(image)
                    
                    # Skip reference header image
                    if image_hash == reference_image_hash:
                        continue

                    # Create rich metadata for the image
                    metadata_text = f"Page {page_num + 1}\n"
                    metadata_text += f"Document Section: {pdf_file_name.split('_')[0]}\n"
                    metadata_text += f"Page Content: {text[:1000]}..."
                    metadata_text += f"\nImage Hash: {str(image_hash)}"
                    
                    # Add keywords based on document structure
                    if "piston" in text.lower():
                        metadata_text += "\nKeywords: piston, engine component"
                    elif "engine" in text.lower():
                        metadata_text += "\nKeywords: engine, diagram"

                    # Create image vector
                    image_vector = model.encode(metadata_text).tolist()

                    # Store image data and vector
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Add a check for minimum image size
                    if len(img_str) < 1000:  # Adjust this threshold as needed
                        st.warning(f"Skipping small image on page {page_num + 1} of {pdf_file_name}")
                        continue

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

    st.success(f"Successfully processed {len(vectors)} vectors from {len(pdf_file_names)} PDF files, including {total_images} images.")

def semantic_search(query, top_k=10):
    if model is None:
        st.error("Model is not loaded. Please load the model before proceeding.")
        return []
    
    if qdrant_client is None:
        st.error("Qdrant client is not initialized. Please initialize the Qdrant client before proceeding.")
        return []
    
    query_vector = model.encode(query).tolist()
    
    # First, search for text results
    text_results = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="text"))])
    )
    
    # Then, search for image results
    image_results = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="image"))])
    )
    
    # Combine results, prioritizing images from the same pages as relevant text
    relevant_pages = set((r.payload['file_name'], r.payload['page']) for r in text_results)
    prioritized_images = [img for img in image_results if (img.payload['file_name'], img.payload['page']) in relevant_pages]
    other_images = [img for img in image_results if (img.payload['file_name'], img.payload['page']) not in relevant_pages]
    
    combined_results = text_results + prioritized_images + other_images
    
    st.write(f"Semantic search returned {len(combined_results)} results ({len(text_results)} text, {len(prioritized_images)} prioritized images, {len(other_images)} other images).")
    st.write(f"Query: {query}")
    st.write("Search results:")
    for i, result in enumerate(combined
