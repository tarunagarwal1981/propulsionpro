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

# Function to recreate the Qdrant collection
def recreate_qdrant_collection():
    try:
        qdrant_client.delete_collection(collection_name="manual_vectors")
    except Exception as e:
        st.warning(f"Failed to delete existing collection: {e}")

    # Create or recreate the Qdrant collection with proper configuration
    qdrant_client.create_collection(
        collection_name="manual_vectors",
        vectors_config={"manual_vectors": VectorParams(
            size=384,  # Size of the vector (embedding dimension)
            distance="Cosine"
        )}
    )

# Function to extract text and images from PDFs, vectorize, and store in Qdrant
def vectorize_pdfs():
    if not minio_client:
        st.error("MinIO client not initialized.")
        return
    
    # Get list of all PDFs from Cloudflare R2
    try:
        objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"], recursive=True)
        pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    except S3Error as e:
        st.error(f"Error listing PDF files from Cloudflare R2: {e}")
        return

    vectors = []
    current_id = 0

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
                        vectors.append(PointStruct(
                            id=str(current_id),
                            vector=embedding,
                            payload={
                                "type": "text",
                                "page": page_num + 1,
                                "content": sentence.strip(),
                                "file_name": pdf_file_name
                            }
                        ))
                        current_id += 1

                # Extract and vectorize images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    # Perform OCR on the image
                    try:
                        image_text = pytesseract.image_to_string(image)
                    except Exception:
                        image_text = "OCR failed for this image"

                    # Get nearby text from the page
                    bbox = img[1]
                    try:
                        extended_rect = fitz.Rect(bbox).extend(50)
                        nearby_text = page.get_text("text", clip=extended_rect)
                    except Exception:
                        nearby_text = "No nearby text found"

                    # Vectorize image with metadata (nearby and OCR text)
                    metadata_text = f"Image OCR text: {image_text}. Nearby text: {nearby_text}."
                    embedding = model.encode(metadata_text).tolist()

                    vectors.append(PointStruct(
                        id=str(current_id),
                        vector=embedding,
                        payload={
                            "type": "image",
                            "page": page_num + 1,
                            "content": metadata_text,
                            "file_name": pdf_file_name,
                            "image_index": img_index
                        }
                    ))
                    current_id += 1

            doc.close()

        except S3Error as e:
            st.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            st.error(f"Error processing file {pdf_file_name}: {e}")

    # Recreate Qdrant collection and store the new vectors
    recreate_qdrant_collection()
    qdrant_client.upsert(collection_name="manual_vectors", points=vectors)

# Streamlit UI
st.title('PropulsionPro: Vectorization and Query System')

if st.button("Vectorize PDFs"):
    with st.spinner("Vectorizing all PDFs from Cloudflare R2 and saving in Qdrant..."):
        vectorize_pdfs()
        st.success("All PDFs have been successfully vectorized and saved in Qdrant!")

st.sidebar.markdown("""
## How to use the system:
1. Click the "Vectorize PDFs" button to vectorize all the available PDFs.
2. The PDFs will be vectorized, and both text and images will be stored in Qdrant with metadata.
3. The vectors are replaced each time the button is clicked.
""")
