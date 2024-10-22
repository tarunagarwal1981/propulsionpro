import streamlit as st
import fitz  # PyMuPDF
from minio import Minio
from minio.error import S3Error
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Filter, FieldCondition, Range
import re
import uuid
import cv2
import numpy as np
import openai
import os
import io
import base64
import logging
import random
import gc
import multiprocessing
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NLTK data path to a writable directory
nltk.data.path.append('/tmp/nltk_data')

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True, raise_on_error=True, download_dir='/tmp/nltk_data')
        nltk.download('averaged_perceptron_tagger', quiet=True, raise_on_error=True, download_dir='/tmp/nltk_data')
        nltk.download('maxent_ne_chunker', quiet=True, raise_on_error=True, download_dir='/tmp/nltk_data')
        nltk.download('words', quiet=True, raise_on_error=True, download_dir='/tmp/nltk_data')
        return True
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")
        return False

nltk_data_available = download_nltk_data()

# Load the embedding model (cached to avoid reloading on every app refresh)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Initialize NER pipeline
@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

ner_model = load_ner_model()

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
    try:
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
    except Exception as e:
        logger.error(f"Error extracting images from page {page_num}: {e}")
        return []

def extract_text_around_image(page, bbox, margin=50):
    x, y, w, h = bbox
    extended_rect = fitz.Rect(x/2-margin, y/2-margin, (x+w)/2+margin, (y+h)/2+margin)
    return page.get_text("text", clip=extended_rect)

def process_text(text):
    if nltk_data_available:
        try:
            sentences = nltk.sent_tokenize(text)
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            entities = [word for word, pos in tagged if pos in ['NNP', 'NNPS']]
            
            # Use the NER model for more accurate entity recognition
            ner_results = ner_model(text)
            named_entities = [entity['word'] for entity in ner_results if entity['entity'] != 'O']
            
            entities = list(set(entities + named_entities))  # Combine and deduplicate entities
        except Exception as e:
            logger.warning(f"Error in NLTK processing: {e}. Falling back to basic processing.")
            return basic_process_text(text)
    else:
        return basic_process_text(text)
    
    return {
        'full_text': text,
        'sentences': sentences,
        'entities': entities
    }

def basic_process_text(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    words = text.split()
    entities = [word for word in words if word.istitle() and len(word) > 1]
    return {
        'full_text': text,
        'sentences': sentences,
        'entities': entities
    }

def compute_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def perform_topic_modeling(tfidf_matrix, num_topics=5):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    return lda

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

    chunk_size = 10
    total_vectors = 0
    total_images = 0

    if not recreate_qdrant_collection():
        return False

    for pdf_file_name in pdf_file_names:
        try:
            response = minio_client.get_object(st.secrets["R2_BUCKET_NAME"], pdf_file_name)
            pdf_content = response.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            st.write(f"Processing: {pdf_file_name}")

            all_text = []
            for chunk_start in range(0, len(doc), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(doc))
                vectors = []
                chunk_images = []

                for page_num in range(chunk_start, chunk_end):
                    try:
                        logger.info(f"Processing page {page_num + 1} of {pdf_file_name}...")
                        page = doc[page_num]
                        
                        text = page.get_text()
                        all_text.append(text)
                        processed_text = process_text(text)
                        
                        for sentence in processed_text['sentences']:
                            embedding = model.encode(sentence).tolist()
                            point_id = str(uuid.uuid4())
                            vectors.append(PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "type": "text",
                                    "page": page_num + 1,
                                    "content": sentence,
                                    "entities": processed_text['entities'],
                                    "file_name": pdf_file_name
                                }
                            ))

                        images = extract_images_from_page(page, page_num)
                        st.write(f"Page {page_num + 1}: {len(images)} images extracted")
                        for img_name, img, bbox in images:
                            surrounding_text = extract_text_around_image(page, bbox)
                            processed_surrounding_text = process_text(surrounding_text)
                            
                            metadata_text = f"Image metadata: {surrounding_text}"
                            embedding = model.encode(metadata_text).tolist()
                            point_id = str(uuid.uuid4())
                            
                            # Convert PIL Image to bytes
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            vectors.append(PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "type": "image",
                                    "page": page_num + 1,
                                    "content": metadata_text,
                                    "surrounding_text": surrounding_text,
                                    "entities": processed_surrounding_text['entities'],
                                    "file_name": pdf_file_name,
                                    "image_name": img_name,
                                    "image_data": base64.b64encode(img_byte_arr).decode('utf-8')
                                }
                            ))
                            chunk_images.append((img_name, img))

                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1} of {pdf_file_name}: {e}")
                        continue

                try:
                    tfidf_matrix, feature_names = compute_tfidf(all_text)
                    lda_model = perform_topic_modeling(tfidf_matrix)
                    
                    for vector in vectors:
                        if vector.payload["type"] == "text":
                            text_vector = tfidf_matrix[all_text.index(vector.payload["content"])]
                            topic_distribution = lda_model.transform(text_vector)[0]
                            vector.payload["topics"] = topic_distribution.tolist()
                except Exception as e:
                    logger.warning(f"Error in TF-IDF or topic modeling: {e}. Skipping this step.")

                try:
                    qdrant_client.upsert(collection_name="manual_vectors", points=vectors)
                    total_vectors += len(vectors)
                    total_images += len(chunk_images)
                except Exception as e:
                    logger.error(f"Error upserting vectors for chunk {chunk_start//chunk_size + 1}: {e}")

                st.write(f"Displaying sample images from pages {chunk_start + 1} to {chunk_end}:")
                display_images = chunk_images if len(chunk_images) <= 4 else random.sample(chunk_images, 4)
                cols = st.columns(4)
                for idx, (img_name, img) in enumerate(display_images):
                    with cols[idx % 4]:
                        st.image(img, caption=img_name, use_column_width=True)

                gc.collect()

            doc.close()

        except S3Error as e:
            logger.error(f"Error downloading file {pdf_file_name} from Cloudflare R2: {e}")
        except Exception as e:
            logger.error(f"Error processing file {pdf_file_name}: {e}")

    st.write(f"Total vectors processed: {total_vectors}")
    st.write(f"Total images extracted: {total_images}")

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
        limit=15,  # Increased from 10 to 15 for more context
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="page",
                    range=Range(
                        gte=1
                    )
                )
            ]
        )
    )
    
    context = ""
    relevant_images = []
    for result in search_result:
        payload = result.payload
        if payload["type"] == "text":
            context += f"From {payload['file_name']}, page {payload['page']}: {payload['content']}\n"
            context += f"Entities: {', '.join(payload['entities'])}\n"
            if 'topics' in payload:
                context += f"Topics: {payload['topics']}\n"
        elif payload["type"] == "image":
            relevant_images.append(payload)
            context += f"Image found in {payload['file_name']}, page {payload['page']}:\n"
            context += f"Surrounding text: {payload['surrounding_text']}\n"
            context += f"Entities: {', '.join(payload['entities'])}\n"

    logger.info(f"Context being sent to LLM:\n{context}")

    try:
        openai.api_key = get_api_key()
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Upgraded to GPT-4 for better comprehension
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in technical documentation. Your task is to answer questions based on the provided context. When answering, follow these guidelines:\n1. If the context contains step-by-step instructions, present them clearly.\n2. Reference specific images when they are relevant to the answer.\n3. If the context doesn't contain enough information to fully answer the question, state this clearly.\n4. Use the entities and topics information to provide a more comprehensive and contextual answer.\n5. Cite specific page numbers and document names when referencing information."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question using only the information provided in the context above. If the context doesn't contain relevant information to fully answer the question, please state that clearly. Consider the entities and topics mentioned in the context for a more comprehensive answer. Cite specific page numbers and document names when referencing information."}
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
        try:
            if vectorize_pdfs():
                st.success("All PDFs have been successfully processed and vectors saved in Qdrant!")
            else:
                st.warning("Processing completed with some errors. Check the logs for more information.")
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Force garbage collection to free up memory
            gc.collect()

st.sidebar.markdown("""
## How to use the system:
1. Click the "Process PDFs from Cloudflare R2" button to start processing all available PDFs in Cloudflare R2.
2. The system will extract text and images from each PDF using advanced image processing techniques.
3. PDFs are processed in chunks of 10 pages at a time to manage memory usage.
4. Text is analyzed for entities, topics, and key phrases using advanced NLP techniques.
5. Images are labeled with surrounding text and relevant entities.
6. A sample of extracted images will be displayed for each chunk.
7. Vectors will be stored in Qdrant after each chunk is processed.
8. You'll see a success message when the process is complete, along with total vectors and images processed.
9. Use the question answering section below to query the processed documents.
10. You can ask follow-up questions to dive deeper into specific parts of the procedures.
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
                cols = st.columns(2)  # Create two columns for images
                for idx, img_data in enumerate(images):
                    with cols[idx % 2]:  # Alternate between columns
                        st.write(f"- {img_data['image_name']} from {img_data['file_name']}, page {img_data['page']}")
                        st.write(f"  Surrounding text: {img_data['surrounding_text'][:100]}...")
                        st.write(f"  Entities: {', '.join(img_data['entities'])}")
                        
                        # Display the image
                        img_bytes = base64.b64decode(img_data['image_data'])
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, caption=img_data['image_name'], use_column_width=True)
            
            # Display debug information
            with st.expander("Debug Information"):
                st.write(f"Number of relevant images: {len(images)}")
                st.write("Context used (first 500 characters):")
                st.write(answer[:500] + "..." if len(answer) > 500 else answer)
    else:
        st.error("Please enter a question.")

# Implement follow-up questions
st.subheader("Follow-up Questions")
follow_up = st.text_input("Ask a follow-up question:")
if st.button("Get Follow-up Answer"):
    if follow_up:
        with st.spinner("Fetching follow-up answer..."):
            follow_up_answer, follow_up_images = rag_pipeline(follow_up)
            st.write("**Follow-up Answer:**", follow_up_answer)
            
            if follow_up_images:
                st.write("**Relevant Images for Follow-up:**")
                cols = st.columns(2)
                for idx, img_data in enumerate(follow_up_images):
                    with cols[idx % 2]:
                        st.write(f"- {img_data['image_name']} from {img_data['file_name']}, page {img_data['page']}")
                        st.write(f"  Surrounding text: {img_data['surrounding_text'][:100]}...")
                        st.write(f"  Entities: {', '.join(img_data['entities'])}")
                        
                        img_bytes = base64.b64decode(img_data['image_data'])
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, caption=img_data['image_name'], use_column_width=True)
    else:
        st.error("Please enter a follow-up question.")

# Implement feedback mechanism
st.subheader("Feedback")
feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
if st.button("Submit Feedback"):
    # Here you would typically send this feedback to a database or logging system
    st.write("Thank you for your feedback!")
    # In a real implementation, you'd use this feedback to improve the system

if __name__ == "__main__":
    multiprocessing.freeze_support()
