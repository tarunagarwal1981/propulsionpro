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
from typing import List, Dict, Tuple, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure session state
def init_session_state():
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'zoomed_images' not in st.session_state:
        st.session_state.zoomed_images = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# Initialize NLTK
def setup_nltk():
    nltk.data.path.append('/tmp/nltk_data')
    try:
        nltk.download('punkt', quiet=True, download_dir='/tmp/nltk_data')
        nltk.download('averaged_perceptron_tagger', quiet=True, download_dir='/tmp/nltk_data')
        nltk.download('maxent_ne_chunker', quiet=True, download_dir='/tmp/nltk_data')
        nltk.download('words', quiet=True, download_dir='/tmp/nltk_data')
        return True
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")
        return False

# Load models
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return embedding_model, ner_model

# Initialize clients
def init_clients():
    try:
        minio_client = Minio(
            st.secrets["R2_ENDPOINT"].replace("https://", ""),
            access_key=st.secrets["R2_ACCESS_KEY"],
            secret_key=st.secrets["R2_SECRET_KEY"],
            secure=True
        )
    except Exception as e:
        logger.error(f"MinIO initialization failed: {e}")
        minio_client = None

    try:
        qdrant_client = QdrantClient(
            url=st.secrets["qdrant"]["url"],
            api_key=st.secrets["qdrant"]["api_key"]
        )
    except Exception as e:
        logger.error(f"Qdrant initialization failed: {e}")
        qdrant_client = None

    return minio_client, qdrant_client

class ImageProcessor:
    def __init__(self, zoom_factor: int = 4):
        self.zoom_factor = zoom_factor

    def extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        try:
            mat = fitz.Matrix(self.zoom_factor, self.zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Image processing
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            images = []
            min_size = 100 * self.zoom_factor / 2
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > min_size and h > min_size:
                    roi = img_np[y:y+h, x:x+w]
                    pil_img = Image.fromarray(roi)
                    
                    # Convert to high-quality bytes
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG', quality=95)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    images.append({
                        "name": f"Page {page_num + 1}, Image {i + 1}",
                        "image_data": base64.b64encode(img_byte_arr).decode('utf-8'),
                        "bbox": (x, y, w, h),
                        "pil_img": pil_img,
                        "size": (w, h)
                    })
            
            return images
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
            return []

    def extract_text_around_image(self, page: fitz.Page, bbox: Tuple[int, int, int, int], 
                                margin: int = 50) -> str:
        try:
            x, y, w, h = bbox
            rect = fitz.Rect(x/2-margin, y/2-margin, (x+w)/2+margin, (y+h)/2+margin)
            return page.get_text("text", clip=rect)
        except Exception as e:
            logger.error(f"Error extracting text around image: {e}")
            return ""

class TextProcessor:
    def __init__(self, ner_model):
        self.ner_model = ner_model
        
    def process_text(self, text: str) -> Dict[str, Any]:
        try:
            # Basic text processing
            sentences = nltk.sent_tokenize(text)
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            
            # Named entity recognition
            ner_results = self.ner_model(text)
            named_entities = [entity['word'] for entity in ner_results if entity['entity'] != 'O']
            
            # Combine with POS-based entities
            pos_entities = [word for word, pos in tagged if pos in ['NNP', 'NNPS']]
            all_entities = list(set(named_entities + pos_entities))
            
            return {
                'full_text': text,
                'sentences': sentences,
                'entities': all_entities,
                'tokens': tokens
            }
        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            return {
                'full_text': text,
                'sentences': [text],
                'entities': [],
                'tokens': text.split()
            }

class TopicModeling:
    def __init__(self, num_topics: int = 5, max_features: int = 100):
        self.num_topics = num_topics
        self.max_features = max_features
        
    def compute_topics(self, texts: List[str]) -> Tuple[Any, List[str]]:
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=self.num_topics, 
            random_state=42
        )
        lda.fit(tfidf_matrix)
        
        return lda, vectorizer.get_feature_names_out()

def display_image_in_streamlit(image_data: str, caption: str):
    try:
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(img, caption=caption, use_column_width=True)
        
        with col2:
            zoom_key = f"zoom_{caption}"
            if st.button(f"Zoom {caption}"):
                st.session_state[zoom_key] = not st.session_state.get(zoom_key, False)
            
            if st.session_state.get(zoom_key, False):
                st.image(img, caption="Zoomed view", width=800)
        
        return True
    except Exception as e:
        logger.error(f"Error displaying image: {e}")
        st.error(f"Failed to display image: {caption}")
        return False

class DocumentProcessor:
    def __init__(self, minio_client, qdrant_client, embedding_model, text_processor, image_processor, topic_modeler):
        self.minio_client = minio_client
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.topic_modeler = topic_modeler

    def recreate_collection(self) -> bool:
        try:
            self.qdrant_client.delete_collection("manual_vectors")
            self.qdrant_client.create_collection(
                collection_name="manual_vectors",
                vectors_config=VectorParams(size=384, distance="Cosine")
            )
            return True
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            return False

    def process_pdf(self, pdf_file_name: str, chunk_size: int = 10) -> bool:
        try:
            # Get PDF from MinIO
            response = self.minio_client.get_object(
                st.secrets["R2_BUCKET_NAME"], 
                pdf_file_name
            )
            pdf_content = response.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")

            st.write(f"Processing: {pdf_file_name}")
            progress_bar = st.progress(0)
            
            all_text = []
            total_vectors = 0
            total_images = 0

            # Process PDF in chunks
            for chunk_start in range(0, len(doc), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(doc))
                progress = chunk_start / len(doc)
                progress_bar.progress(progress)

                vectors = []
                chunk_images = []

                # Process each page in chunk
                for page_num in range(chunk_start, chunk_end):
                    try:
                        page = doc[page_num]
                        
                        # Process text
                        text = page.get_text()
                        all_text.append(text)
                        processed_text = self.text_processor.process_text(text)
                        
                        # Create text vectors
                        for sentence in processed_text['sentences']:
                            embedding = self.embedding_model.encode(sentence).tolist()
                            vectors.append(PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "type": "text",
                                    "page": page_num + 1,
                                    "content": sentence,
                                    "entities": processed_text['entities'],
                                    "file_name": pdf_file_name
                                }
                            ))

                        # Process images
                        images = self.image_processor.extract_images_from_page(page, page_num)
                        for img_data in images:
                            surrounding_text = self.image_processor.extract_text_around_image(
                                page, 
                                img_data["bbox"]
                            )
                            processed_surrounding_text = self.text_processor.process_text(surrounding_text)
                            
                            embedding = self.embedding_model.encode(
                                f"Image metadata: {surrounding_text}"
                            ).tolist()
                            
                            vectors.append(PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "type": "image",
                                    "page": page_num + 1,
                                    "content": f"Image metadata: {surrounding_text}",
                                    "surrounding_text": surrounding_text,
                                    "entities": processed_surrounding_text['entities'],
                                    "file_name": pdf_file_name,
                                    "image_name": img_data["name"],
                                    "image_data": img_data["image_data"],
                                    "size": img_data["size"]
                                }
                            ))
                            chunk_images.append((img_data["name"], img_data["pil_img"]))

                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue

                # Compute topics for the chunk
                try:
                    if all_text:
                        lda_model, feature_names = self.topic_modeler.compute_topics(all_text)
                        for vector in vectors:
                            if vector.payload["type"] == "text":
                                text_idx = all_text.index(vector.payload["content"])
                                topic_dist = lda_model.transform([all_text[text_idx]])[0]
                                vector.payload["topics"] = topic_dist.tolist()
                except Exception as e:
                    logger.warning(f"Error in topic modeling: {e}")

                # Store vectors in Qdrant
                try:
                    self.qdrant_client.upsert(
                        collection_name="manual_vectors",
                        points=vectors
                    )
                    total_vectors += len(vectors)
                    total_images += len(chunk_images)
                except Exception as e:
                    logger.error(f"Error storing vectors: {e}")

                # Display sample images
                self._display_sample_images(chunk_images)
                gc.collect()

            progress_bar.progress(1.0)
            st.success(f"Processed {total_vectors} vectors and {total_images} images")
            return True

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file_name}: {e}")
            return False

    def _display_sample_images(self, images: List[Tuple[str, Image.Image]]):
        if not images:
            return
            
        st.write("Sample images from current chunk:")
        display_images = images[:4] if len(images) <= 4 else random.sample(images, 4)
        cols = st.columns(4)
        
        for idx, (img_name, img) in enumerate(display_images):
            with cols[idx % 4]:
                st.image(img, caption=img_name, use_column_width=True)

class RAGPipeline:
    def __init__(self, qdrant_client, embedding_model):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        openai.api_key = self._get_api_key()

    def _get_api_key(self):
        if 'openai' in st.secrets:
            return st.secrets['openai']['api_key']
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        return api_key

    def get_answer(self, question: str, chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        # Get relevant context
        query_embedding = self.embedding_model.encode(question).tolist()
        search_results = self.qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_embedding,
            limit=15,
            query_filter=Filter(
                must=[FieldCondition(key="page", range=Range(gte=1))]
            )
        )

        # Process results
        context = ""
        relevant_images = []
        
        for result in search_results:
            payload = result.payload
            if payload["type"] == "text":
                context += f"From {payload['file_name']}, page {payload['page']}:\n"
                context += f"{payload['content']}\n"
                if 'topics' in payload:
                    context += f"Topics: {payload['topics']}\n"
            elif payload["type"] == "image":
                relevant_images.append(payload)
                context += f"\nImage reference: {payload['image_name']}"
                context += f" from {payload['file_name']}, page {payload['page']}:\n"
                context += f"Image context: {payload['surrounding_text']}\n"

        # Prepare chat messages
        messages = [
            {"role": "system", "content": """You are a technical documentation assistant. When answering:
                1. Provide clear step-by-step instructions when available
                2. Reference specific images by their names when relevant
                3. Cite page numbers and document names
                4. Use technical terminology accurately
                5. If information is incomplete, clearly state what's missing"""}
        ]
        
        if chat_history:
            messages.extend(chat_history)
            
        messages.extend([
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ])

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            answer = response.choices[0].message['content'].strip()
            return answer, relevant_images
        except Exception as e:
            logger.error(f"Error getting GPT response: {e}")
            return "Sorry, there was an error processing your question.", []

def create_streamlit_ui():
    st.title('Advanced PDF Processor and Query System')
    
    # Initialize session state
    init_session_state()
    
    # Initialize components
    embedding_model, ner_model = load_models()
    minio_client, qdrant_client = init_clients()
    
    if not all([minio_client, qdrant_client]):
        st.error("Error initializing clients. Please check your configuration.")
        return

    # Create processor instances
    text_processor = TextProcessor(ner_model)
    image_processor = ImageProcessor()
    topic_modeler = TopicModeling()
    doc_processor = DocumentProcessor(
        minio_client, qdrant_client, embedding_model,
        text_processor, image_processor, topic_modeler
    )
    rag_pipeline = RAGPipeline(qdrant_client, embedding_model)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Process PDFs", "Query Documents"])

    if page == "Process PDFs":
        st.header("PDF Processing")
        if st.button("Process PDFs from Storage"):
            try:
                objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"])
                pdf_files = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
                
                if not pdf_files:
                    st.warning("No PDF files found in storage.")
                    return
                
                doc_processor.recreate_collection()
                
                for pdf_file in pdf_files:
                    if pdf_file not in st.session_state.processed_files:
                        success = doc_processor.process_pdf(pdf_file)
                        if success:
                            st.session_state.processed_files.add(pdf_file)
                
                st.success("All PDFs processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

    else:  # Query Documents page
        st.header("Document Query")
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Processing your question..."):
                    answer, images = rag_pipeline.get_answer(
                        question, 
                        st.session_state.chat_history
                    )
                    
                    st.write("**Answer:**", answer)
                    
                    if images:
                        st.write("**Relevant Images:**")
                        for idx, img_data in enumerate(images):
                            st.write(f"\n### Image {idx + 1}: {img_data['image_name']}")
                            display_image_in_streamlit(
                                img_data['image_data'],
                                f"Image {idx + 1} from page {img_data['page']}"
                            )
                            
                            with st.expander("Image Details"):
                                st.write(f"Source: {img_data['file_name']}, Page: {img_data['page']}")
                                st.write("Context:", img_data['surrounding_text'])
                                if img_data['entities']:
                                    st.write("Entities:", ", ".join(img_data['entities']))
                    
                    # Update chat history
                    st.session_state.chat_history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ])
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    create_streamlit_ui()
