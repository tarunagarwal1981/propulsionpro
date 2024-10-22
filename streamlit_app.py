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
from typing import List, Dict, Tuple, Any, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SessionStateManager:
    """Manages Streamlit session state"""
    @staticmethod
    def init_session_state():
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'zoomed_images' not in st.session_state:
            st.session_state.zoomed_images = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []

    @staticmethod
    def add_error(error: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.error_log.append(f"{timestamp}: {error}")

class ModelLoader:
    """Handles loading and caching of ML models"""
    @st.cache_resource
    def load_embedding_model():
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    @st.cache_resource
    def load_ner_model():
        try:
            return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            raise

    @staticmethod
    def setup_nltk():
        nltk.data.path.append('/tmp/nltk_data')
        required_packages = ['punkt', 'averaged_perceptron_tagger', 
                           'maxent_ne_chunker', 'words']
        
        for package in required_packages:
            try:
                nltk.download(package, quiet=True, download_dir='/tmp/nltk_data')
            except Exception as e:
                logger.error(f"Failed to download NLTK package {package}: {e}")
                return False
        return True

class ClientManager:
    """Manages MinIO and Qdrant clients"""
    @staticmethod
    def initialize_minio() -> Optional[Minio]:
        try:
            return Minio(
                st.secrets["R2_ENDPOINT"].replace("https://", ""),
                access_key=st.secrets["R2_ACCESS_KEY"],
                secret_key=st.secrets["R2_SECRET_KEY"],
                secure=True
            )
        except Exception as e:
            logger.error(f"MinIO initialization failed: {e}")
            SessionStateManager.add_error(f"MinIO initialization failed: {str(e)}")
            return None

    @staticmethod
    def initialize_qdrant() -> Optional[QdrantClient]:
        try:
            return QdrantClient(
                url=st.secrets["qdrant"]["url"],
                api_key=st.secrets["qdrant"]["api_key"]
            )
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            SessionStateManager.add_error(f"Qdrant initialization failed: {str(e)}")
            return None

class ImageProcessor:
    """Handles image processing and manipulation"""
    def __init__(self, zoom_factor: int = 4):
        self.zoom_factor = zoom_factor

    def extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        try:
            # Create matrix for high-quality rendering
            mat = fitz.Matrix(self.zoom_factor, self.zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Image processing for contour detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                255 - binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            images = []
            min_size = 100 * self.zoom_factor / 2
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > min_size and h > min_size:
                    # Extract ROI
                    roi = img_np[y:y+h, x:x+w]
                    pil_img = Image.fromarray(roi)
                    
                    # Convert to bytes with high quality
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG', quality=95)
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Create image metadata
                    image_info = {
                        "name": f"Page {page_num + 1}, Image {i + 1}",
                        "image_data": base64.b64encode(img_byte_arr).decode('utf-8'),
                        "bbox": (x, y, w, h),
                        "pil_img": pil_img,
                        "size": (w, h),
                        "extraction_info": {
                            "zoom_factor": self.zoom_factor,
                            "threshold": 240,
                            "min_size": min_size
                        }
                    }
                    images.append(image_info)
            
            return images

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
            SessionStateManager.add_error(f"Image extraction error on page {page_num}: {str(e)}")
            return []

    def extract_text_around_image(self, page: fitz.Page, bbox: Tuple[int, int, int, int], 
                                margin: int = 50) -> str:
        try:
            x, y, w, h = bbox
            # Create extended rectangle around image
            rect = fitz.Rect(
                x/2-margin, 
                y/2-margin, 
                (x+w)/2+margin, 
                (y+h)/2+margin
            )
            return page.get_text("text", clip=rect)
        except Exception as e:
            logger.error(f"Error extracting text around image: {e}")
            return ""

class TextProcessor:
    """Handles text processing and entity extraction"""
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
            named_entities = [entity['word'] for entity in ner_results 
                            if entity['entity'] != 'O']
            
            # Technical term extraction (based on POS patterns)
            technical_terms = self._extract_technical_terms(tagged)
            
            # Combine entities
            all_entities = list(set(named_entities + technical_terms))
            
            return {
                'full_text': text,
                'sentences': sentences,
                'entities': all_entities,
                'tokens': tokens,
                'technical_terms': technical_terms
            }
        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            return {
                'full_text': text,
                'sentences': [text],
                'entities': [],
                'tokens': text.split(),
                'technical_terms': []
            }

    def _extract_technical_terms(self, tagged_tokens: List[Tuple[str, str]]) -> List[str]:
        """Extract technical terms based on POS patterns"""
        technical_terms = []
        i = 0
        while i < len(tagged_tokens) - 1:
            current_token, current_pos = tagged_tokens[i]
            next_token, next_pos = tagged_tokens[i + 1]
            
            # Pattern: Noun + Noun (e.g., "piston rod")
            if current_pos.startswith('NN') and next_pos.startswith('NN'):
                technical_terms.append(f"{current_token} {next_token}")
                i += 2
            # Pattern: Adjective + Noun (e.g., "hydraulic system")
            elif current_pos.startswith('JJ') and next_pos.startswith('NN'):
                technical_terms.append(f"{current_token} {next_token}")
                i += 2
            else:
                i += 1
        
        return technical_terms

class TopicModeling:
    """Handles topic modeling and text classification"""
    def __init__(self, num_topics: int = 5, max_features: int = 100):
        self.num_topics = num_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
    
    def compute_topics(self, texts: List[str]) -> Tuple[Any, List[str], np.ndarray]:
        try:
            # Compute TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=self.num_topics,
                random_state=42,
                n_jobs=-1
            )
            lda.fit(tfidf_matrix)
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get topic distributions
            topic_distributions = lda.transform(tfidf_matrix)
            
            return lda, feature_names, topic_distributions
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return None, [], np.array([])

    def get_topic_keywords(self, model, feature_names: List[str], 
                         top_n: int = 5) -> List[List[str]]:
        """Get top keywords for each topic"""
        try:
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                top_keywords = [feature_names[i] 
                              for i in topic.argsort()[:-top_n-1:-1]]
                topics.append(top_keywords)
            return topics
        except Exception as e:
            logger.error(f"Error getting topic keywords: {e}")
            return []

class DocumentProcessor:
    """Handles document processing and vectorization"""
    def __init__(self, minio_client, qdrant_client, embedding_model, 
                 text_processor, image_processor, topic_modeler):
        self.minio_client = minio_client
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.topic_modeler = topic_modeler

    def recreate_collection(self) -> bool:
        try:
            # Delete existing collection
            self.qdrant_client.delete_collection("manual_vectors")
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name="manual_vectors",
                vectors_config=VectorParams(size=384, distance="Cosine")
            )
            logger.info("Vector collection recreated successfully")
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
            
            # Initialize counters and storage
            all_text = []
            total_vectors = 0
            total_images = 0
            
            # Process PDF in chunks
            for chunk_start in range(0, len(doc), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(doc))
                progress = chunk_start / len(doc)
                progress_bar.progress(progress)

                chunk_data = self._process_chunk(
                    doc, chunk_start, chunk_end, pdf_file_name, all_text
                )
                
                total_vectors += chunk_data['vectors']
                total_images += chunk_data['images']
                
                # Display progress
                st.write(f"Processed pages {chunk_start + 1} to {chunk_end}")
                st.write(f"Found {chunk_data['images']} images in this chunk")
                
                # Force garbage collection
                gc.collect()

            progress_bar.progress(1.0)
            doc.close()
            
            st.success(
                f"Successfully processed {total_vectors} vectors and "
                f"{total_images} images from {pdf_file_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file_name}: {e}")
            SessionStateManager.add_error(
                f"Error processing PDF {pdf_file_name}: {str(e)}"
            )
            return False

    def _process_chunk(self, doc, start: int, end: int, 
                      pdf_file_name: str, all_text: List[str]) -> Dict[str, int]:
        vectors = []
        chunk_images = []
        
        for page_num in range(start, end):
            try:
                page_data = self._process_page(
                    doc[page_num], page_num, pdf_file_name
                )
                vectors.extend(page_data['vectors'])
                chunk_images.extend(page_data['images'])
                all_text.append(page_data['text'])
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        try:
            # Compute topics for the chunk
            if all_text:
                lda_model, feature_names, topic_distributions = \
                    self.topic_modeler.compute_topics(all_text)
                
                if lda_model is not None:
                    # Add topic information to vectors
                    for i, vector in enumerate(vectors):
                        if vector.payload["type"] == "text":
                            vector.payload["topics"] = \
                                topic_distributions[i].tolist()
                            vector.payload["topic_keywords"] = \
                                self.topic_modeler.get_topic_keywords(
                                    lda_model, feature_names
                                )

            # Store vectors in Qdrant
            if vectors:
                self.qdrant_client.upsert(
                    collection_name="manual_vectors",
                    points=vectors
                )

            return {
                'vectors': len(vectors),
                'images': len(chunk_images)
            }

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return {'vectors': 0, 'images': 0}

    def _process_page(self, page: fitz.Page, page_num: int, 
                     pdf_file_name: str) -> Dict[str, Any]:
        # Extract text
        text = page.get_text()
        processed_text = self.text_processor.process_text(text)
        
        vectors = []
        images = []
        
        # Process text
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
                    "technical_terms": processed_text['technical_terms'],
                    "file_name": pdf_file_name
                }
            ))

        # Process images
        page_images = self.image_processor.extract_images_from_page(page, page_num)
        for img_data in page_images:
            surrounding_text = self.image_processor.extract_text_around_image(
                page, img_data["bbox"]
            )
            
            processed_surrounding_text = self.text_processor.process_text(
                surrounding_text
            )
            
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
                    "technical_terms": processed_surrounding_text['technical_terms'],
                    "file_name": pdf_file_name,
                    "image_name": img_data["name"],
                    "image_data": img_data["image_data"],
                    "size": img_data["size"]
                }
            ))
            images.append((img_data["name"], img_data["pil_img"]))

        return {
            'vectors': vectors,
            'images': images,
            'text': text
        }

class RAGPipeline:
    """Handles the Retrieval-Augmented Generation pipeline"""
    def __init__(self, qdrant_client, embedding_model):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.openai_api_key = self._get_api_key()

    def _get_api_key(self) -> str:
        """Get OpenAI API key from secrets or environment"""
        if 'openai' in st.secrets:
            return st.secrets['openai']['api_key']
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        return api_key

    def get_answer(self, question: str, chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        try:
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
            context = self._process_search_results(search_results)
            
            # Generate answer
            answer = self._generate_answer(question, context, chat_history)
            
            relevant_images = [result.payload for result in search_results 
                             if result.payload["type"] == "image" and 
                             'image_data' in result.payload]
            
            return answer, relevant_images

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return "Sorry, there was an error processing your question.", []

    def _process_search_results(self, search_results) -> str:
        context = []
        
        for result in search_results:
            payload = result.payload
            if payload["type"] == "text":
                context.append(f"From {payload['file_name']}, page {payload['page']}:")
                context.append(payload['content'])
                if 'technical_terms' in payload and payload['technical_terms']:
                    context.append(f"Technical terms: {', '.join(payload['technical_terms'])}")
                if 'topics' in payload:
                    context.append(f"Topics: {', '.join(map(str, payload['topics']))}")
            elif payload["type"] == "image":
                context.append(f"\nImage reference: {payload.get('image_name', 'Unnamed Image')}")
                context.append(f"From {payload['file_name']}, page {payload['page']}:")
                context.append(f"Image context: {payload.get('surrounding_text', 'No context available')}")
                if 'technical_terms' in payload and payload['technical_terms']:
                    context.append(f"Technical terms: {', '.join(payload['technical_terms'])}")
        
        return "\n".join(context)

    def _generate_answer(self, question: str, context: str, 
                        chat_history: List[Dict] = None) -> str:
        try:
            openai.api_key = self.openai_api_key
            
            messages = [
                {"role": "system", "content": """You are a technical documentation assistant specialized in engineering manuals. When answering:
                    1. Provide clear, numbered step-by-step instructions when appropriate
                    2. Reference specific images by their exact names when relevant
                    3. Always cite page numbers and document names for each piece of information
                    4. Use technical terminology accurately and consistently
                    5. If information is incomplete, clearly state what's missing
                    6. When describing procedures, include relevant safety warnings
                    7. Organize complex information into clearly labeled sections
                    8. When technical specifications are mentioned, highlight them clearly"""}
            ]
            
            if chat_history:
                messages.extend(chat_history)
                
            messages.extend([
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ])

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, there was an error generating the answer."

def display_image_in_streamlit(image_data: Dict[str, Any], caption: str):
    """Enhanced image display function with proper error handling"""
    try:
        # Validate image data
        if not isinstance(image_data, dict):
            raise ValueError(f"Invalid image data type: {type(image_data)}")

        if 'image_data' not in image_data:
            raise KeyError("Image data not found in payload")

        # Display image info
        st.write(f"**{caption}**")
        st.write(f"Source: {image_data.get('file_name', 'Unknown')}, "
                f"Page: {image_data.get('page', 'Unknown')}")

        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display image
            img_bytes = base64.b64decode(image_data['image_data'])
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, use_column_width=True)
        
        with col2:
            # Add zoom functionality
            zoom_key = f"zoom_{caption}"
            if st.button(f"🔍 Zoom", key=f"zoom_button_{caption}"):
                st.session_state[zoom_key] = not st.session_state.get(zoom_key, False)
            
            if st.session_state.get(zoom_key, False):
                st.image(img, width=800)

        # Show metadata in expander
        with st.expander("📑 Image Details"):
            if 'surrounding_text' in image_data:
                st.write("Context:", image_data['surrounding_text'])
            if 'technical_terms' in image_data and image_data['technical_terms']:
                st.write("Technical Terms:", ", ".join(image_data['technical_terms']))
            if 'entities' in image_data and image_data['entities']:
                st.write("Entities:", ", ".join(image_data['entities']))

        return True

    except Exception as e:
        logger.error(f"Error displaying image {caption}: {e}")
        st.warning(f"Could not display image: {caption}")
        # Show available metadata
        if isinstance(image_data, dict):
            with st.expander(f"📑 Available Information for {caption}"):
                for key, value in image_data.items():
                    if key != 'image_data' and value:
                        st.write(f"{key}: {value}")
        return False

def create_streamlit_ui():
    """Creates the main Streamlit user interface"""
    st.set_page_config(
        page_title="Technical Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )

    # Initialize session state and components
    SessionStateManager.init_session_state()
    
    # Load models
    with st.spinner("Loading models..."):
        embedding_model, ner_model = ModelLoader.load_models()
        ModelLoader.setup_nltk()

    # Initialize clients
    minio_client, qdrant_client = ClientManager.initialize_minio(), ClientManager.initialize_qdrant()
    
    if not all([minio_client, qdrant_client]):
        st.error("Error initializing system. Please check the logs.")
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

    # Create sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Process Documents", "Query System"])

    if page == "Process Documents":
        _show_document_processing_page(minio_client, doc_processor)
    else:
        _show_query_page(rag_pipeline)

def _show_document_processing_page(minio_client, doc_processor):
    st.title("Document Processing")
    st.write("Process technical documentation for the query system.")

    if st.button("Process Documents"):
        try:
            objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"])
            pdf_files = [obj.object_name for obj in objects 
                        if obj.object_name.endswith('.pdf')]
            
            if not pdf_files:
                st.warning("No PDF files found in storage.")
                return
            
            doc_processor.recreate_collection()
            
            for pdf_file in pdf_files:
                if pdf_file not in st.session_state.processed_files:
                    with st.spinner(f"Processing {pdf_file}..."):
                        success = doc_processor.process_pdf(pdf_file)
                        if success:
                            st.session_state.processed_files.add(pdf_file)
            
            st.success("All documents processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

def _show_query_page(rag_pipeline):
    st.title("Technical Documentation Assistant")
    st.write("Ask questions about your technical documentation.")

    # Main query interface
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
                    for idx, img_data in enumerate(images, 1):
                        display_image_in_streamlit(
                            img_data,
                            f"Image {idx} from {img_data.get('file_name', 'Unknown')}"
                        )
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
        else:
            st.error("Please enter a question.")

    # Follow-up questions
    if st.session_state.chat_history:
        st.subheader("Follow-up Question")
        follow_up = st.text_input("Ask a follow-up question:")
        if st.button("Get Follow-up Answer"):
            if follow_up:
                with st.spinner("Processing follow-up..."):
                    answer, images = rag_pipeline.get_answer(
                        follow_up,
                        st.session_state.chat_history
                    )
                    st.write("**Follow-up Answer:**", answer)
                    
                    if images:
                        st.write("**Relevant Images:**")
                        for idx, img_data in enumerate(images, 1):
                            display_image_in_streamlit(
                                img_data,
                                f"Follow-up Image {idx}"
                            )

    # Error log display
    if st.session_state.error_log:
        with st.expander("System Logs"):
            for error in st.session_state.error_log:
                st.write(error)

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        create_streamlit_ui()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")
