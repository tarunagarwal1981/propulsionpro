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
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionStateManager:
    @staticmethod
    def init_session_state():
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'zoomed_images' not in st.session_state:
            st.session_state.zoomed_images = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

class NLTKManager:
    @staticmethod
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

class ModelManager:
    @staticmethod
    @st.cache_resource
    def load_models():
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        return embedding_model, ner_model

class ClientManager:
    @staticmethod
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
            sentences = nltk.sent_tokenize(text)
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            
            ner_results = self.ner_model(text)
            named_entities = [entity['word'] for entity in ner_results if entity['entity'] != 'O']
            
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

class UIManager:
    @staticmethod
    def display_image(image_data: str, caption: str):
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
        query_embedding = self.embedding_model.encode(question).tolist()
        search_results = self.qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_embedding,
            limit=15,
            query_filter=Filter(
                must=[FieldCondition(key="page", range=Range(gte=1))]
            )
        )

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
            elif payload["type"] == "excel":
                context += f"From {payload['file_name']}, row {payload['row_index']}:\n"
                context += f"{payload['content']}\n"

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

class StreamlitApp:
    def __init__(self):
        SessionStateManager.init_session_state()
        self.embedding_model, self.ner_model = ModelManager.load_models()
        self.minio_client, self.qdrant_client = ClientManager.init_clients()
        
        if not all([self.minio_client, self.qdrant_client]):
            st.error("Error initializing clients. Please check your configuration.")
            return

        self.text_processor = TextProcessor(self.ner_model)
        self.image_processor = ImageProcessor()
        self.topic_modeler = TopicModeling()
        self.rag_pipeline = RAGPipeline(self.qdrant_client, self.embedding_model)

    def run(self):
        st.title('Document Query System')
        st.header("Ask Your Question")
        
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                self._process_question(question)
            else:
                st.error("Please enter a question.")

    def _process_question(self, question: str):
        with st.spinner("Processing your question..."):
            answer, images = self.rag_pipeline.get_answer(
                question, 
                st.session_state.chat_history
            )
            
            st.write("**Answer:**", answer)
            
            if images:
                st.write("**Relevant Images:**")
                for idx, img_data in enumerate(images):
                    st.write(f"\n### Image {idx + 1}: {img_data['image_name']}")
                    UIManager.display_image(
                        img_data['image_data'],
                        f"Image {idx + 1} from page {img_data['page']}"
                    )
                    
                    with st.expander("Image Details"):
                        st.write(f"Source: {img_data['file_name']}, Page: {img_data['page']}")
                        st.write("Context:", img_data['surrounding_text'])
                        if img_data['entities']:
                            st.write("Entities:", ", ".join(img_data['entities']))
            
            st.session_state.chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

def main():
    multiprocessing.freeze_support()
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
