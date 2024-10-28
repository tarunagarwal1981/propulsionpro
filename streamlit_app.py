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

[... Previous ImageProcessor, TextProcessor, TopicModeling classes remain unchanged ...]

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

[... Previous DocumentProcessor class remains unchanged ...]

[... Previous RAGPipeline class remains unchanged ...]

def create_streamlit_ui():
    st.title('Document Query System')
    
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

    # Comment out sidebar navigation
    # st.sidebar.title("Navigation")
    # page = st.sidebar.radio("Choose a page:", ["Process PDFs", "Query Documents"])
    
    # Comment out the Process PDFs page condition
    # if page == "Process PDFs":
    #     st.header("PDF Processing")
    #     if st.button("Process PDFs and Excels from Storage"):
    #         try:
    #             objects = minio_client.list_objects(st.secrets["R2_BUCKET_NAME"])
    #             pdf_files = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
    #             excel_files = [obj.object_name for obj in objects if obj.object_name.endswith('.xlsx') or obj.object_name.endswith('.xls')]
                
    #             if not pdf_files and not excel_files:
    #                 st.warning("No PDF or Excel files found in storage.")
    #                 return
                
    #             doc_processor.recreate_collection()
                
    #             for pdf_file in pdf_files:
    #                 if pdf_file not in st.session_state.processed_files:
    #                     success = doc_processor.process_pdf(pdf_file)
    #                     if success:
    #                         st.session_state.processed_files.add(pdf_file)

    #             for excel_file in excel_files:
    #                 if excel_file not in st.session_state.processed_files:
    #                     success = doc_processor.process_excel(excel_file)
    #                     if success:
    #                         st.session_state.processed_files.add(excel_file)
                
    #             st.success("All PDFs and Excels processed successfully!")
    #         except Exception as e:
    #             st.error(f"Error processing files: {e}")

    # else:  # Query Documents page
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
