import streamlit as st
import re
import json
import PyPDF2
from pdf2image import convert_from_bytes
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from qdrant_client import QdrantClient, models
import uuid
import os
import openai
import base64
import threading

# Disable file watcher
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'

# Streamlit configuration
st.set_page_config(page_title="PropulsionPro", page_icon="🚢", layout="wide")

class DocumentProcessor:
    def __init__(self, text, images):
        self.text = text
        self.images = images
        self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_sections(self):
        sections = re.split(r'\n(?=\d{3}-\d+\.)', self.text)
        return [section.strip() for section in sections if section.strip()]

    def extract_data_tables(self, text):
        tables = re.findall(r'(D\d+-\d+.*?(?:\n.*?)+)', text)
        return tables

    def extract_procedures(self, text):
        procedures = re.findall(r'(\d+\.\s.*?(?:\n(?!\d+\.).+)*)', text, re.DOTALL)
        return procedures

    def identify_image_descriptions(self, text):
        image_desc = re.findall(r'(Figure \d+:.*?(?:\n(?!Figure \d+:).+)*)', text, re.DOTALL)
        return image_desc

    def vectorize_text(self, text):
        try:
            vector = self.vectorizer.encode(text)
            return vector.tolist()
        except Exception as e:
            st.error(f"Vectorization failed: {str(e)}")
            return [0] * 384  # Return a default vector of zeros in case of failure

    def process_section(self, section, index):
        title_match = re.match(r'(\d{3}-\d+\..*?)\n', section)
        title = title_match.group(1) if title_match else f"Untitled Section {index}"
        content = section[len(title):].strip() if title_match else section

        tables = self.extract_data_tables(content)
        procedures = self.extract_procedures(content)
        image_descriptions = self.identify_image_descriptions(content)
        vector = self.vectorize_text(content)

        # Assign an image to this section if available
        image = self.images[index] if index < len(self.images) else None
        image_data = image_to_base64(image) if image else None

        return {
            "title": title,
            "content": content,
            "tables": tables,
            "procedures": procedures,
            "image_descriptions": image_descriptions,
            "vector": vector,
            "image": image_data
        }

    def process_document(self):
        sections = self.extract_sections()
        processed_sections = [self.process_section(section, i) for i, section in enumerate(sections)]
        return processed_sections

def extract_text_from_pdf(pdf_file, max_pages=5):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        if page_num >= max_pages:
            break
        text += page.extract_text() + "\n"
    return text

def extract_images_from_pdf(pdf_file, max_images=5):
    try:
        images = convert_from_bytes(pdf_file.getvalue(), poppler_path='/usr/bin')
        return images[:max_images]
    except Exception as e:
        st.warning(f"Image extraction failed: {str(e)}")
        return []

def image_to_base64(image):
    if image is None:
        return None
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

def initialize_qdrant():
    try:
        client = QdrantClient(
            url=st.secrets["qdrant"]["url"],
            api_key=st.secrets["qdrant"]["api_key"]
        )
        collections = client.get_collections().collections
        if not any(collection.name == "manual_vectors" for collection in collections):
            client.create_collection(
                collection_name="manual_vectors",
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        return client
    except KeyError as e:
        st.error(f"Qdrant initialization failed: Missing secret key {e}")
        return None
    except Exception as e:
        st.error(f"Qdrant initialization error: {str(e)}")
        return None

qdrant_client = initialize_qdrant()

try:
    openai.api_key = get_api_key()
except ValueError as e:
    st.error(str(e))
    st.stop()

def save_to_qdrant(processed_doc, file_name):
    if qdrant_client is None:
        st.warning("Qdrant is not available. Saving data locally.")
        return

    points = []
    for section in processed_doc:
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=section['vector'],
            payload={
                "title": section['title'],
                "content": section['content'],
                "file_name": file_name,
                "image": section['image']
            }
        )
        points.append(point)
    
    try:
        qdrant_client.upsert(
            collection_name="manual_vectors",
            points=points
        )
        st.success("Data saved to Qdrant successfully!")
    except Exception as e:
        st.error(f"Failed to save data to Qdrant: {str(e)}")
        st.warning("Saving data locally as a fallback.")

def process_pdf_in_background(pdf_file):
    pdf_text = extract_text_from_pdf(pdf_file)
    pdf_images = extract_images_from_pdf(pdf_file)
    processor = DocumentProcessor(pdf_text, pdf_images)
    processed_doc = processor.process_document()
    save_to_qdrant(processed_doc, pdf_file.name)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, top_k=5):
    if qdrant_client is None:
        st.warning("Qdrant is not available. Search functionality is limited.")
        return []

    query_vector = sentence_transformer.encode(query).tolist()
    try:
        search_result = qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_vector,
            limit=top_k
        )
        return search_result
    except Exception as e:
        st.error(f"Failed to perform search in Qdrant: {str(e)}")
        return []

def generate_response(query, context, images):
    try:
        image_descriptions = [f"[Image {i+1}]" for i in range(len(images))]
        context_with_images = f"{context}\n\nAvailable images: {', '.join(image_descriptions)}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures. Use the provided images in your explanation by referring to them as [Image X]."},
                {"role": "user", "content": f"Context:\n{context_with_images}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "I'm sorry, but I couldn't generate a response at this time. Please try again later."

def main():
    st.title('PropulsionPro: Marine Engine Maintenance Assistant')

    if qdrant_client:
        st.sidebar.success("Connected to Qdrant")
    else:
        st.sidebar.error("Not connected to Qdrant. Some features may be limited.")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        with st.spinner("Processing PDF..."):
            pdf_file = BytesIO(uploaded_file.getvalue())
            # Run processing in a separate thread to avoid blocking the UI
            thread = threading.Thread(target=process_pdf_in_background, args=(pdf_file,))
            thread.start()
            thread.join()

    user_query = st.text_input("Ask a question about marine engine maintenance:")

    if user_query:
        with st.spinner("Searching for relevant information..."):
            search_results = semantic_search(user_query)
            if search_results:
                context = "\n".join([result.payload['content'] for result in search_results])
                images = [Image.open(BytesIO(base64.b64decode(result.payload['image']))) for result in search_results if result.payload.get('image')]
                response = generate_response(user_query, context, images)

                st.subheader("Response:")
                st.write(response)

                if images:
                    st.subheader("Relevant Images:")
                    for i, img in enumerate(images):
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)

                st.subheader("Relevant Sections:")
                for result in search_results:
                    st.write(f"From: {result.payload['file_name']}")
                    st.write(f"Section: {result.payload['title']}")
                    st.write(result.payload['content'][:500] + "...")
                    st.write("---")
            else:
                st.warning("No relevant information found. This could be due to Qdrant connection issues or lack of matching content.")

    st.sidebar.title("How to use PropulsionPro")
    st.sidebar.markdown("""
    1. Upload a PDF manual using the file uploader.
    2. Wait for the PDF to be processed and saved to the vector database.
    3. Ask a question about marine engine maintenance in the text input field.
    4. Review the AI-generated response, relevant images, and sections from the manual.

    Note: If Qdrant is not available, some features may be limited.
    """)

    if st.button("Refresh App"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
