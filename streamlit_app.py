import streamlit as st
import re
import json
import PyPDF2
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_bytes
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import uuid
import os
import openai
import base64

# Streamlit configuration
st.set_page_config(page_title="PropulsionPro", page_icon="🚢", layout="wide")

# Install necessary packages (this will only work in environments that allow package installation)
# st.sidebar.info("Installing necessary packages...")
# os.system("apt-get install -y poppler-utils")
# os.system("pip install pdf2image PyPDF2 qdrant-client openai")

class DocumentProcessor:
    def __init__(self, text):
        self.text = text
        self.vectorizer = TfidfVectorizer()

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
        vector = self.vectorizer.fit_transform([text])
        return vector.toarray()[0]

    def process_section(self, section):
        title_match = re.match(r'(\d{3}-\d+\..*?)\n', section)
        title = title_match.group(1) if title_match else "Untitled Section"
        content = section[len(title):].strip() if title_match else section

        tables = self.extract_data_tables(content)
        procedures = self.extract_procedures(content)
        image_descriptions = self.identify_image_descriptions(content)
        vector = self.vectorize_text(content)

        return {
            "title": title,
            "content": content,
            "tables": tables,
            "procedures": procedures,
            "image_descriptions": image_descriptions,
            "vector": vector.tolist()
        }

    def process_document(self):
        sections = self.extract_sections()
        processed_sections = [self.process_section(section) for section in sections]
        return processed_sections

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_images_from_pdf(pdf_file):
    images = convert_from_bytes(pdf_file.getvalue())
    return images

def save_to_json(data):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    return json.dumps(data, cls=NumpyEncoder, indent=2)

def visualize_sections(processed_doc):
    section_lengths = [len(section['content']) for section in processed_doc]
    section_titles = [section['title'] for section in processed_doc]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(section_titles, section_lengths)
    ax.set_title('Section Lengths')
    ax.set_xlabel('Sections')
    ax.set_ylabel('Character Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def display_images(images):
    for i, image in enumerate(images):
        st.image(image, caption=f"Image {i+1}", use_column_width=True)

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

qdrant_client = init_qdrant()

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def save_to_qdrant(processed_doc, file_name):
    points = []
    for section in processed_doc:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=section['vector'],
            payload={
                "title": section['title'],
                "content": section['content'],
                "file_name": file_name
            }
        )
        points.append(point)
    
    qdrant_client.upsert(
        collection_name="manual_vectors",
        points=points
    )

def semantic_search(query, top_k=5):
    query_vector = TfidfVectorizer().fit_transform([query]).toarray()[0]
    search_result = qdrant_client.search(
        collection_name="manual_vectors",
        query_vector=query_vector.tolist(),
        limit=top_k
    )
    return search_result

def generate_response(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about marine engine maintenance procedures."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message['content']

# Streamlit UI
st.title('PropulsionPro: Marine Engine Maintenance Assistant')

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    with st.spinner("Processing PDF..."):
        pdf_file = BytesIO(uploaded_file.getvalue())
        
        pdf_text = extract_text_from_pdf(pdf_file)
        pdf_images = extract_images_from_pdf(pdf_file)

        processor = DocumentProcessor(pdf_text)
        processed_doc = processor.process_document()

        save_to_qdrant(processed_doc, uploaded_file.name)

        st.success("PDF processed and saved to vector database!")

    st.subheader("Document Structure")
    visualize_sections(processed_doc)

    st.subheader("Extracted Images")
    display_images(pdf_images)

    json_str = save_to_json(processed_doc)
    st.download_button(
        label="Download processed document as JSON",
        data=json_str,
        file_name="processed_document.json",
        mime="application/json"
    )

# User query
user_query = st.text_input("Ask a question about marine engine maintenance:")

if user_query:
    with st.spinner("Searching for relevant information..."):
        search_results = semantic_search(user_query)
        context = "\n".join([result.payload['content'] for result in search_results])
        response = generate_response(user_query, context)

        st.subheader("Response:")
        st.write(response)

        st.subheader("Relevant Sections:")
        for result in search_results:
            st.write(f"From: {result.payload['file_name']}")
            st.write(f"Section: {result.payload['title']}")
            st.write(result.payload['content'][:500] + "...")
            st.write("---")

# Sidebar with instructions
st.sidebar.title("How to use PropulsionPro")
st.sidebar.markdown("""
1. Upload a PDF manual using the file uploader.
2. Wait for the PDF to be processed and saved to the vector database.
3. Ask a question about marine engine maintenance in the text input field.
4. Review the AI-generated response and relevant sections from the manual.
""")
