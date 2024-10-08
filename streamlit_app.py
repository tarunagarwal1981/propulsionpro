import streamlit as st
import fitz  # PyMuPDF
import io
import os
from minio import Minio
from minio.error import S3Error
from PIL import Image
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Function to get OpenAI API key
def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

# Set OpenAI API key
openai.api_key = get_api_key()

# Initialize MinIO client (for R2)
minio_client = Minio(
    st.secrets["R2_ENDPOINT"].replace("https://", ""),
    access_key=st.secrets["R2_ACCESS_KEY"],
    secret_key=st.secrets["R2_SECRET_KEY"],
    secure=True
)

BUCKET_NAME = st.secrets["R2_BUCKET_NAME"]

def get_pdf_from_r2(file_name):
    try:
        response = minio_client.get_object(BUCKET_NAME, file_name)
        return response.read()
    except S3Error as exc:
        st.error(f"Error downloading file from R2: {exc}")
        return None

def extract_content(pdf_content):
    text_content = []
    images = []
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    for page in doc:
        text_content.append(page.get_text())
        for img in page.get_images():
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return ' '.join(text_content), images

def process_pdf(pdf_content):
    text_content, images = extract_content(pdf_content)
    
    # Create text chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base, images

def answer_query(query, knowledge_base):
    docs = knowledge_base.similarity_search(query)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    return response

# Streamlit UI
st.title('PropulsionPro: Marine Engine Manual Query System')

try:
    objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
    pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
except S3Error as exc:
    st.error(f"Error listing objects in R2: {exc}")
    pdf_file_names = []

selected_file = st.selectbox('Select a PDF manual:', pdf_file_names)

if selected_file:
    pdf_content = get_pdf_from_r2(selected_file)
    if pdf_content:
        st.success(f'Manual "{selected_file}" successfully loaded from R2!')
        knowledge_base, images = process_pdf(pdf_content)

        query = st.text_input('Enter your maintenance query:')
        if query:
            response = answer_query(query, knowledge_base)
            st.subheader('Answer:')
            st.write(response)

            st.subheader('Related Images:')
            for i, img in enumerate(images):
                st.image(img, caption=f'Image {i+1}', use_column_width=True)

st.sidebar.markdown("""
## How to use PropulsionPro:
1. Select a PDF manual from the dropdown.
2. Wait for the manual to load and process.
3. Enter your maintenance query in the search box.
4. View the answer and related images below.
""")
