import streamlit as st
import PyPDF2
import io
from minio import Minio
from minio.error import S3Error

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
    content = []
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    for page in pdf_reader.pages:
        content.append(page.extract_text())
    return ' '.join(content)

def search_content(query, content):
    return [sentence for sentence in content.split('.') if query.lower() in sentence.lower()]

# Streamlit UI
st.title('PropulsionPro: Marine Engine Manual Query System')

# Get list of PDF files from R2
try:
    objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
    pdf_file_names = [obj.object_name for obj in objects if obj.object_name.endswith('.pdf')]
except S3Error as exc:
    st.error(f"Error listing objects in R2: {exc}")
    pdf_file_names = []

# Dropdown to select PDF file
selected_file = st.selectbox('Select a PDF manual:', pdf_file_names)

if selected_file:
    pdf_content = get_pdf_from_r2(selected_file)
    if pdf_content:
        content = extract_content(pdf_content)
        st.success(f'Manual "{selected_file}" successfully loaded from R2!')

        # Search functionality
        query = st.text_input('Enter your maintenance query:')
        if query:
            results = search_content(query, content)
            if results:
                for i, result in enumerate(results, 1):
                    st.subheader(f'Result {i}')
                    st.write(result)
            else:
                st.write('No matching procedures found.')

# File uploader for adding new PDFs to R2
uploaded_file = st.file_uploader("Upload a new PDF manual", type="pdf")
if uploaded_file is not None:
    try:
        minio_client.put_object(
            BUCKET_NAME, 
            uploaded_file.name, 
            uploaded_file, 
            length=uploaded_file.size,
            content_type='application/pdf'
        )
        st.success(f"File {uploaded_file.name} successfully uploaded to R2!")
    except S3Error as exc:
        st.error(f"Error uploading file to R2: {exc}")

st.sidebar.markdown("""
## How to use PropulsionPro:
1. Select a PDF manual from the dropdown or upload a new one.
2. Wait for the manual to load.
3. Enter your maintenance query in the search box.
4. View the matching results below.
""")
