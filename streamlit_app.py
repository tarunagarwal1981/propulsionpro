import streamlit as st
import openai
import base64
from PIL import Image
from io import BytesIO
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re
import numpy as np

# Streamlit configuration
st.set_page_config(page_title="Engine Maintenance Assistant", page_icon="🔧", layout="wide")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets.get("qdrant", {}).get("url", ""),
    api_key=st.secrets.get("qdrant", {}).get("api_key", "")
)

# Load the SentenceTransformer models
sentence_transformer_384 = SentenceTransformer('all-MiniLM-L6-v2')
sentence_transformer_1000 = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

def get_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Set OPENAI_API_KEY as an environment variable.")
    return api_key

try:
    openai.api_key = get_api_key()
except ValueError as e:
    st.error(str(e))
    st.stop()

def generate_response(query, context, images):
    try:
        image_descriptions = [f"[Image {i+1}: {img['description']}]" for i, img in enumerate(images)]
        context_with_images = f"{context}\n\nAvailable images: {', '.join(image_descriptions)}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about engine maintenance. Use the provided images in your explanation by referring to them as [Image X]. Provide step-by-step instructions when applicable."},
                {"role": "user", "content": f"Context:\n{context_with_images}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "I'm sorry, but I couldn't generate a response at this time. Please try again later."

def fetch_context_and_images(query, collection_name, top_k=5):
    try:
        if collection_name == "manual_vectors":
            query_vector = sentence_transformer_384.encode(query).tolist()
        elif collection_name == "document_sections":
            query_vector = sentence_transformer_1000.encode(query).tolist()
        else:
            raise ValueError(f"Unknown collection name: {collection_name}")
        
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        context = "\n".join([result.payload['content'] for result in search_result if 'content' in result.payload])
        images = []
        for i, result in enumerate(search_result):
            if 'image' in result.payload:
                try:
                    image_data = base64.b64decode(result.payload['image'])
                    image = Image.open(BytesIO(image_data))
                    images.append({
                        'image': image,
                        'description': result.payload.get('description', f"Image {i+1}")
                    })
                except Exception as e:
                    st.error(f"Failed to process image {i+1} from {collection_name}: {str(e)}")
        return context, images
    except Exception as e:
        st.error(f"Failed to fetch context and images from {collection_name}: {str(e)}")
        return "", []

def display_results(response, images, collection_name):
    st.subheader(f"Results from {collection_name}")
    
    # Display debugging information
    st.write(f"Number of images fetched: {len(images)}")
    for i, img in enumerate(images):
        st.write(f"Image {i+1} description: {img['description']}")
    
    # Split the response into paragraphs
    paragraphs = response.split('\n\n')
    for paragraph in paragraphs:
        # Display the paragraph text
        st.write(paragraph)
        
        # Check if the paragraph mentions an image
        image_matches = re.findall(r'\[Image (\d+)', paragraph)
        
        # Display mentioned images
        if image_matches:
            cols = st.columns(len(image_matches))
            for i, match in enumerate(image_matches):
                image_index = int(match) - 1
                if image_index < len(images):
                    with cols[i]:
                        st.image(images[image_index]['image'], 
                                 caption=f"Image {image_index + 1}: {images[image_index]['description']}", 
                                 use_column_width=True)
                else:
                    st.warning(f"Image {image_index + 1} not found in fetched data.")
    
    # Display all fetched images
    st.subheader(f"All Fetched Images from {collection_name}:")
    for i, img in enumerate(images):
        st.image(img['image'], caption=f"Image {i+1}: {img['description']}", use_column_width=True)

def main():
    st.title('Engine Maintenance Assistant')

    user_query = st.text_input("Enter your maintenance query:")

    if user_query:
        with st.spinner("Fetching relevant information..."):
            # Fetch from manual_vectors
            context_manual, images_manual = fetch_context_and_images(user_query, "manual_vectors")
            response_manual = generate_response(user_query, context_manual, images_manual)
            
            # Fetch from document_sections
            context_doc, images_doc = fetch_context_and_images(user_query, "document_sections")
            response_doc = generate_response(user_query, context_doc, images_doc)

            if not (context_manual or images_manual) and not (context_doc or images_doc):
                st.warning("No relevant information or images found in either collection.")
            else:
                # Display results from manual_vectors
                display_results(response_manual, images_manual, "manual_vectors")
                
                # Display results from document_sections
                display_results(response_doc, images_doc, "document_sections")
        
        # Feedback mechanism
        st.subheader("Was this response helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 No"):
                st.text_area("Please tell us how we can improve:", key="feedback")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback! We'll use it to improve our system.")

if __name__ == "__main__":
    main()
