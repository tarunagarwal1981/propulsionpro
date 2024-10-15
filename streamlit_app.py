import streamlit as st
import openai
import base64
from PIL import Image
from io import BytesIO
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests

# Streamlit configuration
st.set_page_config(page_title="RAG Query Pipeline", page_icon="🔍", layout="wide")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=st.secrets.get("qdrant", {}).get("url", ""),
    api_key=st.secrets.get("qdrant", {}).get("api_key", "")
)

# Load the SentenceTransformer model
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

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
        image_descriptions = [f"[Image {i+1}]" for i in range(len(images))]
        context_with_images = f"{context}\n\nAvailable images: {', '.join(image_descriptions)}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions. Use the provided images in your explanation by referring to them as [Image X]."},
                {"role": "user", "content": f"Context:\n{context_with_images}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        return "I'm sorry, but I couldn't generate a response at this time. Please try again later."

def fetch_context_and_images(query, top_k=5):
    try:
        # Encode the query using the SentenceTransformer model
        query_vector = sentence_transformer.encode(query).tolist()
        
        # Search the Qdrant vector database
        search_result = qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_vector,
            limit=top_k
        )
        
        # Extract context and images from search results
        context = "\n".join([result.payload['content'] for result in search_result if 'content' in result.payload])
        images = [
            Image.open(BytesIO(base64.b64decode(result.payload['image'])))
            for result in search_result if result.payload.get('image')
        ]
        return context, images
    except Exception as e:
        st.error(f"Failed to fetch context and images from Qdrant: {str(e)}")
        return "", []

def main():
    st.title('RAG Query Pipeline with OpenAI')

    user_query = st.text_input("Enter your query:")

    if user_query:
        with st.spinner("Fetching relevant information..."):
            context, images = fetch_context_and_images(user_query)
            if not context and not images:
                st.warning("No relevant information or images found.")
            else:
                response = generate_response(user_query, context, images)

                st.subheader("Response:")
                st.write(response)

                if images:
                    st.subheader("Associated Images:")
                    for i, img in enumerate(images):
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)

if __name__ == "__main__":
    main()
    
