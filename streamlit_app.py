import streamlit as st
import openai
import base64
from PIL import Image
from io import BytesIO
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re

# Streamlit configuration
st.set_page_config(page_title="Engine Maintenance Assistant", page_icon="🔧", layout="wide")

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

def fetch_context_and_images(query, top_k=5):
    try:
        query_vector = sentence_transformer.encode(query).tolist()
        
        search_result = qdrant_client.search(
            collection_name="manual_vectors",
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
                    st.error(f"Failed to process image {i+1}: {str(e)}")
        return context, images
    except Exception as e:
        st.error(f"Failed to fetch context and images from Qdrant: {str(e)}")
        return "", []

def main():
    st.title('Engine Maintenance Assistant')

    user_query = st.text_input("Enter your maintenance query:")

    if user_query:
        with st.spinner("Fetching relevant information..."):
            context, images = fetch_context_and_images(user_query)
            if not context and not images:
                st.warning("No relevant information or images found.")
            else:
                response = generate_response(user_query, context, images)

                st.subheader("Maintenance Instructions:")
                
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
                                    
                                    # Attempt to display image using base64 encoding
                                    img_base64 = base64.b64encode(images[image_index]['image'].tobytes()).decode()
                                    st.markdown(f'<img src="data:image/png;base64,{img_base64}" alt="Image {image_index + 1}" style="width:100%">', unsafe_allow_html=True)
                            else:
                                st.warning(f"Image {image_index + 1} not found in fetched data.")
                
                # Display all fetched images
                st.subheader("All Fetched Images:")
                for i, img in enumerate(images):
                    st.image(img['image'], caption=f"Image {i+1}: {img['description']}", use_column_width=True)
                
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
