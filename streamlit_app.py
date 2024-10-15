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
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        return client
    except KeyError as e:
        st.error(f"Qdrant initialization failed: Missing secret key {e}")
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
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=section['vector'],
            payload={
                "title": section['title'],
                "content": section['content'],
                "file_name": file_name,
                "images": [image_to_base64(img) for img in section['images']] if 'images' in section else []
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

def semantic_search(query, top_k=5):
    if qdrant_client is None:
        st.warning("Qdrant is not available. Search functionality is limited.")
        return []

    query_vector = TfidfVectorizer().fit_transform([query]).toarray()[0]
    try:
        search_result = qdrant_client.search(
            collection_name="manual_vectors",
            query_vector=query_vector.tolist(),
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

# Streamlit UI
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
        
        pdf_text = extract_text_from_pdf(pdf_file)
        pdf_images = extract_images_from_pdf(pdf_file)

        processor = DocumentProcessor(pdf_text)
        processed_doc = processor.process_document()

        save_to_qdrant(processed_doc, uploaded_file.name)

    st.subheader("Document Structure")
    visualize_sections(processed_doc)

    
    
user_query = st.text_input("Ask a question about marine engine maintenance:")

if user_query:
    with st.spinner("Searching for relevant information..."):
        search_results = semantic_search(user_query)
        if search_results:
            context = "\n".join([result.payload['content'] for result in search_results])
            images = []
            for result in search_results:
                if 'images' in result.payload:
                    try:
                        for img_base64 in result.payload['images']:
                            img = Image.open(BytesIO(base64.b64decode(img_base64)))
                            images.append(img)
                    except Exception as e:
                        st.warning(f"Failed to load an image: {str(e)}")
            
            response = generate_response(user_query, context, images)

            st.subheader("Response:")
            st.write(response)

            
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
