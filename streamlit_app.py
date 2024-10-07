import streamlit as st
from utils.pdf_processor import extract_content, search_content

# Load content once at startup
content = extract_content('manuals/sample_manual.pdf')

st.title('Marine Engine Manual Query System')

query = st.text_input('Enter your maintenance query:')
if query:
    results = search_content(query, content)
    if results:
        for i, result in enumerate(results, 1):
            st.subheader(f'Result {i}')
            st.write(result)
    else:
        st.write('No matching procedures found.')
