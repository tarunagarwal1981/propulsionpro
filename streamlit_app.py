import streamlit as st
from utils.pdf_processor import extract_content, search_content
from utils.text_processor import calculate_relevance

# Load content once at startup
content = extract_content('manuals/sample_manual.pdf')

st.title('Marine Engine Manual Query System')

query = st.text_input('Enter your maintenance query:')
if query:
    results = search_content(query, content)
    if results:
        # Sort results by relevance
        sorted_results = sorted(results, key=lambda x: calculate_relevance(query, x), reverse=True)
        for i, result in enumerate(sorted_results, 1):
            st.subheader(f'Result {i}')
            st.write(result)
            st.write(f"Relevance: {calculate_relevance(query, result):.2f}")
    else:
        st.write('No matching procedures found.')
