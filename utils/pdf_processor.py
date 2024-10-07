import PyPDF2
import re

def extract_content(pdf_path):
    content = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            # Basic parsing to separate steps
            steps = re.split(r'\d+\.', text)[1:]  # Split by numbered steps
            content.extend(steps)
    return content

def search_content(query, content):
    # Simple search implementation
    return [step for step in content if query.lower() in step.lower()]
