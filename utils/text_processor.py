# utils/text_processor.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens

def calculate_relevance(query, text):
    query_tokens = set(preprocess_text(query))
    text_tokens = set(preprocess_text(text))
    
    # Simple relevance score based on token overlap
    relevance = len(query_tokens.intersection(text_tokens)) / len(query_tokens)
    
    return relevance
