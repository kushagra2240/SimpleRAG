import streamlit as st
import pdfplumber
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import torch
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Configuration
API_KEY = OPEN_API_KEY
PDF_PATHS = ["D:\GenAI\RAG\FSB Bulletin on AI and ML.pdf", "D:\GenAI\RAG\OCC - MRM.pdf", "D:\GenAI\RAG\SR - 11-7 MRM.pdf"]
AZURE_ENDPOINT = AZURE_OPENAI_ENDPOINT
COLLECTION_NAME = "word2vec_doc_collection"
API_VERSION = "2024-07-01-preview"
MODEL_NAME = "gpt-4o-mini"

# Gensim Word2Vec model (download or train your own)
try:
    word2vec_model = api.load("word2vec-google-news-300") #try to load if already downloaded
except:
    print("Downloading word2vec model...")
    word2vec_model = api.load("word2vec-google-news-300")

# Initialize ChromaDB client
def initialize_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chromadb")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return chroma_client, collection

# Text and Metadata Extraction
def extract_text_from_pdf(pdf_path):
    chunks = []
    metadata = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                chunks.append(text)
                metadata.append({"page": f"Page {page_num}", "doc_id": f"{pdf_path}_{page_num}"})
    return chunks, metadata

# Generate Word2Vec embeddings
def generate_embeddings(texts, model):
    embeddings = []
    for text in texts:
        words = text.split()
        word_embeddings = [model.get_vector(word) for word in words if word in model.key_to_index] #added key_to_index to handle vocab mismatch
        if word_embeddings:
            text_embedding = np.mean(word_embeddings, axis=0)
        else:
            text_embedding = np.zeros(model.vector_size) #added zero vector for out of vocab texts
        embeddings.append(text_embedding.tolist())
    return embeddings

# Upsert to ChromaDB with custom embeddings
def upsert_into_chromadb(client, collection, chunks, metadata, embeddings):
    collection.add(documents=chunks, metadatas=metadata, embeddings=embeddings, ids=[meta['doc_id'] for meta in metadata])

# Process PDFs
def process_pdfs(client, collection, pdf_files, word2vec_model):
    for pdf_file in pdf_files:
        chunks, metadata = extract_text_from_pdf(pdf_file)
        embeddings = generate_embeddings(chunks, word2vec_model)
        upsert_into_chromadb(client, collection, chunks, metadata, embeddings)

# Query ChromaDB (now using embeddings)
def query_chromadb(prompt, word2vec_model, n_results=3):
    query_embedding = generate_embeddings([prompt], word2vec_model)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results

# ... (rest of the code: generate_response, flatten, main) ...
def generate_response(prompt, retrieved_chunks):
    system_prompt = """
    You are a Risk Management Officer and have to ensure bank is prepared for a regulatory examination focusing on quantity of risk and quality of model risk based on the provided context only.
    Use the retrieved information and reference sources accurately.
    """
    full_prompt = f"{system_prompt}\n\n**Context:**\n{retrieved_chunks}\n\n**Question:** {prompt}"
    print(full_prompt)
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )
    model_name = MODEL_NAME
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

# Flatten results (helper function)
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

# Initialize ChromaDB
client, collection = initialize_chromadb()

# Process PDFs (only once unless you change the PDFs)
process_pdfs(client, collection, PDF_PATHS, word2vec_model)

# Streamlit App
def main():
    st.title("RAG Document Search")
    user_query = st.text_input("Enter your query:")

    if user_query:
        chromadb_results = query_chromadb(user_query, word2vec_model)
        flat_chunks = flatten(chromadb_results["documents"])
        flat_metadata = flatten(chromadb_results["metadatas"])
        retrieved_chunks = [f"{chunk} (Source: {meta['page']})" for chunk, meta in zip(flat_chunks, flat_metadata)]
        full_retrieved_chunks = "\n\n---\n\n".join(retrieved_chunks)

        response = generate_response(user_query, full_retrieved_chunks)
        st.write("**Response:**")
        st.write(response)

if __name__ == "__main__":
    main()