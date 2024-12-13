import streamlit as st
import pdfplumber
import chromadb
# import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()   
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Configuration 
API_KEY = OPEN_API_KEY
PDF_PATHS = ["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"]
AZURE_ENDPOINT = AZURE_OPENAI_ENDPOINT
COLLECTION_NAME = "doc_collection"
API_VERSION = "2024-07-01-preview"
MODEL_NAME = "gpt-4o-mini"

# Initialize ChromaDB client
def initialize_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chromadb")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return chroma_client, collection

# Text and Metadata Extraction
def extract_text_from_pdf(pdf_path):
    """Extracts text and metadata from a single PDF."""
    chunks = []
    metadata = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                chunks.append(text)
                metadata.append({"page": f"Page {page_num}", "doc_id": f"{pdf_path}_{page_num}"})
    # print(metadata)
    return chunks, metadata

# Upsert to ChromaDB
def upsert_into_chromadb(client, collection, chunks, metadata):
    collection.add(documents=chunks, metadatas=metadata, ids=[meta['doc_id'] for meta in metadata])

# Process PDFs (separate function for clarity)
def process_pdfs(client, collection, pdf_files):
    for pdf_file in pdf_files:
        chunks, metadata = extract_text_from_pdf(pdf_file)
        upsert_into_chromadb(client, collection, chunks, metadata)

# Query ChromaDB
def query_chromadb(prompt, n_results=3):
    results = collection.query(
    query_texts=[prompt],  # User's query
    n_results=n_results,   # Number of relevant chunks to retrieve
    include=["documents", "metadatas"]  # Retrieve both document text and metadata
    )
    return results

# Generate Response with System Prompt
def generate_response(prompt, retrieved_chunks):
    system_prompt = """
    You are a Risk Management Officer and have to ensure bank is prepared for a regulatory examination focusing on quantity of risk and quality of model risk based on the provided context only.
    Use the retrieved information and reference sources accurately.
    """
    full_prompt = f"{system_prompt}\n\n**Context:**\n{retrieved_chunks}\n\n**Question:** {prompt}"
    print (full_prompt)
    # response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": full_prompt}])
    client = AzureOpenAI(
            api_key = API_KEY,
            api_version = "2024-07-01-preview",
            azure_endpoint =  AZURE_ENDPOINT
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

# Initialize ChromaDB (outside main for one-time setup)
client, collection = initialize_chromadb()
# Process PDFs only once (optional comment out after initial processing)
process_pdfs(client, collection, PDF_PATHS) 

# Streamlit App
def main():
    st.title("RAG Document Search")

    # User query
    user_query = st.text_input("Enter your query:")

    if user_query:
        chromadb_results = query_chromadb(user_query)
        flat_chunks = flatten(chromadb_results["documents"])
        flat_metadata = flatten(chromadb_results["metadatas"])
        retrieved_chunks = [f"{chunk} (Source: {meta['page']})" for chunk, meta in zip(flat_chunks, flat_metadata)]
        full_retrieved_chunks = "\n\n---\n\n".join(retrieved_chunks)

        response = generate_response(user_query, full_retrieved_chunks)
        st.write("**Response:**")
        st.write(response)

if __name__ == "__main__":

    main()