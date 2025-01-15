import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
FILE_PATH = r"D:\GenAI\RAG\data\ffiec_itbooklet_developmentacquisitionmaintenance.pdf"  
COLLECTION_NAME = "ffiec_itbooklet_langchain"
K_SIMILARITY = 10
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  
EMBEDDING_MODEL = "text-embedding-3-small"
OUTPUT_FILE = r"D:\GenAI\RAG\result\rag_results_langchain_base_query.json"

# Function to load and split documents
def load_and_split_documents(file_path: str) -> List[str]:
    start_time = datetime.now()
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    split_time = datetime.now() - start_time
    print(f"Document split time: {split_time.total_seconds()} seconds")
    return all_splits

def create_or_load_chroma_db(collection_name, embeddings, all_splits):
    start_time = datetime.now()
    persist_directory = rf"D:\GenAI\RAG\chromadb\{collection_name}"
    print(f"Persist Directory: {persist_directory}")
    os.makedirs(persist_directory, exist_ok=True)

    if not os.path.exists(persist_directory):
        print(f"Error: Persist directory does not exist: {persist_directory}")
        return None

    try:
        print(f"Creating/Loading Chroma collection: {collection_name}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # Check if the collection is empty by querying for all ids
        if not vector_store.get()['ids']:  # More efficient way to check if empty
            print("Chroma collection is empty. Adding documents...")
            vector_store = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_directory)
            print(f"Chroma collection created successfully with {len(vector_store.get()['ids'])} documents.")
        else:
            print(f"Chroma collection loaded successfully with {len(vector_store.get()['ids'])} documents.")

    except Exception as e:
        print(f"Error creating/loading Chroma collection: {e}")
        return None

    db_time = datetime.now() - start_time
    print(f"Vector DB creation/load time: {db_time.total_seconds()} seconds")
    return vector_store

# Function to retrieve contexts
def retrieve_contexts(vector_store: Chroma, query: str, k: int = 10) -> List[Dict[str, str]]:
    start_time = datetime.now()

    results = vector_store.similarity_search_with_score(query, k=k)

    retrieve_time = datetime.now() - start_time
    print(f"Vector DB retrieval time: {retrieve_time.total_seconds()} seconds")
    contexts = []
    for doc, score in results:
        page = doc.metadata.get("page", None)  # Extract the correct "page" key
        contexts.append({"content": doc.page_content, "page_number": str(page), "score": str(score)})
    return contexts

# Function to generate response using GPT
def generate_response(context: str, query: str) -> Tuple[str, int, int,int]:
    start_time = datetime.now()

    prompt = f"""Use the following context to answer the question at the end. ".
    \n\nContext:\n{context}
    \n\nQuestion: {query}
    \n\nAnswer:"""

    # full_prompt = f"**Context:**\n{context}\n\n**Question:** {prompt}"
    # print(f"the full prompt is : {full_prompt}")

    client = AzureOpenAI(api_key=OPENAI_API_KEY, 
                    api_version=OPENAI_API_VERSION, 
                    azure_endpoint=OPENAI_API_BASE
                )

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT_NAME, 
        messages=[{"role": "user", "content": prompt}]
    )
    generation_time = datetime.now() - start_time
    
    prompt_tokens = len(prompt)
    completion_tokens = len(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens, str(generation_time.total_seconds())

# Function to process a single query
def process_query(vector_store, query: str, k: int = 10) -> Dict:
    contexts_and_scores = retrieve_contexts(vector_store, query, k=k)
    context = "\n".join([content["content"] for content in contexts_and_scores])

    response, total_context_tokens, completion_tokens, generation_time = generate_response(context, query)

    total_query_tokens = len(query)
    return {
        "query": query,
        "answer": response,
        "contexts": [
            {
                "content": context.get("content", None), 
                "page_number": context.get("page_number", None), 
                "similarity_score": context.get("score", None)
                } 
                for context in contexts_and_scores
            ],
        "query_tokens": total_query_tokens,
        "context_tokens": total_context_tokens,
        "response_tokens": completion_tokens,
        "total_tokens_sent": total_query_tokens + total_context_tokens,
        "total_tokens_received": completion_tokens,
        "generation_time":generation_time
    }



def main():
    start_time = datetime.now()
    all_splits = load_and_split_documents(FILE_PATH)

    print(f"the loader has split the document into : {len(all_splits)} chunks")

    embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    )

    vector_store = create_or_load_chroma_db(COLLECTION_NAME, embeddings, all_splits)
    if vector_store is None:
        print("Error: Could not create or load Chroma collection. Exiting.")
        return 

    base_queries = [
        "What are the minimum security controls required for the development of IT systems?",
        "What are the minimum test criteria that should be used during development of IT systems?",
        "what are the minimum expectations for IT system performance and reliability according to the guidance?",
        "What are the minimum threats identified for the IT systems and components in the document?",
        "What are the greatest hits of bb king"
    ]

    all_results = []
    for query in base_queries:
        query_result = process_query(vector_store, query, k=K_SIMILARITY)
        all_results.append(query_result)

    with open(OUTPUT_FILE, "w") as f: 
        json.dump(all_results, f, indent=4)
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time.total_seconds()} seconds")

if __name__ == "__main__":
    main()