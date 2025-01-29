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
from PyPDF2 import PdfReader
import pandas as pd

load_dotenv()

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FILE_PATH = r"D:\GenAi\BankingRAG\RAG_exploration\data\ffiec_itbooklet_developmentacquisitionmaintenance.pdf"  
COLLECTION_NAME = f"Horizon_{CHUNK_SIZE}"
K_SIMILARITY = 10 # number of chunks to be retrieved
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
LLM_TEMPERATURE = 0  
EMBEDDING_MODEL = "text-embedding-3-small"
SOURCE_DIRECTORY = r"D:\GenAI\RAG\data\POV Dataset"
QUESTIONS_SOURCE = r'D:\GenAI\RAG\data\Benchmark_ExistingPOV_Horizon.xlsx'
OUTPUT_FILE = r"D:\GenAI\RAG\result\rag_results_horizon_base_query_q_t0_cs1000_jb.json"


def load_documents_from_files(file_paths: List[str]) -> List[Dict]:

    """Loads text and metadata from multiple PDF files.

    Args:
        file_paths: A list of paths to PDF files.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - "content": The extracted text from a page.
        - "metadata": A dictionary containing "page_number" and "document_name".
        Returns an empty list if there are errors loading any of the files
    """

    all_docs = []
    for file_path in file_paths:
        try:
            reader = PdfReader(file_path)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                metadata = {"page_number": page_num + 1, "document_name": os.path.basename(file_path)}
                all_docs.append({"content": text, "metadata": metadata})
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return all_docs

def load_documents_from_folder(directory: str) -> List[Dict]:

    """Loads text and metadata from a folder.

    Args:
        directory: The path to the directory containing pdf files.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - "content": The extracted text from a page.
        - "metadata": A dictionary containing "page_number" and "document_name".
        Returns an empty list if there are errors loading any of the files
    """

    all_docs = []
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found")
        return []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory,filename)
            print(file_path)
            try:
                reader = PdfReader(file_path)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    metadata = {"page_number": page_num + 1, "document_name": filename}
                    all_docs.append({"content": text, "metadata": metadata})
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return all_docs

def split_documents(documents: List[Dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:

    """Splits loaded documents into smaller chunks with metadata.

    Args:
        documents: A list of dictionaries, where each contains "content" and "metadata".
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of Document objects, each containing a chunk of text and its metadata.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    all_splits = []
    for doc in documents:
        splits = text_splitter.create_documents([doc['content']], metadatas=[doc['metadata']])
        all_splits.extend(splits)
    return all_splits

def create_or_load_chroma_db(collection_name, embeddings, all_splits):
    start_time = datetime.now()
    persist_directory = rf"D:\GenAi\BankingRAG\RAG_exploration\chromadb\{collection_name}"
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
def retrieve_contexts(vector_store: Chroma, query: str, k: int = 10) -> List[Tuple[str, float]]:
    start_time = datetime.now()

    results = vector_store.similarity_search_with_score(query, k=k)

    retrieve_time = datetime.now() - start_time
    print(f"Vector DB retrieval time: {retrieve_time.total_seconds()} seconds")
    contexts = []
    for doc,score in results:
        page_number = doc.metadata.get("page_number",None)
        document_name = doc.metadata.get("document_name",None)

        contexts.append({
        "content":doc.page_content, 
        "page_number": str(page_number), 
        "document_name": str(document_name),
        "score": str(score)})
    print (f"----------------- the retrieved context for Query {query} are {contexts}--------------------------------")
    return contexts

# Function to generate response using GPT
def generate_response(context: str, query: str) -> Tuple[str, int, int,int]:
    start_time = datetime.now()

    prompt = f"""Use the following context to answer the question at the end.
    \n\nContext:\n{context}
    \n\nQuestion: {query}\n\n
    \n\nInstruction: Answer the users Question using the Context above.
    Keep your answer grounded in the facts of the Context.
    If the Context doesn’t contain the facts to answer the Question say "I couldn't find documents relavant to the topic":"""

    # full_prompt = f"**Context:**\n{context}\n\n**Question:** {prompt}"

    client = AzureOpenAI(api_key=OPENAI_API_KEY, 
                    api_version=OPENAI_API_VERSION, 
                    azure_endpoint=OPENAI_API_BASE                    
                )

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT_NAME, 
        messages=[{"role": "user", "content": prompt}],
        temperature = LLM_TEMPERATURE
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
                "content" : context_meta_info.get("content", None),
                "page_number" : context_meta_info.get("page_number",None),
                "document_name" : context_meta_info.get("document_name",None),
                "similarity_score" : context_meta_info.get("score",None)
            }
            for context_meta_info in contexts_and_scores
        ],
        "query_tokens": total_query_tokens,
        "context_tokens": total_context_tokens,
        "response_tokens": completion_tokens,
        "total_tokens_sent": total_query_tokens + total_context_tokens,
        "total_tokens_received": completion_tokens,
        "generation_time":generation_time
    }
#df cleanup
def drop_below_empty_row(df: pd.DataFrame) -> pd.DataFrame:
    empty_row_index = df.isna().all(axis=1).idxmax()

    if pd.notna(empty_row_index):
        return df.loc[:empty_row_index-1]
    else:
        return df
    

def main():
    start_time = datetime.now()

    all_docs = load_documents_from_folder(SOURCE_DIRECTORY)  
    print(f"there are {len(all_docs)} documents loaded")  
    all_splits = split_documents(all_docs)

    embeddings = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"the loader has split the documents into : {len(all_splits)} chunks")

    embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    )

    vector_store = create_or_load_chroma_db(COLLECTION_NAME, embeddings, all_splits)
    if vector_store is None:
        print("Error: Could not create or load Chroma collection. Exiting.")
        return 

    df_datasource = pd.read_excel(QUESTIONS_SOURCE)
    df_datasource = drop_below_empty_row(df_datasource)
    base_queries = [question.strip() for question in df_datasource['Question'] if pd.notna(question) and question != ""]

    all_results = []
    for query in base_queries:
        print (f"query in main() is {query}")
        query_result = process_query(vector_store, query, k=K_SIMILARITY)
        all_results.append(query_result)

    with open(OUTPUT_FILE, "w") as f: 
        json.dump(all_results, f, indent=4)
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time.total_seconds()} seconds")

if __name__ == "__main__":
    main()