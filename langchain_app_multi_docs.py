import streamlit as st
import os
from datetime import datetime
from typing import Dict, List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from openai import AzureOpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

# Configuration 
FILE_PATHS = [
    r"D:\GenAI\RAG\FSB Bulletin on AI and ML.pdf",
    r"D:\GenAI\RAG\OCC - MRM.pdf",  
    r"D:\GenAI\RAG\SR - 11-7 MRM.pdf"
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LLM_TEMPERATURE = 0
COLLECTION_NAME = f"multi_pdf_langchain_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
K_SIMILARITY = 10
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
EMBEDDING_MODEL = "text-embedding-3-small"

def load_documents(file_paths: List[str]) -> List[Dict]:

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

    """Creates or loads a Chroma vector database.

    Args:
        collection_name: The name of the Chroma collection.
        embeddings: The embedding function to use.
        all_splits: A list of Document objects to add to the database.

    Returns:
        A Chroma vector store object, or None if an error occurs.
    """

    start_time = datetime.now()
    persist_directory = rf"D:\GenAI\RAG\chromadb\{collection_name}"
    print(f"Persist Directory: {persist_directory}")
    os.makedirs(persist_directory, exist_ok=True)

    try:
        print(f"Creating/Loading Chroma collection: {collection_name}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        if not vector_store.get()['ids']:
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

def retrieve_contexts(vector_store: Chroma, query: str, k: int = 10) -> List[Dict[str, str]]:

    """Retrieves relevant contexts from the Chroma vector database.

    Args:
        vector_store: The Chroma vector store.
        query: The query to search for.
        k: The number of nearest neighbors to retrieve.

    Returns:
        A list of dictionaries, where each contains "content", "page_number", "document_name", and "score".
    """

    start_time = datetime.now()
    results = vector_store.similarity_search_with_score(query, k=k)
    retrieve_time = datetime.now() - start_time
    print(f"Vector DB retrieval time: {retrieve_time.total_seconds()} seconds")
    contexts = []
    for doc, score in results:
        page_number = doc.metadata.get("page_number", None)
        document_name = doc.metadata.get("document_name", None)
        contexts.append({"content": doc.page_content, "page_number": page_number, "document_name": document_name, "score": str(score)})
    return contexts

def generate_response(context: str, query: str) -> Tuple[str, int, int, str]:

    """Generates a response using the GPT model.

    Args:
        context: The retrieved context to use for generating the response.
        query: The user's query.

    Returns:
        A tuple containing the response text, prompt tokens, completion tokens and generation time.
    """

    start_time = datetime.now()
    prompt = f"""Use the following context to answer the question at the end. ".
    \n\nContext:\n{context}
    \n\nQuestion: {query}
    \n\nAnswer:
    \n\nInstruction: Answer the Question using the Context above.
    Keep your answer grounded in the facts of the Context.
    If the Context doesn't contain the facts to answer the Question say "I couldn't find documents relevant to the topic": """

    client = AzureOpenAI(api_key=OPENAI_API_KEY,
                         api_version=OPENAI_API_VERSION,
                         azure_endpoint=OPENAI_API_BASE
                         )

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE
    )
    generation_time = datetime.now() - start_time

    prompt_tokens = len(prompt)
    completion_tokens = len(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens, str(generation_time.total_seconds())

def process_query(vector_store, query: str, k: int = 10) -> Dict:

    """Processes a user query and returns the answer and related information.

    Args:
        vector_store: The Chroma vector store.
        query: The user's query.
        k: The number of contexts to retrieve.

    Returns:
        A dictionary containing the query, answer, contexts, token usage, and generation time.
    """

    contexts_and_scores = retrieve_contexts(vector_store, query, k=k)
    context = "\n".join([c["content"] for c in contexts_and_scores])  # Corrected context joining

    response, total_context_tokens, completion_tokens, generation_time = generate_response(context, query)

    total_query_tokens = len(query)
    return {
        "query": query,
        "answer": response,
        "contexts": contexts_and_scores, # Return the list of context dictionaries
        "query_tokens": total_query_tokens,
        "context_tokens": total_context_tokens,
        "response_tokens": completion_tokens,
        "total_tokens_sent": total_query_tokens + total_context_tokens,
        "total_tokens_received": completion_tokens,
        "generation_time": generation_time
    }

def main():

    """Main function to run the Streamlit app."""

    st.title("LangChain RAG - Multiple PDF Question Answering App")

    if "vector_store" not in st.session_state:
        all_docs = load_documents(FILE_PATHS)
        if not all_docs:
            st.error("Failed to load documents. Check file paths and PDF format.")
            return
        all_splits = split_documents(all_docs)
        embeddings = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)
        st.session_state.vector_store = create_or_load_chroma_db(COLLECTION_NAME, embeddings, all_splits)

        if st.session_state.vector_store is None:
            st.error("Could not create or load Chroma collection. Please check logs.")
            return

    vector_store = st.session_state.vector_store

    user_query = st.text_input("Ask your question here:")

    if vector_store is None:
        st.error("Vector store not initialized. Please check logs.")
        return

    if user_query:
        with st.spinner("Generating response..."):
            query_result = process_query(vector_store, user_query, k=K_SIMILARITY)

        st.subheader("Answer:")
        st.write(query_result["answer"])

        with st.expander("Context"):
            for context in query_result["contexts"]:
                st.write(f"- **Document:** {context.get('document_name', 'N/A')}, **Page:** {context.get('page_number', 'N/A')}: {context['content']}") 

        st.subheader("Generation Time:")
        st.write(f"{query_result['generation_time']} seconds")

if __name__ == "__main__":
    main()