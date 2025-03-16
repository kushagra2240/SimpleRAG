import os
from datetime import datetime
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from config import EMBEDDING_MODEL, COLLECTION_NAME, DB_BUILDING_BATCH_SIZE

def create_or_load_chroma_db(all_splits):
    start_time = datetime.now()
    persist_directory = rf"D:\GenAi\BankingRAG\RAG_exploration\chromadb\{COLLECTION_NAME}"
    print(f"Persist Directory: {persist_directory}")
    os.makedirs(persist_directory, exist_ok=True)

    if not os.path.exists(persist_directory):
        print(f"Error: Persist directory does not exist: {persist_directory}")
        return None

    embeddings = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = None

    try:
        print(f"Creating/Loading Chroma collection: {COLLECTION_NAME}")
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        if not vector_store.get()['ids']:
            batch_size = DB_BUILDING_BATCH_SIZE
            all_sub_splits = [all_splits[x:x+batch_size] for x in range(0, len(all_splits), batch_size)]
            print(f"Number of splits is {len(all_sub_splits)}")
            print("Chroma collection is empty. Adding documents...")
            vector_store = Chroma.from_documents(documents=all_sub_splits[0], embedding=embeddings, persist_directory=persist_directory)
            for split in all_sub_splits[1:]:
                vector_store.add_documents(split)
            print(f"Chroma collection created successfully with {len(vector_store.get()['ids'])} documents.")
        else:
            print(f"Chroma collection loaded successfully with {len(vector_store.get()['ids'])} documents.")

    except Exception as e:
        print(f"Error creating/loading Chroma collection: {e}")
        return None

    db_time = datetime.now() - start_time
    print(f"Vector DB creation/load time: {db_time.total_seconds()} seconds")
    return vector_store

def retrieve_from_vector_db(vector_store: Chroma, query: str, k: int) -> List[Dict]:
    start_time = datetime.now()
    results = vector_store.similarity_search_with_score(query, k=k)
    retrieve_time = datetime.now() - start_time
    print(f"Vector DB retrieval time: {retrieve_time.total_seconds()} seconds")

    # print("\n--- Raw VectorDB Retrieval Results (from vector_db.py) ---") # Added print statement
    # for chunk, score in results:
    #     print(f"Chunk content (truncated): {chunk.page_content[:50]}...") # Print first 100 chars
    #     print(f"Metadata: {chunk.metadata}")
    #     print(f"Relevance Score: {score}")
    # print("--- End Raw VectorDB Retrieval Results ---\n")

    vector_contexts = []
    for doc, score in results:
        page_number = doc.metadata.get("page_number", None)
        document_name = doc.metadata.get("document_name", None)
        vector_contexts.append({
            "content": doc.page_content,
            "page_number": str(page_number),
            "document_name": str(document_name),
            "score": str(score),
            "score_type": "vector"
        })
    return vector_contexts