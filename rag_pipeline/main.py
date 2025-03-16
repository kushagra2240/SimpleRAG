import json
from datetime import datetime
import pandas as pd

import sys
print("Python sys.path:")
for path in sys.path:
    print(path)
print("-" * 30) 

from vector_db import create_or_load_chroma_db, retrieve_from_vector_db
from llm_operations import generate_response, process_query_with_bm25 # Import process_query_with_bm25 from llm_operations
from reranking import get_reranker # Import get_reranker instead of BERTReranker directly
from bm25_retrieval import create_bm25_index, retrieve_bm25_contexts
from document_processing import load_documents_from_folder, split_documents
from retrieval import retrieve_contexts
from config import SOURCE_DIRECTORY, QUESTIONS_SOURCE, OUTPUT_FILE, RERANKING_CHUNK_COUNT, COMBINED_RETRIEVAL_COUNT, OPENAI_DEPLOYMENT_NAME, ENABLE_RERANKING

import tiktoken

def drop_below_empty_row(df: pd.DataFrame) -> pd.DataFrame:
    empty_row_index = df.isna().all(axis=1).idxmax()
    if pd.notna(empty_row_index):
        return df.loc[:empty_row_index-1]
    else:
        return df
        
def main():
    start_time = datetime.now()

    all_docs = load_documents_from_folder(SOURCE_DIRECTORY)
    print(f"There are {len(all_docs)} documents loaded")
    all_splits = split_documents(all_docs)
    print(f"The loader has split the documents into: {len(all_splits)} chunks")

    vector_store = create_or_load_chroma_db(all_splits)
    if vector_store is None:
        print("Error: Could not create or load Chroma collection. Exiting.")
        return

    bm25_index = create_bm25_index(all_splits) # Create BM25 index

    df_datasource = pd.read_excel(QUESTIONS_SOURCE)
    df_datasource = drop_below_empty_row(df_datasource)
    base_queries = [question.strip() for question in df_datasource['Question'] if pd.notna(question) and question != ""]

    all_results = []
    for query in base_queries:
        print(f"Processing query: {query}")
        query_result = process_query_with_bm25(vector_store, bm25_index, query, COMBINED_RETRIEVAL_COUNT, ENABLE_RERANKING) # Pass enable_reranking flag
        all_results.append(query_result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=4)

    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time.total_seconds()} seconds")

if __name__ == "__main__":
    main()