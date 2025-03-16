from typing import List, Dict

from bm25_retrieval import retrieve_bm25_contexts
from vector_db import retrieve_from_vector_db
from config import COMBINED_RETRIEVAL_COUNT, ENABLE_BM25_RETRIEVAL # Import ENABLE_BM25_RETRIEVAL

def retrieve_contexts(vector_store, bm25_index, query: str, k: int = COMBINED_RETRIEVAL_COUNT) -> list:
    """Retrieves contexts, optionally using both VectorDB and BM25.
    If ENABLE_BM25_RETRIEVAL is False, only retrieves from VectorDB.
    """
    bm25_contexts = [] # Initialize as empty list

    if ENABLE_BM25_RETRIEVAL: # Check the flag
        bm25_k = k // 2  # Example: Adjust split as needed, BM25 gets half of the initial k
        bm25_contexts = retrieve_bm25_contexts(bm25_index, vector_store, query, bm25_k)
    else:
        print("BM25 retrieval is disabled. Retrieving only from VectorDB.")
        bm25_k = 0 # Set bm25_k to 0 if BM25 is disabled

    vector_k = k - bm25_k # VectorDB gets the remaining count 
    vector_contexts = retrieve_from_vector_db(vector_store, query, vector_k)

    print("\n--- VectorDB Contexts (from retrieval.py) ---") # Print VectorDB contexts
    for context in vector_contexts:
        print(f"VectorDB Context: {context}")
    print("--- End VectorDB Contexts ---\n")

    combined_contexts_dict = {}

    for bm25_context in bm25_contexts:
        content = bm25_context['content']
        combined_contexts_dict[content] = bm25_context

    for vector_context in vector_contexts:
        content = vector_context['content']
        combined_contexts_dict[content] = vector_context

    combined_contexts = list(combined_contexts_dict.values())

    # Sort combined contexts - you might want to define a combined ranking strategy.
    ranked_combined_contexts = sorted(
        combined_contexts,
        key=lambda x: float(x.get('score', -1)) if x.get('score_type') == 'vector' else float(x.get('bm25_score', -1)),
        reverse=True
    )[:k] # Take top k after combining, now based on combined retrieval count

    final_contexts = []
    for combined_context in ranked_combined_contexts:
        page_number = combined_context.get('page_number')
        document_name = combined_context.get('document_name')

        if page_number is None or document_name is None:
            metadata = combined_context.get('metadata', {})
            page_number = metadata.get('page_number')
            document_name = metadata.get('document_name')

        final_contexts.append({
            "content": combined_context['content'],
            "page_number": str(page_number) if page_number else "N/A",
            "document_name": str(document_name) if document_name else "N/A",
            "bm25_score": combined_context.get('bm25_score'),
            "vector_score": combined_context.get('score'),
            "score_type": combined_context['score_type']
        })
    return final_contexts