from typing import List, Dict

from bm25_retrieval import retrieve_bm25_contexts
from vector_db import retrieve_from_vector_db
from config import COMBINED_RETRIEVAL_COUNT # Or INITIAL_CONTEXT_COUNT_BEFORE_RERANKING if you renamed it


def retrieve_contexts(vector_store, bm25_index, query: str, k: int = COMBINED_RETRIEVAL_COUNT) -> list:
    """Retrieves contexts using both VectorDB and BM25 and combines them."""
    bm25_k = k // 2  # Example: Adjust split as needed, BM25 gets half of the initial k
    vector_k = k - bm25_k # VectorDB gets the remaining

    bm25_contexts = retrieve_bm25_contexts(bm25_index, vector_store, query, bm25_k)
    vector_contexts = retrieve_from_vector_db(vector_store, query, vector_k)

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
        page_number = combined_context.get('page_number') # Try to get directly
        document_name = combined_context.get('document_name') # Try to get directly

        if page_number is None or document_name is None: # If not found directly, try metadata
            metadata = combined_context.get('metadata', {}) # Get metadata, default to empty dict if missing
            page_number = metadata.get('page_number') # Get from metadata
            document_name = metadata.get('document_name') # Get from metadata

        final_contexts.append({
            "content": combined_context['content'],
            "page_number": str(page_number) if page_number else "N/A", # Handle potential None and convert to string
            "document_name": str(document_name) if document_name else "N/A", # Handle potential None and convert to string
            "bm25_score": combined_context.get('bm25_score'),
            "vector_score": combined_context.get('score'),
            "score_type": combined_context['score_type']
        })
    return final_contexts