from typing import List, Dict
from rank_bm25 import BM25Okapi

def tokenize_text(text: str) -> List[str]:
    return text.lower().split(" ")

def create_bm25_index(all_splits: List[Dict]) -> BM25Okapi:
    document_contents = [split.page_content for split in all_splits]
    tokenized_corpus = [tokenize_text(content) for content in document_contents]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("BM25 index created.")
    return bm25_index

def retrieve_bm25_contexts(bm25_index: BM25Okapi, vector_store, query: str, k: int) -> List[Dict]:
    tokenized_query = tokenize_text(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_results_with_scores = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k]

    bm25_contexts = []
    all_documents_in_chroma = vector_store.get(include=['metadatas', 'documents'])

    for doc_index, score in bm25_results_with_scores:
        if 0 <= doc_index < len(all_documents_in_chroma['ids']):
            bm25_contexts.append({
                "content": all_documents_in_chroma['documents'][doc_index],
                "metadata": all_documents_in_chroma['metadatas'][doc_index],
                "bm25_score": score,
                "score_type": "bm25"
            })
    return bm25_contexts