import os
import json # Import json for potential error handling in no-rerank case
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import tiktoken
from openai import AzureOpenAI

from config import OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_API_BASE, OPENAI_DEPLOYMENT_NAME, LLM_TEMPERATURE
from reranking import rerank_contexts  # Import the reranking function
from retrieval import retrieve_contexts


def generate_response(context: str, query: str) -> Tuple[str, int, int, str]:
    start_time = datetime.now()

    prompt = f"""Use the following context to answer the question at the end.
    \n\nContext:\n{context}
    \n\nQuestion: {query}\n\n
    \n\nInstruction: Answer the users Question using the Context above.
    Keep your answer grounded in the facts of the Context.
    If the Context doesn’t contain the facts to answer the Question say "I couldn't find documents relavant to the topic":"""

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

    encoding = tiktoken.encoding_for_model(OPENAI_DEPLOYMENT_NAME)
    prompt_tokens = len(encoding.encode(prompt))
    completion_tokens = len(encoding.encode(response.choices[0].message.content.strip()))

    return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens, str(generation_time.total_seconds())


def process_query_with_bm25(vector_store, bm25_index, query: str, k: int, enable_reranking: bool) -> Dict:

    contexts_and_scores = retrieve_contexts(vector_store, bm25_index, query, k)
    reranked_contexts: List[Dict] = contexts_and_scores # Initialize with original contexts in case reranking is disabled
    reranking_time = 0.0 # Initialize reranking time to 0

    if enable_reranking:
        reranked_contexts, reranking_time = rerank_contexts(contexts_and_scores, query) # Call the unified reranking function
        context = "\n".join([content["content"] for content in reranked_contexts])
    else:
        # If reranking is disabled, use the initially retrieved contexts directly
        # and create a context string from the top contexts (e.g., top 5 if that was intended before reranking)
        top_n_no_rerank = 5 # Or use RERANKING_CHUNK_COUNT from config if you want consistency
        context_for_llm = contexts_and_scores[:min(top_n_no_rerank, len(contexts_and_scores))] # Take top N contexts
        context = "\n".join([c["content"] for c in context_for_llm])
        reranked_contexts = context_for_llm # To keep the reranked_contexts key consistent in output, even when no actual reranking happened
        print("Reranking is disabled. Using initial retrieval contexts.")


    response, total_context_tokens, completion_tokens, generation_time = generate_response(context, query)

    encoding = tiktoken.encoding_for_model(OPENAI_DEPLOYMENT_NAME)
    total_query_tokens = len(encoding.encode(query))

    return {
        "query": query,
        "answer": response,
        "retrieved_contexts": [
            {
                "original_content": og_context.get("content"),
                "original_page_number": og_context.get("page_number"),
                "original_document_name": og_context.get("document_name"),
                "original_similarity_score": og_context.get("score") if og_context.get('score_type') == 'vector' else og_context.get('bm25_score'),
                "score_type": og_context.get('score_type')
            }
            for og_context in contexts_and_scores
        ],
        "reranked_contexts": [
            {
                "combined_content": re_context.get("content"),
                "page_number": re_context.get("page_number"),
                "document_name": re_context.get("document_name"),
            }
            for re_context in reranked_contexts
        ],
        "query_tokens": total_query_tokens,
        "context_tokens": total_context_tokens,
        "response_tokens": completion_tokens,
        "total_tokens_sent": total_query_tokens + total_context_tokens,
        "total_tokens_received": completion_tokens,
        "generation_time": generation_time,
        "reranking_time": reranking_time,
    }