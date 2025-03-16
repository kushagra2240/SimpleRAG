import time
from typing import List, Tuple, Dict, Callable, Optional
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import RERANKER_TYPE, RERANKING_CHUNK_COUNT

class Reranker(ABC): # Abstract Base Class for Rerankers
    @abstractmethod
    def rerank(self, query: str, contexts: List[Dict], top_k: int) -> Tuple[List[Dict], float]:
        pass

class BERTReranker(Reranker): # Inherits from Reranker Base Class
    def __init__(self, model_name="yikuan8/Clinical-Longformer"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = self.tokenizer.model_max_length
        default_max_len = 4096

        if self.max_len is None:
            print(f"Warning: model_max_length is None for {model_name}. Using default of 512.")
            self.max_len = default_max_len
        elif self.max_len > 8192:
            print(f"Warning: model_max_length is extremely large ({self.max_len}) for {model_name}. Using default of 512.")
            self.max_len = default_max_len
        elif not isinstance(self.max_len, int) or self.max_len <= 0:
            print(f"Warning: model_max_length is not a valid integer ({self.max_len}) for {model_name}. Using default of 512.")
            self.max_len = default_max_len
        else:
            print(f"Max sequence length for {model_name}: {self.max_len}")

        self.max_seq_length = self.max_len

    def rerank(self, query: str, contexts: List[Dict], top_k: int) -> Tuple[List[Dict], float]:
        start_time = time.time()
        scores = []
        for context in contexts:
            inputs = self.tokenizer(
                query,
                context["content"],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_len
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            score = logits[0, 1].item()
            scores.append((context, score))

        ranked_contexts = sorted(scores, key=lambda x: x[1], reverse=True)
        reranked_contexts = [context for context, score in ranked_contexts[:top_k]]

        end_time = time.time()
        reranking_time = end_time - start_time

        return reranked_contexts, reranking_time


class LLMReranker(Reranker): # Wraps the LLM reranking function to fit the Reranker interface
    def rerank(self, query: str, contexts: List[Dict], top_k: int) -> Tuple[List[Dict], float]:
        start_time = time.time()
        reranked_contexts = llm_function_reranker(contexts, query, top_k) # Use the function from llm_operations
        end_time = time.time()
        reranking_time = end_time - start_time
        return reranked_contexts, reranking_time


# Example for FlashRankReranker 
# class FlashRankReranker(Reranker):
#     def __init__(self, model_name="your_flashrank_model"): # Example model name
#         # Initialize FlashRank model here
#         pass
#
#     def rerank(self, query: str, contexts: List[Dict], top_k: int) -> Tuple[List[Dict], float]:
#         start_time = time.time()
#         # Implement FlashRank reranking logic here
#         # ... use FlashRank model to rerank contexts ...
#         reranked_contexts = ... # Result from FlashRank
#         end_time = time.time()
#         reranking_time = end_time - start_time
#         return reranked_contexts, reranking_time


def get_reranker(reranker_type: str) -> Optional[Reranker]: # Factory function to get Reranker instance
    if reranker_type == "bert":
        return BERTReranker()
    elif reranker_type == "llm":
        return LLMReranker()
    # elif reranker_type == "flashrank": # Uncomment when FlashRankReranker is implemented
    #     return FlashRankReranker()
    elif reranker_type == "none":
        return None # Or return a NoReranker class if you want explicit no-op reranker
    else:
        print(f"Warning: Unknown reranker type: {reranker_type}. Using BERT reranker as default.")
        return BERTReranker()

def rerank_contexts(contexts: List[Dict], query: str) -> Tuple[List[Dict], float]: # Central reranking function

    reranker = get_reranker(RERANKER_TYPE) # Get the reranker based on config

    if reranker:
        reranked_contexts, reranking_time = reranker.rerank(query, contexts, RERANKING_CHUNK_COUNT)
        return reranked_contexts, reranking_time
    else:
        print("Reranking disabled or no valid reranker selected. Returning original contexts.")
        return contexts, 0.0 # Return original contexts and 0 reranking time