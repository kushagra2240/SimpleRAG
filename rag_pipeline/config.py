import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
LLM_TEMPERATURE = 0.5
RANKER_LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
COLLECTION_NAME = f"Horizon_{CHUNK_SIZE}_cleaned"

# RETRIEVAL_TOP_K = 15  # old reranking code. remove after test
RERANKING_CHUNK_COUNT = 5
COMBINED_RETRIEVAL_COUNT = 25 # Total chunks from BM25 + Vector DB before reranking

SOURCE_FILES_PATHS = ""
SOURCE_DIRECTORY = r"D:\GenAI\RAG\data\POV Dataset"
QUESTIONS_SOURCE = r'D:\GenAI\RAG\data\Benchmark_ExistingPOV_Horizon.xlsx'
OUTPUT_FILE = r"D:\GenAI\RAG\result\rag_results_horizon_longform_bert_bm_25_reranking_base_query_q_t0_cs2000_jb.json"
DB_BUILDING_BATCH_SIZE = 100 # Define batch size for DB building

ENABLE_RERANKING = True # Flag to enable or disable reranking
RERANKER_TYPE = "bert" # Or "llm", or "flashrank" (when implemented), or "none"

ENABLE_BM25_RETRIEVAL = True