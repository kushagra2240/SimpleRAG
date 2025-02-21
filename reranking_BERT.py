import os
import json
from datetime import datetime
import time
from typing import Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from openai import AzureOpenAI

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import tiktoken
import torch 

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
LLM_TEMPERATURE = 0 
RANKER_LLM_TEMPERATURE = 0 
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
  
COLLECTION_NAME = f"Horizon_{CHUNK_SIZE}_cleaned"

RETRIEVAL_TOP_K = 15 # number of chunks to be retrieved
RERANKING_CHUNK_COUNT = 5

SOURCE_FILES_PATHS = ""
SOURCE_DIRECTORY = r"D:\GenAI\RAG\data\POV Dataset"
QUESTIONS_SOURCE = r'D:\GenAI\RAG\data\Benchmark_ExistingPOV_Horizon.xlsx'
OUTPUT_FILE = r"D:\GenAI\RAG\result\rag_results_horizon_longform_bert_reranking_base_query_q_t0_cs2000_jb.json"

# New: BERT Reranker
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
# model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")

class BERTReranker:
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
        elif not isinstance(self.max_len, int) or self.max_len <= 0: # Added type and value check
            print(f"Warning: model_max_length is not a valid integer ({self.max_len}) for {model_name}. Using default of 512.")
            self.max_len = default_max_len
        else:
            print(f"Max sequence length for {model_name}: {self.max_len}")

        self.max_seq_length = self.max_len # Store the *validated* max length
        # print(f"Max sequence length for {model_name}: {self.tokenizer.model_max_length}")

    def rerank(self, query: str, contexts: List[Dict], top_k: int = 5) -> Tuple[List[Dict], float]:
        start_time = time.time() 
        scores = []
        for context in contexts:
            inputs = self.tokenizer(
                query,
                context["content"],
                return_tensors="pt",
                truncation=True,  # Enable truncation
                max_length=self.max_len # Use model's max length
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            score = logits[0, 1].item()  # Probability of being acceptable
            scores.append((context, score))

        ranked_contexts = sorted(scores, key=lambda x: x[1], reverse=True)
        reranked_contexts = [context for context, score in ranked_contexts[:top_k]]

        end_time = time.time()  # Record end time
        reranking_time = end_time - start_time

        return reranked_contexts, reranking_time

def clean_text(text: str) -> str:
    """
    Cleans up text by removing extra whitespace, tabs, and newlines.
    You can expand this function to include more cleaning steps as needed.
    """
    text = text.replace('\t', ' ').replace('\n', ' ')  # Replace tabs and newlines with spaces
    text = ' '.join(text.split())  # Remove extra spaces (multiple spaces become single space)
    return text

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
                cleaned_text = clean_text(text)
                metadata = {"page_number": page_num + 1, "document_name": os.path.basename(file_path)}
                all_docs.append({"content": cleaned_text, "metadata": metadata})
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
            try:
                reader = PdfReader(file_path)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    cleaned_text = clean_text(text)
                    metadata = {"page_number": page_num + 1, "document_name": filename}
                    all_docs.append({"content": cleaned_text, "metadata": metadata})
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
    # print (f"----------------- the length of retrieved context for Query {query} are {len(contexts)}--------------------------------")
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

    # --- Using tiktoken for token counting ---
    encoding = tiktoken.encoding_for_model(OPENAI_DEPLOYMENT_NAME) # Get encoding for your model
    prompt_tokens = len(encoding.encode(prompt)) # Tokenize and count prompt tokens
    completion_tokens = len(encoding.encode(response.choices[0].message.content.strip())) # Tokenize and count completion tokens
    # --- End tiktoken token counting ---
    
    # prompt_tokens = len(prompt)
    # completion_tokens = len(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip(), prompt_tokens, completion_tokens, str(generation_time.total_seconds())

def llm_rerank_contexts(contexts: List[Dict], query: str, rerank_k: int = RERANKING_CHUNK_COUNT) -> List[Dict]:
    """Reranks retrieved contexts using an LLM, keeping original context information."""

    if len(contexts) <= rerank_k:
        return contexts

    reranker_prompt = f"""You are an expert at combining relevant information from multiple sources.
Given the following list of contexts related to a question, combine the information in the MOST RELEVANT contexts into {rerank_k} concise summaries.  
Focus on the information that directly answers the question.  If a context is not relevant, do not include it in any of the summaries. 
Keep track of which original contexts contributed to each summary.

Question: {query}

Contexts:
"""
    for i, context in enumerate(contexts):
        reranker_prompt += f"Context {i+1}:\nContent: {context['content']}\nPage Number: {context['page_number']}\nDocument Name: {context['document_name']}\n\n"

    reranker_prompt += f"""
    \n\nReturn the {rerank_k} combined and concise context summaries in the following *valid* JSON format.  
For each summary, indicate which original context numbers (1, 2, 3...) were used to create it.  
If you cannot generate valid JSON, or if you do not find enough relevant information to create {rerank_k} summaries, 
return a JSON array with FEWER THAN {rerank_k} summaries, or even an EMPTY JSON array [] if you find NO relevant information.
[
  {{"combined_context": "summary 1", "original_contexts": [1, 3]}},  // Example: Summary 1 uses contexts 1 and 3
  {{"combined_context": "summary 2", "original_contexts": [2, 4, 5]}}, // Example: Summary 2 uses contexts 2, 4, and 5
  ...
]
    """

    client = AzureOpenAI(api_key=os.getenv("OPEN_API_KEY"),
                      api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                      )

    try:
        reranker_response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": reranker_prompt}],
            temperature=RANKER_LLM_TEMPERATURE  
        )
        print("Reranker LLM Response:")  # Debugging: Print the raw response
        print(reranker_response.choices[0].message.content)

        try:
            reranked_contexts_json = json.loads(reranker_response.choices[0].message.content.strip())
        except json.JSONDecodeError as e:
            print(f"Inner Error decoding JSON response from reranker: {e}")
            reranked_contexts_json = []  # Treat as empty if JSON is invalid

        reranked_contexts = []
        for combined_context_data in reranked_contexts_json:
            original_context_indices = combined_context_data.get("original_contexts", [])
            original_contexts = [contexts[i - 1] for i in original_context_indices if 0 < i <= len(contexts)]

            combined_context = {
                "content": combined_context_data.get("combined_context"),
                "page_number": None,
                "document_name": None,
                "score": None,
                "original_contexts": original_contexts
            }
            reranked_contexts.append(combined_context)
        if not reranked_contexts:  # Fallback if LLM returns nothing
            print("Warning: Reranker LLM did not return any combined contexts. Using original contexts.")
            return contexts[:min(RERANKING_CHUNK_COUNT, len(contexts))]

        return reranked_contexts

    except Exception as e:
        print(f"Error in reranking: {e}")
        return []  # Return an empty list if it fails


# Function to process a single query
def process_query(vector_store, query: str, k: int = 10) -> Dict:
    contexts_and_scores = retrieve_contexts(vector_store, query, k=k)
    
    # BERT Reranking
    bert_reranker = BERTReranker()
    bert_reranked_contexts, reranking_time = bert_reranker.rerank(query, contexts_and_scores)
    
    context = "\n".join([content["content"] for content in bert_reranked_contexts])
    
    response, total_context_tokens, completion_tokens, generation_time = generate_response(context, query)
    
    # total_query_tokens = len(query)
    # --- Using tiktoken for token counting ---
    encoding = tiktoken.encoding_for_model(OPENAI_DEPLOYMENT_NAME) # Get encoding for your model
    total_query_tokens = len(encoding.encode(query)) # Tokenize and count query tokens
    
    return {
        "query": query,
        "answer": response,
        "retrieved_contexts": [ 
            {
                "original_content": og_context.get("content"),
                "original_page_number": og_context.get("page_number"),
                "original_document_name": og_context.get("document_name"),
                "original_similarity_score": og_context.get("score")
            }
            for og_context in contexts_and_scores
        ],
        "reranked_contexts": [
            {
                "combined_content": re_context.get("content"),
                "page_number": re_context.get("page_number"),
                "document_name": re_context.get("document_name"),
            } # Renamed key, added original_contexts
            for re_context in bert_reranked_contexts
        ],
        "query_tokens": total_query_tokens,
        "context_tokens": total_context_tokens,
        "response_tokens": completion_tokens,
        "total_tokens_sent": total_query_tokens + total_context_tokens,
        "total_tokens_received": completion_tokens,
        "generation_time": generation_time,
        "reranking_time": reranking_time
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
        query_result = process_query(vector_store, query, k=RETRIEVAL_TOP_K)
        all_results.append(query_result)

    with open(OUTPUT_FILE, "w") as f: 
        json.dump(all_results, f, indent=4)
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time.total_seconds()} seconds")

if __name__ == "__main__":
    main()