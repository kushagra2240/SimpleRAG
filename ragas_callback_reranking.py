import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import ast

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness

from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import EvaluationResult

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

import time
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables
env_path = Path(__file__).resolve().parent.parent /'.env'
load_dotenv(dotenv_path=env_path)

# API Keys and Model Setup (same as before)
try:
    api_key = "83fbAYH1eWkbEAaCRmubWuJPMBPKt3sa9MdBm60TGxJJJgbJeVzbJQQJ99AKACYeBjFXJ3w3AAABACOG9tjp"  
    api_version = "2024-07-01-preview"
    azure_endpoint = "https://sme-code-auzre-openai.openai.azure.com" 
    model_name = "gpt-4o-mini"
    model_params = {"temperature": 0}
    llm = AzureChatOpenAI(
        api_key=api_key,
        openai_api_version=api_version,
        azure_endpoint=azure_endpoint,
        azure_deployment=model_name,
        model=model_name,
        **model_params
    )
    print("llm model instantiated")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

try:
    evaluator_llm = LangchainLLMWrapper(llm)
    print("llm wrapper enabled")
except Exception as e:
    print(f"Error initializing LLMWrapper: {e}")
    exit(1)


# Embeddings setup (same as before)
try:
    EMBEDDING_MODEL = "text-embedding-3-small" 
    hf_embeddings = AzureOpenAIEmbeddings( 
        openai_api_version=api_version, 
        azure_endpoint=azure_endpoint,    
        azure_deployment=EMBEDDING_MODEL, 
        model=EMBEDDING_MODEL,             
        api_key=api_key 
    )
    print(f"Embeddings initialized with {EMBEDDING_MODEL} successfully.")
except Exception as e:
    print(f"Error initializing Embeddings: {e}")
    exit(1)


# File paths and data loading (same as before)
results_folder_str = r"D:\GenAI\RAG\result"
results_filename = "rag_results_horizon_bert_reranking_base_query_q_t0_cs1000_jb.json"
results_filepath = os.path.join(results_folder_str, results_filename)
data_folder_str = r"D:\GenAI\RAG\data"
data_filename = "Benchmark_ExistingPOV_Horizon.xlsx"
data_filepath = os.path.join(data_folder_str, data_filename)


try: # Load Ground Truth Data (same as before)
    excel_data = pd.read_excel(data_filepath)
    ground_truth_data = []
    for index, row in excel_data.iterrows():
        question = row['Question']
        expected_answer = row['Expected answer']
        if pd.isna(question) and pd.isna(expected_answer): break
        if not pd.isna(question) and not pd.isna(expected_answer):
            ground_truth_data.append({'question': question, 'ground_truth': expected_answer})
    ground_truth_df = pd.DataFrame(ground_truth_data)
    ground_truth_df['question'] = ground_truth_df['question'].str.strip().str.lower()
except Exception as e:
    print(f"Error loading ground truth data: {e}")
    exit(1)

try: # Load RAG Results Data (same as before)
    with open(results_filepath, "r") as f:
        rag_results = json.load(f)
except Exception as e:
    print(f"Error loading RAG results: {e}")
    exit(1)

ragas_data = [] # Process RAG Results Data (same as before)
for item in rag_results:
    ragas_data.append({
        "question": item["query"],
        "answer": item["answer"],
        "context": item.get("retrieved_contexts", []),
    })
ragas_df = pd.DataFrame(ragas_data)
ragas_df['question'] = ragas_df['question'].str.strip().str.lower()


final_df = pd.merge(ragas_df, ground_truth_df, on="question", how='inner', indicator=True)
final_df.rename(columns={"question": "user_input", "context": "retrieved_contexts"}, inplace=True)
final_df.drop(columns=['_merge'], inplace=True)


metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

def evaluate_ragas(df, metrics_list): # Modified function to accept metrics_list
    """
    Evaluates a DataFrame row by row using RAGAS metrics, tracking token usage and time per row.
    Returns:
        df_results: DataFrame with RAGAS metrics AND token usage info per row, or None if error.
    """
    all_results = [] # List to store results for each row
    print("Columns received in evaluate_ragas function:", df.columns)

    for index, row in df.iterrows():
        row_result = {} # Dictionary to store results for current row
        try:
            user_input = str(row["user_input"])
            response = str(row["answer"])
            retrieved_contexts = row["retrieved_contexts"]
            ground_truth = row["ground_truth"]

            print(f"Debug: Processing question: {user_input[:50]}...")

            contexts_list = [] # Context list 
            if isinstance(retrieved_contexts, str): contexts_list = [retrieved_contexts]
            elif isinstance(retrieved_contexts, list): contexts_list = [str(item['original_content']) for item in retrieved_contexts if isinstance(item, dict) and 'original_content' in item]
            else: contexts_list = [str(retrieved_contexts)]

            reference = "" 
            if isinstance(ground_truth, str): reference = ground_truth
            elif isinstance(ground_truth, list): reference = " ".join([str(item) for item in ground_truth])
            else: reference = str(ground_truth)


            sample = SingleTurnSample( # Create SingleTurnSample
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts_list,
                reference=reference,
            )
            evaluation_dataset = EvaluationDataset(samples=[sample]) # Create dataset with SINGLE sample


            start_time = time.time() 
            try: # callback and evaluation per question
                with get_openai_callback() as token_callback: 
                    eval_result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=metrics_list, 
                        llm=evaluator_llm,
                        embeddings=hf_embeddings
                    )
                end_time = time.time()
                generation_time = end_time - start_time
                total_tokens = token_callback.total_tokens
                prompt_tokens = token_callback.prompt_tokens
                completion_tokens = token_callback.completion_tokens

                print(f"  Question Evaluation Time: {generation_time:.2f} seconds")
                print(f"  Question Tokens Used: {total_tokens}")


                row_result = eval_result.to_pandas().iloc[0].to_dict() 

                # Add token info to the row_result dictionary
                row_result['ragas_generation_time'] = generation_time
                row_result['ragas_total_tokens'] = total_tokens
                row_result['ragas_prompt_tokens'] = prompt_tokens
                row_result['ragas_completion_tokens'] = completion_tokens


            except Exception as eval_error: 
                print(f"  Error during evaluation for question: {user_input[:50]}... Error: {eval_error}")
                row_result['error'] = str(eval_error) 
                row_result['ragas_generation_time'] = None 
                row_result['ragas_total_tokens'] = None
                row_result['ragas_prompt_tokens'] = None
                row_result['ragas_completion_tokens'] = None


        except Exception as sample_error: 
            print(f"Error processing row {index} (question: {row['user_input'][:50]}...): {sample_error}")
            row_result['error'] = str(sample_error) 
            row_result['ragas_generation_time'] = None 
            row_result['ragas_total_tokens'] = None
            row_result['ragas_prompt_tokens'] = None
            row_result['ragas_completion_tokens'] = None

        row_result['user_input'] = row['user_input'] 
        row_result['answer'] = row['answer'] 
        row_result['ground_truth'] = row['ground_truth'] 
        all_results.append(row_result) 


    df_results = pd.DataFrame(all_results) 
    return df_results 

# Call evaluate_ragas with the final_df and the metrics list
df_results_per_question = evaluate_ragas(final_df, metrics) 

if df_results_per_question is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ragas_filename = f"ragas_results_per_question_reranking_bert.csv" 
    results_folder_str = r"D:\GenAI\RAG\result"
    results_filepath = os.path.join(results_folder_str, ragas_filename)

    try:
        if ragas_filename.endswith(".csv"):
            df_results_per_question.to_csv(results_filepath, index=False, encoding='utf-8') 
            print(f"Ragas results saved to CSV (per question): {results_filepath}")
        elif ragas_filename.endswith(".json"):
            df_results_per_question.to_json(results_filepath, orient="records", indent=4, force_ascii=False) 
            print(f"Ragas results saved to JSON (per question): {results_filepath}")
        else:
            print("Unsupported file format. Please use .csv or .json")

    except Exception as e:
        print(f"Error saving RAGAS results: {e}")
else:
    print("RAGAS evaluation failed for all questions.")