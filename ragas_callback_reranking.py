import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import ast

from ragas import evaluate
#faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas.metrics.faithfulness import Faithfulness ## MODIFIED IMPORT: Explicitly import Faithfulness class
from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.evaluation import EvaluationResult

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

import time
from langchain_community.callbacks.manager import get_openai_callback

import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List, Optional
import datetime # ADDED for timestamp in log file ## ADDED

class PromptPrinterCallback(BaseCallbackHandler):
    def __init__(self, log_filepath="on_llm_start_events.log"): ## MODIFIED: Added log_filepath parameter, default log file name
        self.prompts: List[str] = []  # Store ALL prompts in a list
        self.log_filepath = log_filepath # Store filepath ## ADDED
        self.log_file = None # File handle, initialized later ## ADDED

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Add timestamp ## ADDED
        log_entry = f"Timestamp: {timestamp}\n" # Start log entry with timestamp ## ADDED
        log_entry += "on_llm_start CALLED!\n" # ADDED to log file
        log_entry += "Faithfulness Metric Prompt: on_llm_start event\n" # Marker ## ADDED to log file
        log_entry += f"Prompts received in on_llm_start: {prompts}\n" # ADDED to log file
        log_entry += f"Type of prompts: {type(prompts)}\n" # ADDED to log file
        log_entry += f"Initial self.prompts: {len(self.prompts)} elements\n" # Length instead of full list ## MODIFIED: Log length, not full list, ADDED to log file
        log_entry += f"Type of self.prompts: {type(self.prompts)}\n" # ADDED to log file
        try:
            if prompts:
                self.prompts.extend(prompts)
                log_entry += f"Length of self.prompts AFTER extend: {len(self.prompts)}\n" # ADDED to log file
        except Exception as e:
            log_entry += f"Error in on_llm_start during prompts.extend(): {e}\n" # ADDED to log file
            log_entry += f"Current self.prompts (before error): {len(self.prompts)} elements\n" # Length ## MODIFIED: Log length, ADDED to log file
            log_entry += f"Prompts value (causing error): {prompts}\n" # ADDED to log file

        if self.log_file: # Write to file only if file is open ## ADDED check
            self.log_file.write(log_entry + "\n---\n") # Write entry and separator to file ## MODIFIED: Write to file instead of print
            self.log_file.flush() # Ensure data is written immediately ## ADDED

    def get_prompts(self) -> List[str]:
        """Method to retrieve ALL captured prompts."""
        print(f"get_prompts() called. Current self.prompts: {self.prompts}") # Keep print for get_prompts for now
        return self.prompts

    def reset_prompts(self):
        """Method to reset the stored prompts for a new question."""
        print("reset_prompts() called!") # Keep print for reset_prompts for now
        self.prompts = []

    def open_log_file(self): # Method to open log file ## ADDED
        """Opens the log file in append mode."""
        self.log_file = open(self.log_filepath, "a", encoding="utf-8") # Open in append mode ## ADDED

    def close_log_file(self): # Method to close log file ## ADDED
        """Closes the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

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

# Instantiate PromptPrinterCallback *ONCE* here, outside the loop
prompt_callback_instance = PromptPrinterCallback()
llm.callbacks = [prompt_callback_instance] # Register the callback with the LLM object
print(f"Callback registered: {llm.callbacks}")
print("llm wrapper enabled with prompt callback (registered on LLM object)")


# File paths and data loading (same as before)
results_folder_str = r"D:\GenAI\RAG\result"
results_filename = "rag_results_horizon_bert_reranking_base_query_q_t0_cs1000_jb.json" # Corrected filename
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

print(f"Debugging: Type of 'faithfulness' object: {type(faithfulness)}")
print(f"Debugging: Attributes of 'faithfulness' object: {dir(faithfulness)}")
# metrics = [faithfulness] ## OLD metrics definition
faithfulness_metric = Faithfulness(llm=evaluator_llm) # Pass llm_for_eval to faithfulness metric ## MODIFIED: Initialize faithfulness metric with llm
metrics = [faithfulness_metric] # Use the configured metric ## MODIFIED: Use the configured metric

def evaluate_ragas(df, metrics_list, llm_for_eval, callback_instance): # Pass callback_instance
    """
    Evaluates a DataFrame row by row using RAGAS metrics, tracking token usage and time per row.
    Returns:
        df_results: DataFrame with RAGAS metrics AND token usage info per row, or None if error.
    """
    print(f"--- Debugging inside evaluate_ragas ---")
    print(f"Type of callback_instance: {type(callback_instance)}")
    print(f"Methods and attributes of callback_instance: {dir(callback_instance)}")
    print(f"--- End debugging evaluate_ragas ---")
    all_results = []
    print("Columns received in evaluate_ragas function:", df.columns)
    prompts_list = []
    encoding = tiktoken.encoding_for_model(model_name)
    all_prompts = []
    print(f"df shape is {df.shape}")

    for index, row in df.iterrows():
        row_result = {}
        total_context_tokens = 0
        prompts_per_metric = []
        try:
            user_input = str(row["user_input"])
            response = str(row["answer"])
            retrieved_contexts = row["retrieved_contexts"]
            ground_truth = row["ground_truth"]

            print(f"Debug: Processing question: {user_input[:50]}...\n")

            contexts_list = []
            if isinstance(retrieved_contexts, str):
                contexts_list = [retrieved_contexts]
            elif isinstance(retrieved_contexts, list):
                contexts_list = [str(item['original_content']) for item in retrieved_contexts if isinstance(item, dict) and 'original_content' in item]
            else:
                contexts_list = [str(retrieved_contexts)]
            print(f"-----------------------Contexts List length is {len(contexts_list)}-----------------------------")

            for context in contexts_list:
                total_context_tokens += len(encoding.encode(context))

            print(f"total number of context tokens is {total_context_tokens}")

            reference = ""
            if isinstance(ground_truth, str):
                reference = ground_truth
            elif isinstance(ground_truth, list):
                reference = " ".join([str(item) for item in ground_truth])
            else:
                reference = str(ground_truth)

            sample = SingleTurnSample( # Create SingleTurnSample
                user_input=user_input,
                response=response,
                retrieved_contexts=contexts_list,
                reference=reference,
            )
            evaluation_dataset = EvaluationDataset(samples=[sample]) # Create dataset with SINGLE sample

            start_time = time.time()
            try: # callback and evaluation per question
                callback_instance.reset_prompts() # Reset the SAME callback instance HERE before each question

                with get_openai_callback() as token_callback:
                    eval_result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=metrics_list,
                        llm=llm_for_eval, # Use the evaluator_llm passed as argument
                        embeddings=hf_embeddings
                    )
                    end_time = time.time()
                    generation_time = end_time - start_time
                    total_tokens = token_callback.total_tokens
                    prompt_tokens = token_callback.prompt_tokens
                    completion_tokens = token_callback.completion_tokens

                    print(f"  Question Evaluation Time: {generation_time:.2f} seconds")
                    print(f"  Question Tokens Used: {total_tokens}")
                    print(f"Prompts List right before get_prompts(): {callback_instance.prompts}") # Debugging print: Prompts right before get_prompts()

                    row_result = eval_result.to_pandas().iloc[0].to_dict()

                    # Add token info to the row_result dictionary
                    row_result['ragas_generation_time'] = generation_time
                    row_result['ragas_total_tokens'] = total_tokens
                    row_result['ragas_prompt_tokens'] = prompt_tokens
                    row_result['ragas_completion_tokens'] = completion_tokens
                    row_result['context_tokens'] = total_context_tokens

                    # --- Capture and store prompts for each metric ---
                    prompts_list = [] # MODIFIED: Initialize as empty list, we're not using get_prompts here anymore
                    print(f"Prompts List IMMEDIATELY after get_prompts(): {prompts_list}") # Debug print - will be empty now
                    for i, metric in enumerate(metrics_list): # Iterate through metrics and corresponding prompts
                        prompt = prompts_list[i] if i < len(prompts_list) else "Prompt not found" # Get prompt, handle case if prompt is missing
                        prompts_per_metric.append({ # Store question, metric, and prompt
                            "question": user_input,
                            "metric": metric.name, # Get metric name (e.g., "faithfulness")
                            "prompt": prompt
                        })
                    # --- End Prompt Capture ---


            except Exception as eval_error:
                print(f" Error during evaluation for question: {user_input[:50]}... Error: {eval_error}")
                row_result['error'] = str(eval_error)
                row_result['ragas_generation_time'] = None
                row_result['ragas_total_tokens'] = None
                row_result['ragas_prompt_tokens'] = None
                row_result['ragas_completion_tokens'] = None
                row_result['context_tokens'] = None


        except Exception as sample_error:
            print(f"Error processing row {index} (question: {row['user_input'][:50]}...): {sample_error}")
            row_result['error'] = str(sample_error)
            row_result['ragas_generation_time'] = None
            row_result['ragas_total_tokens'] = None
            row_result['ragas_prompt_tokens'] = None
            row_result['ragas_completion_tokens'] = None
            row_result['context_tokens'] = None

        row_result['user_input'] = row['user_input']
        row_result['answer'] = row['answer']
        row_result['ground_truth'] = row['ground_truth']
        all_results.append(row_result)
        all_prompts.append(prompts_per_metric)


    df_results = pd.DataFrame(all_results)
    return df_results, all_prompts

# Instantiate PromptPrinterCallback *ONCE* outside evaluate_ragas
prompt_callback_instance = PromptPrinterCallback()

prompt_callback_instance.open_log_file() # Open log file before evaluation ## ADDED

# Call evaluate_ragas with the final_df, metrics list, and the evaluator_llm and callback instance
df_results_per_question, prompts_list = evaluate_ragas(final_df, metrics, evaluator_llm, prompt_callback_instance)

prompt_callback_instance.close_log_file() # Close log file after evaluation ## ADDED

if df_results_per_question is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ragas_filename = f"ragas_results_per_question_reranking_bert_1metric.csv"
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

if prompts_list is not None:
    prompt_output_filepath = os.path.join(results_folder_str, "metrics_prompts.json")
    try:
        with open(prompt_output_filepath, "w") as f_prompts:
            json.dump(prompts_list, f_prompts, indent=4, ensure_ascii=False) # changed from prompt_data_for_json to all_prompts
        print(f"RAGAS metric prompts saved to JSON: {prompt_output_filepath}")
    except Exception as e:
        print(f"Error saving RAGAS metric prompts to JSON: {e}")
else:
    print("Could not populate prompts list")