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

from langchain_openai import AzureOpenAIEmbeddings # Import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

# Load environment variables
env_path = Path(__file__).resolve().parent.parent /'.env'
load_dotenv(dotenv_path=env_path)

try:
    api_key = "83fbAYH1eWkbEAaCRmubWuJPMBPKt3sa9MdBm60TGxJJJgbJeVzbJQQJ99AKACYeBjFXJ3w3AAABACOG9tjp"  #ToDo solve why api, endpoint etc can't be passed to wrapper through .env
    api_version = "2024-07-01-preview"
    azure_endpoint = "https://sme-code-auzre-openai.openai.azure.com"
    model_name = "gpt-4o-mini"
    model_params = {
        "temperature": 0
    }
    llm = AzureChatOpenAI(
        api_key=api_key,
        openai_api_version=api_version,
        azure_endpoint=azure_endpoint,
        azure_deployment=model_name,
        model=model_name,
        **model_params
    )
    print("llm model instantiated")
except ValueError as e:
    print(e)
except Exception as e:
    print(e)

try:
    evaluator_llm = LangchainLLMWrapper(llm)
    print("llm wrapper enabled")

except Exception as e:
    print(e)

#set the embedding function through AzureOpenAIEmbeddings using text-embedding-small-v3 model
try:
    EMBEDDING_MODEL = "text-embedding-3-small" # Define embedding model name
    hf_embeddings = AzureOpenAIEmbeddings( # Use AzureOpenAIEmbeddings
        openai_api_version=api_version, # Ensure API version is passed
        azure_endpoint=azure_endpoint,    # Ensure endpoint is passed
        azure_deployment=EMBEDDING_MODEL, # Use embedding model deployment name
        model=EMBEDDING_MODEL,             # Pass model name as well
        api_key=api_key # Ensure API key is passed if needed
    )
    print(f"Embeddings initialized with text-embedding-small-v3 successfully.")
except Exception as e:
    print(f"Error initializing HuggingFaceEmbeddings: {e}")
    exit(1)

#Building results folder path
results_folder_str = r"D:\GenAI\RAG\result"
results_filename = "rag_results_horizon_bert_reranking_base_query_q_t0_cs1000_jb.json" 
results_filepath = os.path.join(results_folder_str, results_filename)

# Load the ground truth data from Excel
data_folder_str = r"D:\GenAI\RAG\data"
data_filename = "Benchmark_ExistingPOV_Horizon.xlsx" 
data_filepath = os.path.join(data_folder_str, data_filename)

try:
    # Read excel file
    excel_data = pd.read_excel(data_filepath)
    ground_truth_data = []
    for index, row in excel_data.iterrows():
        question = row['Question']
        expected_answer = row['Expected answer']
        if pd.isna(question) and pd.isna(expected_answer): # Check for blank rows to stop reading
            break # Stop reading when blank rows are encountered
        if not pd.isna(question) and not pd.isna(expected_answer): # Ensure both question and answer are not NaN
            ground_truth_data.append({'question': question, 'ground_truth': expected_answer})

    ground_truth_df = pd.DataFrame(ground_truth_data)
    # Clean up 'question' column in ground_truth_df
    ground_truth_df['question'] = ground_truth_df['question'].str.strip().str.lower() # Trim and lowercase
    print(f"ground_truth_df shape: {ground_truth_df.shape}")
    print("ground_truth_df 'question' column (first 5 values):\n", ground_truth_df['question'].head())
    print(f"ground_truth_df 'question' column dtype: {ground_truth_df['question'].dtype}")

except FileNotFoundError:
    print(f"Error: File not found at {data_filepath}")
    exit(1)
except Exception as e:
    print(f"Error reading data from {data_filepath}: {e}")
    exit(1)


print(f"Will evaluate results from: {results_filepath}")

try:
    with open(results_filepath, "r") as f:
        rag_results = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"No RAG results file found at: {results_filepath}")
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON format in: {results_filepath}")

ragas_data = []
for item in rag_results:
    ragas_data.append({
        "question": item["query"], 
        "answer": item["answer"],
        "context": item.get("retrieved_contexts", []),
    })

ragas_df = pd.DataFrame(ragas_data)
# Clean up 'question' column in ragas_df
ragas_df['question'] = ragas_df['question'].str.strip().str.lower() # Trim and lowercase
print(f"ragas_df shape: {ragas_df.shape}")
print("ragas_df 'question' column (first 5 values):\n", ragas_df['question'].head())
print(f"ragas_df 'question' column dtype: {ragas_df['question'].dtype}")


final_df = pd.merge(ragas_df, ground_truth_df, on="question", how='inner', indicator=True) 
final_df.rename(columns={"question": "user_input", "context": "retrieved_contexts"}, inplace=True)
final_df.drop(columns=['_merge'], inplace=True) # Remove _merge column


# VERIFY COLUMN NAMES - ADDED THIS LINE FOR DEBUGGING
print(f"final_df columns BEFORE evaluate_ragas: {final_df.columns}")
print(f"final merged df columns are : {final_df.columns}, shape of final merged df is {final_df.shape}")


#Creating RAGAS dataset from final_df
metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness] 

def evaluate_ragas(df):
    """
    Evaluates a DataFrame using RAGAS and answer_relevancy metric for the ENTIRE DataFrame.
    Args:
        df: DataFrame with columns ['user_input', 'answer', 'retrieved_contexts', 'ground_truth'].
    Returns:
        The RAGAS evaluation result or None if an error occurs.
    """
    ragas_data = []
    print("Columns received in evaluate_ragas function:", df.columns)

    for index, row in df.iterrows(): # Loop through all rows
        try:
            user_input = str(row["user_input"]) # Access user_input
            response = str(row["answer"])
            retrieved_contexts = row["retrieved_contexts"]
            ground_truth = row["ground_truth"]

            print(f"Debug: Processing question: {user_input[:50]}...") # Debug print for each question

            if isinstance(retrieved_contexts, str):
                contexts_list = [retrieved_contexts]
            elif isinstance(retrieved_contexts, list):
                # --- extract ONLY 'original_content' strings ---
                contexts_list = [str(item['original_content']) for item in retrieved_contexts if isinstance(item, dict) and 'original_content' in item]
            else:
                contexts_list = [str(retrieved_contexts)]

            ground_truth = row["ground_truth"]
            if isinstance(ground_truth, str):
                reference = ground_truth
            elif isinstance(ground_truth, list):
                reference = " ".join([str(item) for item in ground_truth])
            else:
                reference = str(ground_truth)

            sample = SingleTurnSample(
                user_input=user_input, 
                response=response,
                retrieved_contexts=contexts_list, 
                reference=reference,
            )
            ragas_data.append(sample) # Append sample for each row
        except Exception as e:
            print(f"Error creating SingleTurnSample for row {index}:")
            print(f"Row data: {row.to_dict()}")
            import traceback
            traceback.print_exc()
            continue # Continue to next row even if one fails

    evaluation_dataset = EvaluationDataset(samples=ragas_data)

    print("Columns of DataFrame JUST before evaluate call:", df.columns) # DEBUG PRINT 

    eval_result = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness], 
        llm=evaluator_llm,
        embeddings=hf_embeddings
    )
    return eval_result

eval_result = evaluate_ragas(final_df)
# print(eval_result)

if eval_result:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ragas_filename = f"ragas_results_reranking_bert.csv"
    results_folder_str = r"D:\GenAI\RAG\result"
    results_filepath = os.path.join(results_folder_str, ragas_filename)

    try:
        df_results = eval_result.to_pandas()

        if ragas_filename.endswith(".json"):
            results_list = [] 
            for index, row in df_results.iterrows(): 
                results_list.append(row.to_dict()) 

            with open(results_filepath, "w") as f:
                json.dump(results_list, f, indent=4) # Dump the list directly
            print(f"Ragas results saved to JSON: {results_filepath}")

        elif ragas_filename.endswith(".csv"):
            df_results.to_csv(results_filepath, index=False)
            print(f"Ragas results saved to CSV: {results_filepath}")
        else:
            print("Unsupported file format. Please use .json or .csv")

    except Exception as e:
        print(f"Error saving RAGAS results: {e}")
else:
    print("RAGAS evaluation failed.")