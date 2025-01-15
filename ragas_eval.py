import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import ast

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

# Load environment variables
env_path = Path(__file__).resolve().parent.parent /'.env'
load_dotenv(dotenv_path=env_path)

try:
    api_key = ""   #ToDo solve why api, endpoint etc can't be passed to wrapper through .env
    api_version = "2024-07-01-preview"
    azure_endpoint = "https://sme-code-auzre-openai.openai.azure.com"
    model_name = "gpt-4o-mini"
    model_params = {
        "temperature": 0,
    }
    llm = AzureChatOpenAI(
        api_key=api_key,
        openai_api_version=api_version,
        azure_endpoint=azure_endpoint,
        azure_deployment=model_name,
        model=model_name,
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

#set the embedding function through HuggingFaceEmbeddings using the correct model
try:
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("HuggingFaceEmbeddings initialized successfully.")
except Exception as e:
    print(f"Error initializing HuggingFaceEmbeddings: {e}")
    exit(1)

#Building results folder path
results_folder_str = r"D:\GenAI\RAG\result"
results_filename = "rag_results_for_evaluation.json" #Filename of the results
results_filepath = os.path.join(results_folder_str, results_filename)

# Load the ground truth data
data_folder_str = r"D:\GenAI\RAG\data"
data_filename = "ground_truth.txt"
data_filepath = os.path.join(data_folder_str, data_filename)

try:
    with open(data_filepath, "r", encoding="utf-8") as f:
        list_string = f.read()
    # Clean up any potential whitespace or newline issues
    list_string = list_string.strip()
    # Safely evaluate the string
    loaded_data = ast.literal_eval(list_string)
    ground_truth_df = pd.DataFrame(loaded_data)
    print(ground_truth_df.shape)
except FileNotFoundError:
    print(f"Error: File not found at {data_filepath}")
except (SyntaxError, ValueError) as e:
    print(f"Error parsing data from {data_filepath}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print(f"Will evaluate results from: {results_filepath}")

#reading performance json for query answers
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
        "question": item["question"],
        "answer": item["response"],
        "context": item.get("contexts", []),  # Handle missing contexts gracefully
    })

ragas_df = pd.DataFrame(ragas_data)
final_df = pd.merge(ragas_df, ground_truth_df, on="question")
print(f"final merged df columns are : {final_df.columns}, shape of final merged df is {final_df.shape}")

#Creating RAGAS dataset from final_df
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

def evaluate_ragas(df):
    """
    Evaluates a DataFrame using RAGAS and all available metrics.
    Args:
        df: DataFrame with columns ['question', 'answer', 'context', 'ground_truth'].
    Returns:
        The RAGAS evaluation result or None if an error occurs.
    """
    ragas_data = []
    for index, row in df.iterrows():
        try:
            user_input = str(row["question"])
            response = str(row["answer"])

            context = row["context"]
            if isinstance(context, str):
                retrieved_contexts = [context]
            elif isinstance(context, list):
                retrieved_contexts = [str(item) for item in context]
            else:
                retrieved_contexts = [str(context)]

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
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
            ragas_data.append(sample)
        except Exception as e:
            print(f"Error creating SingleTurnSample for row {index}:") #Print the row index
            print(f"Row data: {row.to_dict()}")  # Print the row data
            import traceback
            traceback.print_exc() #Print the full traceback
            return None #Return None to stop execution
    
    evaluation_dataset = EvaluationDataset(samples=ragas_data)
    
    eval_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=hf_embeddings
    )
    return eval_result


eval_result = evaluate_ragas(final_df)
print(eval_result)

if eval_result:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ragas_filename = f"ragas_results_{timestamp}.json"  
    results_folder_str = r"D:\GenAI\RAG\result"
    results_filepath = os.path.join(results_folder_str, ragas_filename)

    try:
        df_results = eval_result.to_pandas()

        if ragas_filename.endswith(".json"):
            results_dict = {
                "overall_scores": df_results.iloc[0].to_dict(), # Extract overall scores
                "sample_by_sample_scores": df_results.iloc[1:].to_dict(orient="records") # Extract sample-by-sample scores
            }
            with open(results_filepath, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Ragas results saved to JSON: {results_filepath}")

        elif ragas_filename.endswith(".csv"):
            # Save as CSV (simpler and often preferred)
            df_results.to_csv(results_filepath, index=False)
            print(f"Ragas results saved to CSV: {results_filepath}")
        else:
            print("Unsupported file format. Please use .json or .csv")

    except Exception as e:
        print(f"Error saving RAGAS results: {e}")
else:
    print("RAGAS evaluation failed.")