# run_benchmark_rag.py
import os
import json
import requests 
import sys
from rag_components import get_dynamic_schema, get_few_shot_examples

sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
SERVER_URL = "http://localhost:8081/completion"
# --- RAG Configuration ---
# Point to the actual database file for dynamic schema retrieval
DB_PATH = "./evaluation_data/mimic_iv.sqlite"
# Point to the new few-shot examples file
FEW_SHOT_EXAMPLES_PATH = "./evaluation_data/few_shot_examples.json"

BENCHMARK_FILE_PATH = "./evaluation_data/annotated.json"
PREDICTION_FILE_PATH = "./input/res/prediction_rag.json" # Use a new prediction file
MAX_TOKENS = 256

# --- Enhanced RAG Prompt Template ---
PROMPT_TEMPLATE = """### Instruction:
You are an expert SQLite developer. Your task is to convert a question into a syntactically correct SQLite query.
- Use the provided database schema and examples to inform your query.
- Use table aliases to prevent ambiguity.

### Database Schema:
{schema}

### Examples:
{examples}

### Question:
{question}

### SQL:
"""

def run_inference_with_rag(question: str, schema: str, examples: str) -> str:
    """Sends a request to the llama.cpp server with a full RAG prompt."""
    full_prompt = PROMPT_TEMPLATE.format(schema=schema, examples=examples, question=question)
    
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": full_prompt,
        "n_predict": MAX_TOKENS,
        "stop": ["###", ";", "\n\n"] # Stop generation more effectively
    }
    
    try:
        response = requests.post(SERVER_URL, headers=headers, json=data)
        response.raise_for_status()
        generated_sql = response.json()['content']
        return generated_sql.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        return f"ERROR: Failed to get response from server for question: {question}"

def main():
    """Main function to run the benchmark using the RAG system."""
    
    # --- RAG Pre-computation ---
    # Retrieve the dynamic schema and few-shot examples once at the start
    print("Initializing RAG components...")
    schema_context = get_dynamic_schema(DB_PATH)
    few_shot_examples = get_few_shot_examples(FEW_SHOT_EXAMPLES_PATH)
    if not schema_context:
        print("Could not build schema context. Aborting benchmark.")
        return

    try:
        with open(BENCHMARK_FILE_PATH, "r", encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{BENCHMARK_FILE_PATH}'")
        return

    predictions_dict = {}
    if os.path.exists(PREDICTION_FILE_PATH):
        print(f"Resuming from existing prediction file: {PREDICTION_FILE_PATH}")
        try:
            with open(PREDICTION_FILE_PATH, "r", encoding='utf-8') as f:
                predictions_dict = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            predictions_dict = {}

    print("\nStarting RAG benchmark...")
    
    processed_count = 0
    for i, item in enumerate(benchmark_data):
        question = item.get("question")
        item_id = item.get("id")

        if not question or not item_id:
            continue
        if item_id in predictions_dict:
            continue

        # Use the RAG-enhanced inference function
        generated_sql = run_inference_with_rag(question, schema_context, few_shot_examples)
        predictions_dict[item_id] = generated_sql
        
        processed_count += 1
        print(f"--- Processed {processed_count} (Total: {i+1}/{len(benchmark_data)}) (ID: {item_id}) ---")
        print(f"Question: {question}")
        printable_sql = generated_sql.encode('utf-8', 'replace').decode('utf-8')
        print(f"Generated SQL: {printable_sql}\n")

        os.makedirs(os.path.dirname(PREDICTION_FILE_PATH), exist_ok=True)
        with open(PREDICTION_FILE_PATH, "w", encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2)

    print(f"Benchmark finished. Predictions saved to {PREDICTION_FILE_PATH}")

if __name__ == "__main__":
    main()

