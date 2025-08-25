import os
import json
import requests 

import sys
sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
SERVER_URL = "http://localhost:8081/completion"
SCHEMA_PATH = "./evaluation_data/mimic_iv.sql"
BENCHMARK_FILE_PATH = "./evaluation_data/annotated.json"
PREDICTION_FILE_PATH = "./input/res/prediction.json"
MAX_TOKENS = 256

PROMPT_TEMPLATE = """### Instruction:
You are a SQL expert. Given a database schema and a question, your job is to write a syntactically correct SQL query.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""

def run_inference_server(question: str, schema: str) -> str:
    """Sends a request to the running llama.cpp server."""
    full_prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
    
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": full_prompt,
        "n_predict": MAX_TOKENS,
        "stop": ["###"]
    }
    
    try:
        response = requests.post(SERVER_URL, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for bad status codes
        # The generated text is in the 'content' key of the JSON response
        generated_sql = response.json()['content']
        return generated_sql.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        return f"ERROR: Failed to get response from server for question: {question}"

def main():
    """Main function to run the benchmark using the server."""
    try:
        with open(SCHEMA_PATH, "r", encoding='utf-8') as f:
            schema_sql = f.read()
    except FileNotFoundError:
        print(f"Error: Schema file not found at '{SCHEMA_PATH}'")
        return

    try:
        with open(BENCHMARK_FILE_PATH, "r", encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{BENCHMARK_FILE_PATH}'")
        return

    predictions_dict = {}
    # Check if a prediction file already exists to resume
    if os.path.exists(PREDICTION_FILE_PATH):
        print(f"Resuming from existing prediction file: {PREDICTION_FILE_PATH}")
        try:
            with open(PREDICTION_FILE_PATH, "r", encoding='utf-8') as f:
                predictions_dict = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Warning: Could not read existing prediction file. Starting from scratch.")
            predictions_dict = {}

    print("Starting benchmark...")
    
    processed_count = 0
    # Optional: Slice for testing, e.g., benchmark_data[:5]
    for i, item in enumerate(benchmark_data):
        question = item.get("question")
        item_id = item.get("id")

        if not question or not item_id:
            print(f"Skipping item {i+1} due to missing data.")
            continue

        # Skip if already processed
        if item_id in predictions_dict:
            continue

        # Use the new server-based inference function
        generated_sql = run_inference_server(question, schema_sql)
        predictions_dict[item_id] = generated_sql
        
        processed_count += 1
        print(f"--- Processed {processed_count} (Total: {i+1}/{len(benchmark_data)}) (ID: {item_id}) ---")
        print(f"Question: {question}")
        # Encode to UTF-8 and decode back, replacing characters that can't be handled by the console
        printable_sql = generated_sql.encode('utf-8', 'replace').decode('utf-8')
        print(f"Generated SQL: {printable_sql}\n")

        # Save progress incrementally after each prediction
        os.makedirs(os.path.dirname(PREDICTION_FILE_PATH), exist_ok=True)
        with open(PREDICTION_FILE_PATH, "w", encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2)

    print(f"Benchmark finished. Predictions saved to {PREDICTION_FILE_PATH}")

if __name__ == "__main__":
    main()
