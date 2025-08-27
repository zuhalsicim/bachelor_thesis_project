# run_benchmark_rag_with_schema_pruning.py
import os
import json
import requests
import sys
sys.stdout.reconfigure(encoding='utf-8')
from rag_components_with_schema_pruning import get_pruned_schema

SCHEMA_PATH = "./evaluation_data/mimic_iv.sql"
BENCHMARK_FILE_PATH = "./evaluation_data/annotated.json"
PREDICTION_FILE_PATH = "./input/res/prediction_rag.json"
SERVER_URL = "http://localhost:8081/completion"
MAX_TOKENS = 2048

PROMPT_TEMPLATE = """### Instruction:
You are a SQL expert. Given a database schema and a question, your job is to write a syntactically correct SQL query.

### Database Schema:
{important_tables}
{full_schema}

### Question:
{question}

### SQL:
"""

def run_inference_with_rag(full_prompt: str) -> str:
    """Sends a request to the llama.cpp server with a full RAG prompt."""
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": full_prompt,
        "n_predict": MAX_TOKENS,
        "stop": ["###"]
    }
    try:
        response = requests.post(SERVER_URL, headers=headers, json=data)
        response.raise_for_status()
        generated_sql = response.json()['content']
        return generated_sql.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        return f"ERROR: Failed to get response from server for question."

def main():
    """Main function to run the benchmark using the RAG system with schema highlighting (zero-shot)."""
    try:
        with open(SCHEMA_PATH, "r", encoding='utf-8') as f:
            full_schema = f.read()
    except FileNotFoundError:
        print(f"Error: Schema file not found at '{SCHEMA_PATH}'")
        exit(1)
    try:
        with open(BENCHMARK_FILE_PATH, "r", encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{BENCHMARK_FILE_PATH}'")
        exit(1)

    predictions_dict = {}
    if os.path.exists(PREDICTION_FILE_PATH):
        print(f"Resuming from existing prediction file: {PREDICTION_FILE_PATH}")
        try:
            with open(PREDICTION_FILE_PATH, "r", encoding='utf-8') as f:
                predictions_dict = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            predictions_dict = {}

    print("\nStarting RAG benchmark with schema highlighting (zero-shot)...")

    processed_count = 0
    for i, item in enumerate(benchmark_data):
        question = item.get("question")
        item_id = item.get("id")
        pruned_schema = get_pruned_schema(full_schema, question)

        # Mark pruned schema as important if found
        if pruned_schema:
            important_tables = "-- IMPORTANT TABLES:\n" + pruned_schema + "\n\n-- FULL SCHEMA:"
        else:
            important_tables = "-- FULL SCHEMA:"

        if not question or not item_id:
            continue
        if item_id in predictions_dict:
            continue

        full_prompt = PROMPT_TEMPLATE.format(
            important_tables=important_tables,
            full_schema=full_schema,
            question=question
        )

        generated_sql = run_inference_with_rag(full_prompt)
        predictions_dict[item_id] = generated_sql

        processed_count += 1
        print(f"--- Processed {processed_count} (Total: {i+1}/{len(benchmark_data)}) (ID: {item_id}) ---")
        printable_sql = generated_sql.encode('utf-8', 'replace').decode('utf-8')
        print(f"Generated SQL: {printable_sql}\n")

        os.makedirs(os.path.dirname(PREDICTION_FILE_PATH), exist_ok=True)
        with open(PREDICTION_FILE_PATH, "w", encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2)

    print(f"Benchmark finished. Predictions saved to {PREDICTION_FILE_PATH}")

if __name__ == "__main__":
    main()

