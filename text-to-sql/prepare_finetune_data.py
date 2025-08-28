import json
import os

# --- Configuration ---
RAW_TRAIN_DATA_PATH = "./train_data/annotated.json"
SCHEMA_PATH = "./train_data/mimic_iv.sql"
OUTPUT_FINETUNE_FILE = "./train_data/finetune_data_2.jsonl"

FINETUNE_PROMPT_TEMPLATE = """### Instruction:
You are an expert SQLite developer. Your task is to convert a question into a syntactically correct SQLite query. Use the provided database schema.

### Schema:
{schema}

### Question:
{question}

### SQL:
{sql}"""

def format_schema_for_prompt(schema_path: str) -> str:
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_content = f.read()
        return schema_content
    except Exception as e:
        print(f"ERROR: Failed to read schema file: {e}")
        return ""

def create_finetune_dataset():
    schema_context = format_schema_for_prompt(SCHEMA_PATH)
    if not schema_context:
        print("Aborting: Schema could not be loaded.")
        return

    try:
        with open(RAW_TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Training data not found at '{RAW_TRAIN_DATA_PATH}'")
        return

    with open(OUTPUT_FINETUNE_FILE, 'w', encoding='utf-8') as f_out:
        count = 0
        for item in raw_data:
            question = item.get("question")
            sql_query = item.get("query")

            if not question or not sql_query or sql_query.lower() == 'null':
                continue

            full_text = FINETUNE_PROMPT_TEMPLATE.format(
                schema=schema_context,
                question=question,
                sql=sql_query
            )

            # Validate and write one JSON object per line
            obj = {"text": full_text}
            try:
                line = json.dumps(obj, ensure_ascii=False)
                # Test parse to make sure it's valid
                json.loads(line)
                f_out.write(line.strip() + "\n")
                count += 1
            except Exception as e:
                print(f"Skipping entry {count+1} due to serialization error: {e}")

    print(f"Successfully created fine-tuning dataset with {count} valid entries.")
    print(f"   Output file: {OUTPUT_FINETUNE_FILE}")

if __name__ == "__main__":
    create_finetune_dataset()