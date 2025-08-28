# prepare_finetune_data.py
import json
import os

# --- Configuration ---
# Path to the raw training data you have
RAW_TRAIN_DATA_PATH = "./train_data/annotated.json" 
# MODIFIED: You can now use your .sql file path here
SCHEMA_PATH = "./train_data/mimic_iv.sql"
# The output file that will be used for fine-tuning
OUTPUT_FINETUNE_FILE = "./train_data/finetune_data.jsonl"

# --- Prompt template for fine-tuning ---
# This structure is crucial. The model learns to fill in the '### SQL:' part.
FINETUNE_PROMPT_TEMPLATE = """### Instruction:
You are an expert SQLite developer. Your task is to convert a question into a syntactically correct SQLite query. Use the provided database schema.

### Schema:
{schema}

### Question:
{question}

### SQL:
{sql}"""

def format_schema_for_prompt(schema_path: str) -> str:
    """
    MODIFIED: This function now reads the content of any text file (like .sql)
    and returns it as a string for the prompt.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            # Simply read the whole file content
            schema_content = f.read()
        return schema_content
    except Exception as e:
        print(f"ERROR: Failed to read schema file: {e}")
        return ""

def create_finetune_dataset():
    """
    Reads the raw annotated data and converts it into a JSONL file
    formatted for supervised fine-tuning.
    """
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

            # Skip entries that are not valid question-SQL pairs
            if not question or not sql_query or sql_query.lower() == 'null':
                continue

            # Create the full instructional text
            full_text = FINETUNE_PROMPT_TEMPLATE.format(
                schema=schema_context,
                question=question,
                sql=sql_query
            )
            
            # Write each entry as a JSON object on a new line (JSONL format)
            f_out.write(json.dumps({"text": full_text}) + "\\n")
            count += 1

    print(f"Successfully created fine-tuning dataset with {count} valid entries.")
    print(f"   Output file: {OUTPUT_FINETUNE_FILE}")

if __name__ == "__main__":
    create_finetune_dataset()
