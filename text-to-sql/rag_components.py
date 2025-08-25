# rag_components.py
import sqlite3
import json
import random

def get_dynamic_schema(db_path: str) -> str:
    """
    Connects to the SQLite database and retrieves all CREATE TABLE statements.
    This is more informative than a simple column list.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_statements = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            create_statement = cursor.fetchone()[0]
            schema_statements.append(create_statement)
            
        conn.close()
        print(f"Dynamically retrieved schema for {len(tables)} tables.")
        return "\\n\\n".join(schema_statements)
    except Exception as e:
        print(f"Failed to retrieve dynamic schema: {e}")
        return ""

def get_few_shot_examples(examples_path: str, k: int = 3) -> str:
    """
    Loads and randomly selects 'k' few-shot examples to guide the model.
    """
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # Ensure we don't try to sample more examples than available
        k = min(k, len(examples))
        selected_examples = random.sample(examples, k)
        
        formatted_examples = ""
        for ex in selected_examples:
            formatted_examples += f"Question: {ex['question']}\\nSQL: {ex['query']}\\n\\n"
        
        print(f"✅ Loaded {k} few-shot examples.")
        return formatted_examples
    except Exception as e:
        print(f"❌ ERROR: Could not load few-shot examples from '{examples_path}': {e}")
        return ""

