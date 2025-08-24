import json
import sqlite3
import os

def validate_ground_truth_queries():
    """
    Connects to the MIMIC-IV database and attempts to execute every
    ground-truth query from the provided JSON file to check for syntax errors.
    """
    # --- File Paths (Adjust if necessary) ---
    db_path = r"C:\Uni\Bachelorarbeit\bachelor_thesis_project\evaluation_data\mimic_iv.sqlite"
    ground_truth_file = r"C:\Uni\Bachelorarbeit\bachelor_thesis_project\evaluation_data\annotated.json"

    # --- Verification ---
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at '{db_path}'")
        return
    if not os.path.exists(ground_truth_file):
        print(f"ERROR: Ground truth file not found at '{ground_truth_file}'")
        return

    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Loading ground truth queries from: {ground_truth_file}\n")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- Initialization ---
    total_queries = 0
    successful_queries = 0
    failed_queries = []

    # --- Execution Loop ---
    for i, item in enumerate(data):
        query_id = item.get("id")
        query_sql = item.get("query")

        if not query_id or not query_sql:
            continue

        # Skip unanswerable questions, which are marked with the string 'null'
        if query_sql == 'null':
            continue

        total_queries += 1

        try:
            # We execute the query but don't need to fetch the results.
            # We just want to see if it's valid SQL that the DB can parse.
            cursor.execute(query_sql)
            successful_queries += 1
            # Print a dot for progress
            print('.', end='', flush=True)

        except sqlite3.Error as e:
            # If an error occurs, store the details
            failed_queries.append({
                "id": query_id,
                "query": query_sql,
                "error": str(e)
            })
            # Print an 'F' for failure
            print('F', end='', flush=True)
        
        # Print a newline every 100 queries to keep the output clean
        if (i + 1) % 100 == 0:
            print(f"  ({i+1}/{len(data)})")


    # --- Reporting ---
    print("\n\n--- Ground Truth Validation Complete ---")
    print(f"Total Queries Tested: {total_queries}")
    print(f"Successful Executions: {successful_queries}")
    print(f"Failed Executions: {len(failed_queries)}")
    print("----------------------------------------\n")

    if failed_queries:
        print("--- Details of Failed Queries ---")
        for failure in failed_queries:
            print(f"ID: {failure['id']}")
            print(f"  Query: {failure['query']}")
            print(f"  Error: {failure['error']}\n")
    else:
        print("🎉 All ground-truth queries executed successfully!")

    conn.close()

if __name__ == "__main__":
    validate_ground_truth_queries()
