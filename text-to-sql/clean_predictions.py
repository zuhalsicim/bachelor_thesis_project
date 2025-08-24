import json
import re
import os

def extract_sql_cleverly(raw_text: str) -> str:
    """
    Extracts a SQL query from raw text using a multi-step fallback strategy.

    This function tries several patterns to find the SQL query, making it robust
    to common LLM output variations.

    Args:
        raw_text: The raw string output from the language model.

    Returns:
        The cleaned SQL query as a string, or an empty string if no query
        can be reliably extracted.
    """
    # 1. Primary Strategy: Look for a ```sql ... ``` markdown block.
    # This is the most reliable and common format.
    match = re.search(r"```sql\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Fallback 1: Look for a generic ``` ... ``` markdown block.
    # Sometimes the model forgets to add the 'sql' language identifier.
    match = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        # Check if the content looks like SQL to avoid extracting other code/text
        potential_sql = match.group(1).strip()
        if potential_sql.lower().lstrip().startswith('select'):
             return potential_sql

    # 3. Fallback 2: Find the last instance of 'SELECT' and extract from there.
    # This catches cases where the query is not in a markdown block at all.
    # We search from the end of the string to avoid capturing SQL from the prompt.
    last_select_pos = raw_text.lower().rfind('select')
    if last_select_pos != -1:
        # Take the substring from the last 'SELECT' to the end
        potential_sql = raw_text[last_select_pos:]
        
        # Clean up any trailing markdown backticks or explanations
        potential_sql = potential_sql.split('```')[0]
        
        return potential_sql.strip()

    # 4. Final Fallback: If no other pattern matches, return an empty string.
    return ""

def main():
    """
    Main function to load, clean, and save the prediction data using the
    advanced extraction logic.
    """
    # --- File Paths (Adjust if necessary) ---
    # Path to your original, "dirty" prediction file from the model
    original_file_path = r"C:\Uni\Bachelorarbeit\bachelor_thesis_project\text-to-sql\input\res\prediction.json"
    
    # Path where the new, cleverly cleaned file will be saved
    output_directory = r"C:\Uni\Bachelorarbeit\bachelor_thesis_project\evaluation_data"
    cleaned_file_path = os.path.join(output_directory, "prediction_cleaned_advanced.json")

    print(f"Loading original predictions from: {original_file_path}")

    # Load the original JSON data
    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {original_file_path}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: The file at {original_file_path} is not a valid JSON file.")
        return

    cleaned_predictions = {}
    queries_recovered = 0
    
    # Load the previous simple-cleaned file to compare results
    simple_cleaned_path = r"C:\Uni\Bachelorarbeit\bachelor_thesis_project\evaluation_data\prediction_cleaned.json"
    simple_cleaned_data = {}
    if os.path.exists(simple_cleaned_path):
        with open(simple_cleaned_path, 'r', encoding='utf-8') as f:
            simple_cleaned_data = json.load(f)

    print("\n--- Starting Advanced Cleaning Process ---")
    for question_id, generated_text in predictions.items():
        cleaned_sql = extract_sql_cleverly(generated_text)
        cleaned_predictions[question_id] = cleaned_sql

        # Check if this script recovered a query that was previously missed
        previous_result = simple_cleaned_data.get(question_id, "")
        if not previous_result and cleaned_sql:
            queries_recovered += 1
            # Removed emoji from the following line to prevent encoding errors
            print(f"[+] Recovered query for ID: {question_id}")


    # Save the cleaned predictions to the new file
    os.makedirs(output_directory, exist_ok=True)
    
    with open(cleaned_file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_predictions, f, indent=4)

    print("\n--- Cleaning Complete ---")
    # Removed emojis from the following lines to prevent encoding errors
    print(f"Successfully recovered {queries_recovered} additional queries!")
    print(f"Advanced cleaned file saved to: {cleaned_file_path}")


if __name__ == "__main__":
    main()
