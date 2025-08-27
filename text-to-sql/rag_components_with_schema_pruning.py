import sqlite3
import re

def extract_relevant_table_names(question: str, schema_statements: list) -> set:
    """
    Given a question and list of CREATE TABLE statements,
    returns a set of table names that appear relevant.
    Uses simple keyword matching for now.
    """
    # Extract table names from CREATE TABLE statements
    table_names = []
    for stmt in schema_statements:
        match = re.search(r"CREATE TABLE\s+([^\s(]+)", stmt, re.IGNORECASE)
        if match:
            table_names.append(match.group(1))

    # Lowercase question for easier matching
    question_lower = question.lower()
    relevant_tables = set()
    for table in table_names:
        if table.lower() in question_lower:
            relevant_tables.add(table)
    return relevant_tables

def get_pruned_schema(schema_sql: str, question: str) -> str:
    """
    Retrieves only the CREATE TABLE statements from the database
    that are relevant to the user's question.
    """
    try:
        # Split schema_sql into CREATE TABLE statements
        schema_statements = re.findall(r"CREATE TABLE[\s\S]+?;", schema_sql, re.IGNORECASE)
        # Prune schema based on relevance
        relevant_tables = extract_relevant_table_names(question, schema_statements)
        pruned_statements = []
        for stmt in schema_statements:
            for table in relevant_tables:
                if f"CREATE TABLE {table}" in stmt or f"CREATE TABLE IF NOT EXISTS {table}" in stmt:
                    pruned_statements.append(stmt)
        print(f"Pruned schema contains {len(pruned_statements)} tables for question: {question}")
        return "\n\n".join(pruned_statements)
    except Exception as e:
        print(f"Failed to retrieve pruned schema: {e}")
        return ""