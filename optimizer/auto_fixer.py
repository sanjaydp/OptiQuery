import re

def apply_auto_fixes(sql: str) -> tuple:
    fixes_applied = []
    modified_sql = sql

    # Fix SELECT *
    if re.search(r"\bselect\s+\*\b", modified_sql, re.IGNORECASE):
        modified_sql = re.sub(r"\bselect\s+\*\b", "SELECT column1, column2", modified_sql, flags=re.IGNORECASE)
        fixes_applied.append("Replaced SELECT * with specific column placeholders.")

    # Fix JOIN without ON → add INNER JOIN ... ON guess
    join_pattern = re.search(r"FROM\s+(\w+)\s+JOIN\s+(\w+)", modified_sql, re.IGNORECASE)
    if join_pattern and " on " not in modified_sql.lower():
        table1, table2 = join_pattern.group(1), join_pattern.group(2)
        # Guess foreign key
        fk = f"{table1}.{table2[:-1]}_id = {table2}.{table2[:-1]}_id"
        modified_sql = re.sub(r"\bJOIN\s+" + re.escape(table2), f"INNER JOIN {table2} ON {fk}", modified_sql, flags=re.IGNORECASE)
        fixes_applied.append(f"Converted JOIN to INNER JOIN with ON clause: `{fk}`")

    # Add LIMIT if missing
    if "limit" not in modified_sql.lower():
        modified_sql += "\nLIMIT 100;"
        fixes_applied.append("Added LIMIT 100 to restrict excessive row returns.")

    return modified_sql, fixes_applied
