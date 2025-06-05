import re

def apply_auto_fixes(sql: str) -> tuple:
    fixes_applied = []
    modified_sql = sql

    # Replace SELECT * with placeholder columns
    if re.search(r"\bselect\s+\*\b", modified_sql, re.IGNORECASE):
        modified_sql = re.sub(r"\bselect\s+\*\b", "SELECT column1, column2", modified_sql, flags=re.IGNORECASE)
        fixes_applied.append("Replaced SELECT * with specific column placeholders.")

    # Add LIMIT 100 if no LIMIT
    if "limit" not in modified_sql.lower():
        modified_sql += "\nLIMIT 100;"
        fixes_applied.append("Added LIMIT 100 to restrict excessive row returns.")

    # Warn for JOIN without ON (log only, don’t modify)
    if "join" in modified_sql.lower() and " on " not in modified_sql.lower():
        fixes_applied.append("Query may contain JOIN without ON clause — verify join logic.")

    return modified_sql, fixes_applied
