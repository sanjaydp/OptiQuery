import re
import sqlite3

def get_columns_for_table(db_path: str, table: str) -> list[str]:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(f"PRAGMA table_info({table});")
            return [f"{table}.{row[1]}" for row in cursor.fetchall()]
    except Exception:
        return []

def extract_tables(sql: str) -> list[str]:
    tables = []
    from_join_pattern = re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", sql, re.IGNORECASE)
    for group in from_join_pattern:
        tables.extend([t for t in group if t])
    return list(set(tables))  # remove duplicates

def apply_auto_fixes(sql: str, db_path: str = None) -> tuple[str, list[str]]:
    fixes_applied = []
    modified_sql = sql

    # Fix SELECT *
    if re.search(r"\bSELECT\s+\*\b", modified_sql, re.IGNORECASE):
        table_names = extract_tables(modified_sql)
        if db_path and table_names:
            all_cols = []
            for table in table_names:
                cols = get_columns_for_table(db_path, table)
                all_cols.extend(cols)
            if all_cols:
                modified_sql = re.sub(
                    r"\bSELECT\s+\*\b", "SELECT " + ", ".join(all_cols),
                    modified_sql, flags=re.IGNORECASE
                )
                fixes_applied.append(f"Replaced `SELECT *` with: {', '.join(all_cols)}.")
            else:
                modified_sql = re.sub(
                    r"\bSELECT\s+\*\b", "SELECT column1, column2", modified_sql,
                    flags=re.IGNORECASE
                )
                fixes_applied.append("Replaced `SELECT *` with placeholders (no columns found).")
        else:
            modified_sql = re.sub(
                r"\bSELECT\s+\*\b", "SELECT column1, column2", modified_sql,
                flags=re.IGNORECASE
            )
            fixes_applied.append("Replaced `SELECT *` with placeholders (no DB or tables found).")

    # Fix JOIN without ON
    join_match = re.search(r"FROM\s+(\w+)\s+JOIN\s+(\w+)", modified_sql, re.IGNORECASE)
    if join_match and " on " not in modified_sql.lower():
        table1, table2 = join_match.group(1), join_match.group(2)
        fk = f"{table1}.{table2[:-1]}_id = {table2}.{table2[:-1]}_id"
        modified_sql = re.sub(
            r"\bJOIN\s+" + re.escape(table2),
            f"INNER JOIN {table2} ON {fk}",
            modified_sql,
            flags=re.IGNORECASE
        )
        fixes_applied.append(f"Added ON clause to JOIN: `{fk}`")

    # Add LIMIT if missing
    if "limit" not in modified_sql.lower():
        modified_sql += "\nLIMIT 100;"
        fixes_applied.append("Appended `LIMIT 100`.")

    return modified_sql, fixes_applied
