import sqlparse

def analyze_sql(query):
    issues = []
    if "*" in query.lower():
        issues.append("Using SELECT * — consider selecting specific columns.")
    if "limit" not in query.lower():
        issues.append("No LIMIT clause — could lead to large data scans.")
    if "join" in query.lower() and "on" not in query.lower():
        issues.append("JOIN without ON condition — may cause Cartesian product.")
    return issues
