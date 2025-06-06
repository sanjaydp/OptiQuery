import sqlparse

def calculate_query_complexity(sql: str):
    """
    Assigns a complexity score to a SQL query based on:
    - Number of joins
    - Subqueries
    - Use of GROUP BY, ORDER BY
    - UNION/INTERSECT
    - CTEs (WITH clauses)
    Returns a score (0–100) and a label.
    """
    score = 0
    sql_lower = sql.lower()

    # Rule-based weights
    score += 10 * sql_lower.count(" join ")
    score += 15 * sql_lower.count(" select ") if "select" in sql_lower and " from (" in sql_lower else 0
    score += 5 if " group by " in sql_lower else 0
    score += 5 if " order by " in sql_lower else 0
    score += 10 if " union " in sql_lower or " intersect " in sql_lower else 0
    score += 10 if " with " in sql_lower else 0

    score = min(score, 100)  # Clamp to max 100

    if score < 20:
        label = "Low Complexity"
    elif score < 50:
        label = "Moderate Complexity"
    elif score < 80:
        label = "High Complexity"
    else:
        label = "Very High Complexity"

    return score, label
