import sqlparse

def estimate_query_cost(sql: str) -> dict:
    score = 100
    reasons = []

    parsed = sqlparse.parse(sql)[0]
    tokens = [token for token in parsed.tokens if not token.is_whitespace]

    sql_lower = sql.lower()

    # ⚠️ SELECT *
    if "select *" in sql_lower:
        score -= 20
        reasons.append("Uses SELECT * — inefficient column selection.")

    # ⚠️ No WHERE clause
    if "where" not in sql_lower:
        score -= 15
        reasons.append("Missing WHERE clause — may cause full table scan.")

    # ⚠️ No LIMIT clause
    if "limit" not in sql_lower:
        score -= 10
        reasons.append("Missing LIMIT clause — could return too many rows.")

    # ⚠️ JOIN without ON
    if "join" in sql_lower and " on " not in sql_lower:
        score -= 15
        reasons.append("JOIN without ON clause — may result in Cartesian product.")

    # Determine risk level
    risk = "Low"
    if score < 80:
        risk = "Medium"
    if score < 50:
        risk = "High"

    return {
        "score": max(score, 0),
        "risk": risk,
        "issues": reasons
    }
