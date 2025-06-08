import sqlparse
import re

def analyze_sql(query):
    """
    Analyzes a SQL query and returns a dictionary containing issues, suggestions, and complexity metrics.
    """
    try:
        issues = []
        suggestions = []
        
        # Convert to lowercase for case-insensitive analysis
        query_lower = query.lower()
        
        # Check SELECT *
        if "select *" in query_lower:
            issues.append("Using SELECT * — consider selecting specific columns for better performance")
            suggestions.append("Specify only needed columns to reduce I/O and network traffic")

        # Analyze JOIN conditions
        if "join" in query_lower:
            # Check for proper JOIN conditions
            if "on" not in query_lower:
                issues.append("JOIN without ON condition — may cause Cartesian product")
            else:
                # Analyze JOIN key indexing potential
                join_conditions = re.findall(r'on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', query_lower)
                for cond in join_conditions:
                    suggestions.append(f"Consider indexing JOIN columns: {cond[0]} and {cond[1]}")

        # Check WHERE clause
        if "where" not in query_lower:
            issues.append("No WHERE clause — might return too many rows")
        else:
            # Analyze WHERE conditions for indexing
            where_conditions = re.findall(r'where\s+(\w+\.\w+)\s*[=<>]', query_lower)
            for cond in where_conditions:
                suggestions.append(f"Consider indexing filter column: {cond}")

        # Check for LIMIT
        if "limit" not in query_lower:
            issues.append("No LIMIT clause — could lead to large result sets")
            suggestions.append("Add LIMIT clause to control result set size")

        # Check for proper table aliases
        if re.search(r'join\s+\w+\s+\w+\s+on', query_lower):
            suggestions.append("Consider using meaningful table aliases for better readability")

        # Check for column wildcards in specific tables
        table_wildcards = re.findall(r'select\s+(\w+)\.\*', query_lower)
        if table_wildcards:
            for table in table_wildcards:
                issues.append(f"Using {table}.* — consider selecting specific columns")

        # Calculate complexity metrics
        complexity = analyze_query_complexity(query)

        return {
            "issues": issues,
            "suggestions": suggestions,
            "complexity": complexity
        }
    except Exception as e:
        # Return a safe default structure in case of errors
        return {
            "issues": [f"Error analyzing query: {str(e)}"],
            "suggestions": [],
            "complexity": {
                "score": 0,
                "level": "Error",
                "factors": {
                    "joins": 0,
                    "conditions": 0,
                    "subqueries": 0,
                    "functions": 0
                }
            }
        }

def analyze_query_complexity(query):
    """Analyze query complexity based on various factors"""
    try:
        complexity = 0
        query_lower = query.lower()
        
        # Join complexity
        join_count = query_lower.count("join")
        complexity += join_count * 10
        
        # Where clause complexity
        where_operators = re.findall(r'(?:and|or|in|exists|like|between)', query_lower)
        complexity += len(where_operators) * 5
        
        # Subquery complexity
        subquery_count = query_lower.count("select") - 1
        complexity += subquery_count * 15
        
        # Aggregation complexity
        if any(op in query_lower for op in ["group by", "having", "order by"]):
            complexity += 10
        
        # Function calls
        function_count = len(re.findall(r'\w+\(', query_lower))
        complexity += function_count * 3
        
        # Normalize to 0-100 scale
        complexity = min(100, complexity)
        
        return {
            "score": complexity,
            "level": "High" if complexity > 70 else "Medium" if complexity > 40 else "Low",
            "factors": {
                "joins": join_count,
                "conditions": len(where_operators),
                "subqueries": subquery_count,
                "functions": function_count
            }
        }
    except Exception as e:
        # Return safe default values in case of error
        return {
            "score": 0,
            "level": "Error",
            "factors": {
                "joins": 0,
                "conditions": 0,
                "subqueries": 0,
                "functions": 0
            }
        }
