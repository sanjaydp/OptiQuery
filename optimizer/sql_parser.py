import sqlparse
import re
from typing import Dict, List, Any

def analyze_sql(query: str) -> Dict[str, Any]:
    """
    Analyzes a SQL query and returns a dictionary containing issues, suggestions, and complexity metrics.
    """
    try:
        issues = []
        suggestions = []
        optimization_opportunities = []
        
        # Parse the query
        parsed = sqlparse.parse(query)[0]
        tokens = [token for token in parsed.tokens if not token.is_whitespace]
        query_lower = query.lower()
        
        # Analyze SELECT clause
        if "select" in query_lower:
            # Check for SELECT *
            if "select *" in query_lower:
                issues.append("Using SELECT * — consider selecting specific columns for better performance")
                suggestions.append("Specify only needed columns to reduce I/O and network traffic")
            
            # Check for unnecessary DISTINCT
            if "distinct" in query_lower and "group by" in query_lower:
                issues.append("DISTINCT used with GROUP BY — might be redundant")
                suggestions.append("Consider removing DISTINCT if GROUP BY provides the needed uniqueness")
            
            # Check for column wildcards in specific tables
            table_wildcards = re.findall(r'select\s+(\w+)\.\*', query_lower)
            for table in table_wildcards:
                issues.append(f"Using {table}.* — consider selecting specific columns")
        
        # Analyze JOIN conditions
        if "join" in query_lower:
            # Check for proper JOIN conditions
            if "on" not in query_lower:
                issues.append("JOIN without ON condition — may cause Cartesian product")
            
            # Analyze JOIN types and conditions
            join_conditions = re.findall(r'(\w+\s+join).*?on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)', query_lower)
            for join_type, left, right in join_conditions:
                # Check for OUTER joins that might be convertible to INNER
                if "outer" in join_type:
                    optimization_opportunities.append(f"Consider if OUTER JOIN on {left}={right} can be converted to INNER JOIN")
                suggestions.append(f"Consider indexing JOIN columns: {left} and {right}")
            
            # Check for multiple joins without parentheses
            if query_lower.count("join") > 1 and "(" not in query_lower:
                suggestions.append("Consider using parentheses to control JOIN order")
        
        # Analyze WHERE clause
        if "where" not in query_lower:
            issues.append("No WHERE clause — might return too many rows")
        else:
            # Check for function calls on indexed columns
            func_calls = re.findall(r'where.*?(\w+)\s*\(\s*(\w+\.\w+)\s*\)', query_lower)
            for func, col in func_calls:
                issues.append(f"Function call on column {col} prevents index usage")
                suggestions.append(f"Avoid using function {func} on {col} in WHERE clause")
            
            # Check for OR conditions
            if " or " in query_lower:
                optimization_opportunities.append("Consider replacing OR conditions with UNION for better index usage")
            
            # Check for LIKE with leading wildcard
            like_patterns = re.findall(r'where.*?(\w+)\s+like\s+[\'"]%', query_lower)
            for col in like_patterns:
                issues.append(f"LIKE with leading wildcard on {col} prevents index usage")
                suggestions.append(f"Avoid leading wildcard in LIKE pattern on {col}")
            
            # Analyze WHERE conditions for indexing
            where_conditions = re.findall(r'where\s+(\w+\.\w+)\s*[=<>]', query_lower)
            for cond in where_conditions:
                suggestions.append(f"Consider indexing filter column: {cond}")
        
        # Analyze GROUP BY and aggregations
        if "group by" in query_lower:
            # Check for missing HAVING clause with aggregations
            if any(func in query_lower for func in ["count(", "sum(", "avg(", "max(", "min("]):
                if "having" not in query_lower:
                    suggestions.append("Consider adding HAVING clause to filter aggregated results")
            
            # Check GROUP BY column order
            group_cols = re.findall(r'group by\s+(.+?)(?:having|order by|limit|$)', query_lower, re.S)
            if group_cols and len(group_cols[0].strip().split(',')) > 1:
                suggestions.append("Order GROUP BY columns by cardinality (most to least distinct values)")
        
        # Check for LIMIT
        if "limit" not in query_lower:
            issues.append("No LIMIT clause — could lead to large result sets")
            suggestions.append("Add LIMIT clause to control result set size")
        
        # Analyze subqueries
        subquery_count = query_lower.count("select") - 1
        if subquery_count > 0:
            # Check for correlated subqueries
            if re.search(r'\(\s*select.*?where.*?=\s*\w+\.', query_lower, re.S):
                issues.append("Correlated subquery detected — may cause performance issues")
                suggestions.append("Consider replacing correlated subquery with JOIN")
                optimization_opportunities.append("Replace correlated subquery with JOIN")
            
            # Suggest CTEs for multiple subquery references
            if subquery_count > 1:
                suggestions.append("Consider using CTEs (WITH clause) for better readability and maintenance")
        
        # Calculate complexity metrics
        complexity = analyze_query_complexity(query)
        
        # Estimate optimization impact
        impact_score = 0
        impact_score += len([i for i in issues if "performance" in i.lower()]) * 15
        impact_score += len([i for i in issues if "index" in i.lower()]) * 10
        impact_score += len([i for i in issues if "join" in i.lower()]) * 20
        impact_score += len(optimization_opportunities) * 15
        
        # Normalize score and convert to range
        impact_score = min(100, impact_score)
        estimated_impact = (
            "5-10%" if impact_score < 20
            else "10-20%" if impact_score < 40
            else "20-30%" if impact_score < 60
            else "30-50%" if impact_score < 80
            else "50%+"
        )
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "optimization_opportunities": optimization_opportunities,
            "complexity": complexity,
            "estimated_impact": estimated_impact
        }
    except Exception as e:
        return {
            "issues": [f"Error analyzing query: {str(e)}"],
            "suggestions": [],
            "optimization_opportunities": [],
            "complexity": {
                "score": 0,
                "level": "Error",
                "factors": {
                    "joins": 0,
                    "conditions": 0,
                    "subqueries": 0,
                    "functions": 0
                }
            },
            "estimated_impact": "Unknown"
        }

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """Analyze query complexity based on various factors"""
    try:
        complexity = 0
        query_lower = query.lower()
        factors = {
            "joins": 0,
            "conditions": 0,
            "subqueries": 0,
            "functions": 0,
            "grouping": 0,
            "unions": 0
        }
        
        # Join complexity
        factors["joins"] = query_lower.count("join")
        complexity += factors["joins"] * 10
        
        # Where clause complexity
        where_operators = re.findall(r'(?:and|or|in|exists|like|between)', query_lower)
        factors["conditions"] = len(where_operators)
        complexity += factors["conditions"] * 5
        
        # Subquery complexity
        factors["subqueries"] = query_lower.count("select") - 1
        complexity += factors["subqueries"] * 15
        
        # Function calls
        factors["functions"] = len(re.findall(r'\w+\(', query_lower))
        complexity += factors["functions"] * 3
        
        # Grouping complexity
        if "group by" in query_lower:
            group_cols = re.findall(r'group by\s+(.+?)(?:having|order by|limit|$)', query_lower, re.S)
            if group_cols:
                factors["grouping"] = len(group_cols[0].strip().split(','))
                complexity += factors["grouping"] * 5
        
        # UNION complexity
        factors["unions"] = query_lower.count("union")
        complexity += factors["unions"] * 10
        
        # Normalize to 0-100 scale
        complexity = min(100, complexity)
        
        # Determine level with more granular ranges
        level = (
            "Very Low" if complexity <= 20
            else "Low" if complexity <= 40
            else "Medium" if complexity <= 60
            else "High" if complexity <= 80
            else "Very High"
        )
        
        return {
            "score": complexity,
            "level": level,
            "factors": factors,
            "details": {
                "join_complexity": "High" if factors["joins"] > 3 else "Medium" if factors["joins"] > 1 else "Low",
                "filter_complexity": "High" if factors["conditions"] > 5 else "Medium" if factors["conditions"] > 2 else "Low",
                "subquery_complexity": "High" if factors["subqueries"] > 2 else "Medium" if factors["subqueries"] > 0 else "Low",
                "computation_complexity": "High" if factors["functions"] > 5 else "Medium" if factors["functions"] > 2 else "Low"
            }
        }
    except Exception as e:
        return {
            "score": 0,
            "level": "Error",
            "factors": {
                "joins": 0,
                "conditions": 0,
                "subqueries": 0,
                "functions": 0,
                "grouping": 0,
                "unions": 0
            },
            "details": {
                "join_complexity": "Error",
                "filter_complexity": "Error",
                "subquery_complexity": "Error",
                "computation_complexity": "Error"
            }
        }
