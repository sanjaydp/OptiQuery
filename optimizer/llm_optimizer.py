import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import json
import re

load_dotenv()

def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets.")

    return OpenAI(api_key=api_key)

def optimize_query(query: str, schema_info: str = None, table_stats: dict = None) -> str:
    client = get_openai_client()
    
    # Clean and normalize the input query
    cleaned_query = clean_sql_query(query)
    
    # Build a comprehensive context for the LLM
    context = [
        "You are an expert SQL optimization engine. Your task is to optimize the following SQL query.",
        "Provide your response in the following format:",
        "",
        "```json",
        "{",
        '    "optimized_query": "The optimized SQL query with proper formatting",',
        '    "analysis": {',
        '        "complexity": "Analysis of query complexity",',
        '        "bottlenecks": ["List of bottlenecks"],',
        '        "data_access_patterns": "How data is accessed",',
        '        "resource_usage": "Resource utilization analysis"',
        '    },',
        '    "changes_made": ["List of changes"],',
        '    "estimated_improvement": "Estimated improvement percentage",',
        '    "confidence": "high/medium/low",',
        '    "warnings": ["Any warnings"],',
        '    "validation_steps": ["Steps to validate"]',
        "}",
        "```",
        "",
        "Follow these rules:",
        "1. Format SQL keywords in UPPERCASE",
        "2. Use proper indentation",
        "3. Place each clause on a new line",
        "4. Properly escape quotes",
        "5. Ensure syntactically valid SQL",
        "6. Keep the original query logic intact"
    ]
    
    if schema_info:
        context.extend([
            "",
            "Database Schema:",
            schema_info
        ])
    
    if table_stats:
        context.extend([
            "",
            "Table Statistics:"
        ])
        for table, stats in table_stats.items():
            context.extend([
                f"- {table}:",
                f"  • Rows: {stats['row_count']:,}",
                f"  • Indexes: {', '.join(stats['indexes'].keys()) if 'indexes' in stats else 'None'}",
                f"  • Columns: {', '.join(stats['columns']) if 'columns' in stats else 'Unknown'}"
            ])
    
    context.extend([
        "",
        "Original Query:",
        cleaned_query
    ])
    
    # Use GPT-4 for better analysis
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert SQL optimization engine. Always respond with valid JSON containing a properly formatted SQL query."
                },
                {
                    "role": "user",
                    "content": "\n".join(context)
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        # Parse the response
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Validate and clean the optimized query
            if "optimized_query" in result:
                result["optimized_query"] = format_sql_query(result["optimized_query"])
            
            # Ensure all required fields are present
            result.setdefault("analysis", {
                "complexity": "Analysis not available",
                "bottlenecks": [],
                "data_access_patterns": "Not analyzed",
                "resource_usage": "Not analyzed"
            })
            result.setdefault("changes_made", [])
            result.setdefault("estimated_improvement", "0%")
            result.setdefault("confidence", "low")
            result.setdefault("warnings", [])
            result.setdefault("validation_steps", [
                "Compare execution plans",
                "Test with representative data",
                "Monitor resource usage",
                "Verify results"
            ])
            
            # If no optimization was performed
            if result["optimized_query"].strip() == cleaned_query.strip():
                result.update({
                    "estimated_improvement": "0%",
                    "confidence": "high",
                    "changes_made": ["No changes needed - query appears to be already optimized"],
                    "warnings": ["No optimization opportunities identified"],
                    "validation_steps": ["Query is already well-optimized"]
                })
            
            return result
            
        except json.JSONDecodeError:
            # Try to extract query from non-JSON response
            extracted_query = extract_sql_from_text(response.choices[0].message.content)
            if extracted_query:
                formatted_query = format_sql_query(extracted_query)
                return create_fallback_response(formatted_query, "Extracted query from non-JSON response")
            else:
                return create_fallback_response(cleaned_query, "Could not parse optimization response")
                
    except Exception as e:
        return create_fallback_response(cleaned_query, f"Optimization error: {str(e)}")

def clean_sql_query(query: str) -> str:
    """Clean and normalize a SQL query."""
    # Remove multiple whitespace
    query = ' '.join(query.split())
    
    # Ensure proper spacing around operators
    operators = ['=', '<', '>', '<=', '>=', '<>', '!=', '+', '-', '*', '/', '%']
    for op in operators:
        query = query.replace(op, f' {op} ')
    
    # Clean up spacing around parentheses
    query = re.sub(r'\(\s+', '(', query)
    query = re.sub(r'\s+\)', ')', query)
    
    # Ensure single space after commas
    query = re.sub(r',\s*', ', ', query)
    
    # Remove any trailing semicolon for consistency
    query = query.rstrip(';').strip()
    
    return query

def extract_sql_from_text(text: str) -> str:
    """Extract a SQL query from text content."""
    # Try to find SQL between markdown code blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find SQL between quotes in JSON-like text
    sql_match = re.search(r'"optimized_query":\s*"(.*?)"', text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try to find anything that looks like a SQL query
    sql_match = re.search(r'SELECT\s+.*?(?:;|$)', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(0).strip()
    
    return ""

def create_fallback_response(query: str, error_message: str) -> dict:
    """Create a fallback response when optimization fails."""
    return {
        "optimized_query": format_sql_query(query),
        "analysis": {
            "complexity": "Analysis failed",
            "bottlenecks": ["Analysis not available due to error"],
            "data_access_patterns": "Not analyzed",
            "resource_usage": "Not analyzed"
        },
        "changes_made": [],
        "estimated_improvement": "0%",
        "confidence": "low",
        "warnings": [error_message],
        "validation_steps": ["Manual review required"],
        "optimization_reasoning": error_message
    }

def format_sql_query(query: str) -> str:
    """Format a SQL query with proper capitalization, indentation and spacing."""
    # List of SQL keywords to capitalize
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN',
        'ON', 'AND', 'OR', 'IN', 'NOT IN', 'EXISTS', 'NOT EXISTS', 'GROUP BY', 'HAVING',
        'ORDER BY', 'LIMIT', 'OFFSET', 'UNION', 'UNION ALL', 'INSERT INTO', 'VALUES',
        'UPDATE', 'SET', 'DELETE FROM', 'CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'AS',
        'ASC', 'DESC', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'WITH'
    ]
    
    # Clean up extra whitespace first
    query = ' '.join(query.split())
    
    # Capitalize keywords
    # Sort keywords by length in reverse order to handle compound keywords correctly
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = r'(?i)\b' + re.escape(keyword) + r'\b'
        query = re.sub(pattern, keyword, query)
    
    # Add newlines before major clauses
    major_clauses = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
        'LIMIT', 'UNION', 'INSERT', 'UPDATE', 'DELETE'
    ]
    for clause in major_clauses:
        query = re.sub(r'\s+' + clause + r'\b', '\n' + clause, query)
    
    # Add newlines and indentation for JOIN clauses
    join_types = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN']
    for join in join_types:
        query = re.sub(r'\s+' + join + r'\b', '\n    ' + join, query)
    
    # Add indentation for ON clauses in JOINs
    query = re.sub(r'\s+ON\s+', '\n        ON ', query)
    
    # Add indentation for AND/OR conditions
    lines = query.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Determine indentation level
        if any(stripped.startswith(clause) for clause in major_clauses):
            indent_level = 0
        elif any(stripped.startswith(join) for join in join_types):
            indent_level = 1
        elif stripped.startswith('ON '):
            indent_level = 2
        elif stripped.startswith(('AND ', 'OR ')):
            if indent_level < 2:
                indent_level = 2
        
        # Apply indentation
        formatted_lines.append('    ' * indent_level + stripped)
    
    # Ensure proper spacing around operators
    query = '\n'.join(formatted_lines)
    operators = ['=', '<', '>', '<=', '>=', '<>', '!=', '+', '-', '*', '/', '%']
    for op in operators:
        query = re.sub(r'\s*' + re.escape(op) + r'\s*', f' {op} ', query)
    
    # Clean up spacing around parentheses
    query = re.sub(r'\(\s+', '(', query)
    query = re.sub(r'\s+\)', ')', query)
    
    # Ensure single space after commas
    query = re.sub(r',\s*', ', ', query)
    
    # Remove extra blank lines
    query = re.sub(r'\n\s*\n', '\n', query)
    
    return query.strip()

def explain_optimization(original_query: str, optimized_query: str) -> str:
    client = get_openai_client()
    prompt = f"""Explain the optimization differences between these two SQL queries.

Original:
{original_query}

Optimized:
{optimized_query}

List the improvements in bullet points."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains SQL optimizations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
