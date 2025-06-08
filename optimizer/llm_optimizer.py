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
    
    # Build a comprehensive context for the LLM
    context = [
        "You are an expert SQL optimization engine. Analyze the query considering:",
        "1. Query structure and complexity",
        "2. Table relationships and cardinality",
        "3. Index usage and opportunities",
        "4. Join strategies and order",
        "5. Subquery and CTE optimizations",
        "6. Data access patterns",
        "7. Potential bottlenecks",
        "",
        "IMPORTANT: When providing the optimized query:",
        "- Format SQL keywords in UPPERCASE",
        "- Use proper indentation for readability",
        "- Place each major clause (SELECT, FROM, WHERE, etc.) on a new line",
        "- Properly escape quotes and special characters in the JSON response",
        "- Ensure the query is syntactically valid"
    ]
    
    if schema_info:
        context.append("\nDatabase Schema:\n" + schema_info)
    
    if table_stats:
        context.append("\nTable Statistics:")
        for table, stats in table_stats.items():
            context.append(f"- {table}:")
            context.append(f"  • Rows: {stats['row_count']:,}")
            if 'indexes' in stats:
                context.append(f"  • Indexes: {', '.join(stats['indexes'].keys())}")
            if 'columns' in stats:
                context.append(f"  • Columns: {', '.join(stats['columns'])}")
    
    context.extend([
        "\nOptimization Guidelines:",
        "- Suggest indexes only when they provide significant benefit",
        "- Consider table sizes and data distribution",
        "- Evaluate trade-offs between different optimization strategies",
        "- Provide clear reasoning for each optimization",
        "- Be conservative with improvement estimates",
        "- Consider both read and write performance impacts",
        "- Evaluate memory and resource usage",
        "\nAnalyze and optimize the following query:"
    ])
    
    prompt = f"""{'\n'.join(context)}

SQL Query:
{query}

Provide a detailed analysis in JSON format. The response MUST be valid JSON with properly escaped strings:
{{
    "analysis": {{
        "complexity": "detailed analysis of query complexity",
        "bottlenecks": ["identified performance bottlenecks"],
        "data_access_patterns": "analysis of how data is accessed",
        "resource_usage": "expected resource utilization"
    }},
    "optimized_query": "SQL query with proper formatting and escaping",
    "index_suggestions": ["list of suggested indexes with clear justification"],
    "optimization_reasoning": "detailed explanation of each optimization",
    "estimated_improvement": "conservative improvement estimate with range",
    "confidence": "high/medium/low with explanation",
    "changes_made": ["specific changes with impact analysis"],
    "warnings": ["potential risks or trade-offs"],
    "validation_steps": ["suggested steps to validate improvements"]
}}

ENSURE that:
1. The response is valid JSON
2. All strings are properly escaped
3. The optimized query is properly formatted
4. SQL keywords are in UPPERCASE
5. Each major clause is on a new line"""

    # Use GPT-4 for better analysis
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Using the latest GPT-4 model
        messages=[
            {"role": "system", "content": "You are an expert SQL optimization engine with deep understanding of database internals, query planning, and performance tuning. Always provide responses in valid JSON format with properly escaped strings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Lower temperature for more focused responses
        max_tokens=2000
    )
    
    try:
        # First try to parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Format the optimized query if present
        if "optimized_query" in result:
            # Ensure proper SQL formatting
            formatted_query = format_sql_query(result["optimized_query"])
            result["optimized_query"] = formatted_query
        
        # Validate and enhance the response
        if result["optimized_query"].strip() == query.strip():
            result.update({
                "estimated_improvement": "0%",
                "confidence": "high",
                "changes_made": ["No changes needed - query appears to be already optimized"],
                "warnings": ["No optimization opportunities identified"],
                "validation_steps": ["Query is already well-optimized, no validation needed"]
            })
        else:
            # Add validation steps if not present
            if "validation_steps" not in result:
                result["validation_steps"] = [
                    "Compare execution plans of original and optimized queries",
                    "Test with representative data volumes",
                    "Monitor resource utilization",
                    "Verify result consistency"
                ]
        
        return result
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract and format the query
        try:
            content = response.choices[0].message.content
            # Try to extract query between SQL keywords
            query_match = re.search(r'SELECT.*?(?:;|$)', content, re.DOTALL | re.IGNORECASE)
            if query_match:
                extracted_query = format_sql_query(query_match.group(0))
            else:
                extracted_query = query  # Fall back to original query
            
            return {
                "optimized_query": extracted_query,
                "index_suggestions": [],
                "optimization_reasoning": "Error parsing optimization response. Extracted query from response.",
                "estimated_improvement": "0%",
                "confidence": "low",
                "changes_made": [],
                "warnings": ["Optimization analysis failed, but query was extracted"],
                "validation_steps": ["Manual review required"]
            }
        except Exception:
            return {
                "optimized_query": query,
                "index_suggestions": [],
                "optimization_reasoning": f"Error parsing optimization response: {str(e)}",
                "estimated_improvement": "0%",
                "confidence": "low",
                "changes_made": [],
                "warnings": ["Optimization analysis failed"],
                "validation_steps": ["Manual review required"]
            }
    except Exception as e:
        return {
            "optimized_query": query,
            "index_suggestions": [],
            "optimization_reasoning": f"Optimization error: {str(e)}",
            "estimated_improvement": "0%",
            "confidence": "low",
            "changes_made": [],
            "warnings": ["Optimization process failed"],
            "validation_steps": ["Manual review required"]
        }

def format_sql_query(query: str) -> str:
    """Format a SQL query with proper capitalization and indentation."""
    # List of SQL keywords to capitalize
    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
        "ON", "AND", "OR", "IN", "NOT", "EXISTS", "GROUP BY", "HAVING",
        "ORDER BY", "LIMIT", "OFFSET", "UNION", "ALL", "DESC", "ASC"
    ]
    
    # Capitalize keywords
    formatted = query.strip()
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        formatted = re.sub(pattern, keyword, formatted, flags=re.IGNORECASE)
    
    # Add newlines before major clauses
    major_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
    for keyword in major_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        formatted = re.sub(pattern, f"\n{keyword}", formatted)
    
    # Add indentation
    lines = formatted.split('\n')
    formatted_lines = []
    base_indent = " " * 4
    
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(keyword) for keyword in major_keywords):
            formatted_lines.append(stripped)
        else:
            formatted_lines.append(base_indent + stripped)
    
    return '\n'.join(formatted_lines).strip()

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
