import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import json

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
    context = "You are a SQL optimization expert. Your task is to optimize the given SQL query for better performance."
    
    if schema_info:
        context += f"\n\nDatabase Schema:\n{schema_info}"
    
    if table_stats:
        context += "\n\nTable Statistics:\n"
        for table, stats in table_stats.items():
            context += f"- {table}: {stats['row_count']:,} rows\n"
            if 'indexes' in stats:
                context += f"  Existing indexes: {', '.join(stats['indexes'].keys())}\n"
    
    context += "\n\nOptimize the following query for better performance. Consider:"
    context += "\n- Table sizes and data distribution"
    context += "\n- Appropriate indexing suggestions"
    context += "\n- Join order optimization"
    context += "\n- Subquery optimization"
    context += "\n- Proper column selection"
    context += "\n- Query rewriting techniques"
    context += "\n\nBe conservative with improvement estimates. Only suggest significant changes if they will clearly improve performance."
    
    prompt = f"""{context}

SQL Query:
{query}

Return the following in a JSON format:
{{
    "optimized_query": "the optimized SQL query",
    "index_suggestions": ["list of suggested indexes"],
    "optimization_reasoning": "detailed explanation of each optimization made and its expected impact",
    "estimated_improvement": "estimated % improvement (be conservative, use ranges like '10-15%' if uncertain)",
    "confidence": "high/medium/low - your confidence in the optimization's impact",
    "changes_made": ["list each specific change made"],
    "warnings": ["any potential risks or trade-offs"]
}}"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a SQL performance expert that provides detailed optimization suggestions. Always respond with valid JSON. Be conservative in performance improvement estimates."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        # Ensure the optimized query is different from the original
        if result["optimized_query"].strip() == query.strip():
            result["estimated_improvement"] = "0%"
            result["confidence"] = "high"
            result["changes_made"] = ["No changes needed - query appears to be already optimized"]
            result["warnings"] = ["No optimization opportunities identified"]
        return result
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract the optimized query from the raw response
        content = response.choices[0].message.content
        return {
            "optimized_query": query,  # Return original query as fallback
            "index_suggestions": [],
            "optimization_reasoning": "Error parsing optimization response. Using original query.",
            "estimated_improvement": "0%",
            "confidence": "low",
            "changes_made": [],
            "warnings": [f"Error during optimization: {str(e)}"]
        }
    except Exception as e:
        return {
            "optimized_query": query,
            "index_suggestions": [],
            "optimization_reasoning": f"Error during optimization: {str(e)}",
            "estimated_improvement": "0%",
            "confidence": "low",
            "changes_made": [],
            "warnings": ["Optimization failed - using original query"]
        }

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
