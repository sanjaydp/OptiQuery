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
    context = "You are a SQL optimization expert. "
    
    if schema_info:
        context += f"\n\nDatabase Schema:\n{schema_info}"
    
    if table_stats:
        context += "\n\nTable Statistics:\n"
        for table, stats in table_stats.items():
            context += f"- {table}: {stats['row_count']:,} rows\n"
    
    context += "\n\nOptimize the following query for better performance. Consider:"
    context += "\n- Table sizes and data distribution"
    context += "\n- Appropriate indexing suggestions"
    context += "\n- Join order optimization"
    context += "\n- Subquery optimization"
    context += "\n- Proper column selection"
    
    prompt = f"""{context}

SQL Query:
{query}

Return the following in a JSON format:
{{
    "optimized_query": "the optimized SQL query",
    "index_suggestions": ["list of suggested indexes"],
    "optimization_reasoning": "explanation of optimizations made",
    "estimated_improvement": "estimated % improvement in performance"
}}"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a SQL performance expert that provides detailed optimization suggestions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format={ "type": "json" }
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        return {
            "optimized_query": query,
            "index_suggestions": [],
            "optimization_reasoning": "Error parsing optimization response",
            "estimated_improvement": "0%"
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
