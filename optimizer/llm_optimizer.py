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
    context = [
        "You are an expert SQL optimization engine. Analyze the query considering:",
        "1. Query structure and complexity",
        "2. Table relationships and cardinality",
        "3. Index usage and opportunities",
        "4. Join strategies and order",
        "5. Subquery and CTE optimizations",
        "6. Data access patterns",
        "7. Potential bottlenecks"
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

Provide a detailed analysis in JSON format:
{{
    "analysis": {{
        "complexity": "detailed analysis of query complexity",
        "bottlenecks": ["identified performance bottlenecks"],
        "data_access_patterns": "analysis of how data is accessed",
        "resource_usage": "expected resource utilization"
    }},
    "optimized_query": "the optimized SQL query",
    "index_suggestions": ["list of suggested indexes with clear justification"],
    "optimization_reasoning": "detailed explanation of each optimization",
    "estimated_improvement": "conservative improvement estimate with range",
    "confidence": "high/medium/low with explanation",
    "changes_made": ["specific changes with impact analysis"],
    "warnings": ["potential risks or trade-offs"],
    "validation_steps": ["suggested steps to validate improvements"]
}}"""

    # Use GPT-4 for better analysis
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Using the latest GPT-4 model
        messages=[
            {"role": "system", "content": "You are an expert SQL optimization engine with deep understanding of database internals, query planning, and performance tuning. Provide detailed, practical optimizations with clear reasoning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Lower temperature for more focused responses
        max_tokens=2000
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        
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
