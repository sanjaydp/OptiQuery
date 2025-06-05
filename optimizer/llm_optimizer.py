from openai import OpenAI
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

def optimize_query(query: str) -> str:
    prompt = f"""You are a SQL optimization assistant. Improve the following query for performance:

SQL Query:
{query}

Return only the improved SQL query.
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites SQL queries for performance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def explain_optimization(original_query: str, optimized_query: str) -> str:
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
