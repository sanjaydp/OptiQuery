from openai import OpenAI
import streamlit as st

def add_inline_comments(sql_query: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""
You are a SQL code reviewer. For each line of the following SQL query, add a comment at the end of the line explaining improvements or confirming it's good. Use this format:

<SQL LINE>   -- <your comment>

Use ⚠️ for suggestions and ✅ for good practices. Keep it concise.

SQL:
{sql_query}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
