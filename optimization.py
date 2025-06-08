import openai
import streamlit as st
from typing import Optional

def optimize_query(query: str) -> str:
    """Analyze and optimize the SQL query."""
    # Basic analysis without OpenAI
    analysis = []
    
    # Check for SELECT *
    if "SELECT *" in query.upper():
        analysis.append("⚠️ Consider specifying only needed columns instead of SELECT *")
    
    # Check for missing WHERE clause
    if "WHERE" not in query.upper():
        analysis.append("⚠️ Query lacks a WHERE clause - this might return too many rows")
    
    # Check for missing indexes (basic check)
    if "WHERE" in query.upper() or "JOIN" in query.upper():
        analysis.append("💡 Consider adding indexes on columns used in WHERE/JOIN conditions")
    
    # Check for LIKE with leading wildcard
    if "LIKE '%" in query.upper():
        analysis.append("⚠️ LIKE with leading wildcard ('%...') prevents index usage")
    
    # Check for proper JOIN conditions
    if "JOIN" in query.upper() and "ON" not in query.upper():
        analysis.append("❌ JOIN without ON clause detected - this might cause cartesian product")
    
    # Try to get AI-powered suggestions if OpenAI API key is configured
    openai_key = st.session_state.get('openai_api_key')
    if openai_key:
        try:
            openai.api_key = openai_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL optimization expert. Analyze the query and suggest optimizations. Be concise and focus on performance improvements."},
                    {"role": "user", "content": f"Analyze this SQL query for potential optimizations:\n{query}"}
                ],
                max_tokens=150
            )
            ai_suggestions = response.choices[0].message.content
            analysis.append("\n🤖 AI Suggestions:")
            analysis.append(ai_suggestions)
        except Exception as e:
            analysis.append(f"\n⚠️ AI analysis unavailable: {str(e)}")
    
    return "\n".join(analysis) if analysis else "✅ No immediate optimization suggestions." 