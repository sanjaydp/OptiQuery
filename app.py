import streamlit as st
from dotenv import load_dotenv
from optimizer.sql_parser import analyze_sql
from optimizer.llm_optimizer import optimize_query, explain_optimization
from optimizer.report_generator import generate_report
from optimizer.diff_viewer import generate_diff
from optimizer.query_executor import execute_query
from optimizer.line_commenter import add_inline_comments
from optimizer.cost_estimator import estimate_query_cost
from optimizer.auto_fixer import apply_auto_fixes
from optimizer.complexity_analyzer import calculate_query_complexity  # NEW
import os
import plotly.express as px
import pandas as pd
import openai
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="OptiQuery – AI SQL Assistant", page_icon="🧠", layout="wide")
st.markdown("<h1 style='color:#4B8BBE;'>🧠 OptiQuery: SQL Optimizer Assistant</h1>", unsafe_allow_html=True)
db_path = None

if "optimized_sql" not in st.session_state:
    st.session_state.optimized_sql = ""
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "issues" not in st.session_state:
    st.session_state.issues = []
if "complexity_score" not in st.session_state:
    st.session_state.complexity_score = 0
if "complexity_label" not in st.session_state:
    st.session_state.complexity_label = ""

def nl_to_sql(natural_language: str):
    """Converts natural language to SQL using GPT"""
    chat_prompt = f"Convert this natural language query into a SQL query:\n\n{natural_language}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": chat_prompt}]
    )
    
    return response.choices[0].message.content.strip()

# Add a section in Streamlit for Natural Language Input
st.markdown("### 🧠 Convert Natural Language to SQL Query")

nl_query = st.text_area("Enter your query in natural language here:", height=150)

if st.button("🔍 Convert to SQL"):
    if nl_query.strip():
        with st.spinner("Generating SQL..."):
            sql_query = nl_to_sql(nl_query)
        st.subheader("✅ Generated SQL Query:")
        st.code(sql_query, language="sql")
    else:
        st.warning("Please enter a valid natural language query.")


st.markdown("### 📋 Upload a `.sql` file or paste your SQL query below")
uploaded_file = st.file_uploader("Upload SQL File", type=["sql"])
query = ""

if uploaded_file:
    query = uploaded_file.read().decode("utf-8")
else:
    query = st.text_area("Paste your SQL query here", height=200)

fix_mode = st.radio("Choose optimization mode:", ["🪄 Auto-Fix First", "🤖 GPT Only"])

if st.button("🔍 Analyze & Optimize"):
    if not query.strip():
        st.warning("Please upload or paste a SQL query.")
    else:
        st.session_state.original_query = query

        if fix_mode == "🪄 Auto-Fix First":
            with st.spinner("Applying rule-based fixes..."):
                auto_fixed_sql, fixes_applied = apply_auto_fixes(query, db_path)

            st.subheader("✅ Auto-Fixed SQL (Before GPT)")
            st.code(auto_fixed_sql, language="sql")
            if fixes_applied:
                st.markdown("**Applied Fixes:**")
                for fix in fixes_applied:
                    st.markdown(f"- {fix}")
            else:
                st.info("No auto-fixes were applicable.")

            if st.button("🤖 Continue to GPT Optimization"):
                with st.spinner("Optimizing with GPT..."):
                    issues = analyze_sql(auto_fixed_sql)
                    optimized_sql = optimize_query(auto_fixed_sql)
                    explanation = explain_optimization(auto_fixed_sql, optimized_sql)

                st.session_state.optimized_sql = optimized_sql
                st.session_state.explanation = explanation
                st.session_state.issues = issues

        else:
            with st.spinner("Optimizing with GPT..."):
                issues = analyze_sql(query)
                optimized_sql = optimize_query(query)
                explanation = explain_optimization(query, optimized_sql)

            st.session_state.optimized_sql = optimized_sql
            st.session_state.explanation = explanation
            st.session_state.issues = issues

        score, label = calculate_query_complexity(st.session_state.optimized_sql)
        st.session_state.complexity_score = score
        st.session_state.complexity_label = label

if st.session_state.optimized_sql.strip():
    st.subheader("📌 Identified Issues")
    for issue in st.session_state.issues:
        st.markdown(f"- {issue}")

    st.subheader("✅ Optimized Query")
    st.code(st.session_state.optimized_sql, language='sql')

    st.subheader("📊 Query Complexity Score")
    st.markdown(f"**Score**: {st.session_state.complexity_score} – **{st.session_state.complexity_label}**")

    if st.checkbox("🧠 Show Inline AI Review Comments"):
        with st.spinner("Reviewing query line by line..."):
            reviewed = add_inline_comments(st.session_state.optimized_sql)
        st.subheader("🧾 Inline Comments on Optimized Query")
        st.code(reviewed, language="sql")

    st.subheader("💬 Optimization Explanation")
    st.write(st.session_state.explanation)

    report_md = generate_report(
        st.session_state.original_query,
        st.session_state.issues,
        st.session_state.optimized_sql,
        st.session_state.explanation
    )

    st.download_button(
        label="📥 Download Report as .md",
        data=report_md,
        file_name="optiquery_report.md",
        mime="text/markdown"
    )

    diff_text = generate_diff(st.session_state.original_query, st.session_state.optimized_sql)
    st.subheader("🔀 Before vs After Diff")
    st.code(diff_text, language='diff')

# 💬 Chat Assistant Panel
with st.sidebar:
    st.markdown("## 💬 Ask OptiQuery Assistant")
    user_question = st.chat_input("Ask about the query...")
    if user_question and st.session_state.optimized_sql.strip():
        with st.spinner("Getting AI response..."):
            chat_prompt = f"You are a SQL expert. Based on the following query, answer this: {user_question}\n\nSQL Query:\n{st.session_state.optimized_sql}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": chat_prompt}]
            )

            st.markdown(f"**Answer:** {response.choices[0].message.content.strip()}")

st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/yourusername/optiquery)")
