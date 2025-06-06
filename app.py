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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

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

def explain_sql_with_plan(query):
    """Run EXPLAIN command on the query to get the execution plan."""
    explain_query = f"EXPLAIN {query}"
    explain_output = execute_query(db_path, explain_query)  # You would need to execute the query in the DB

    if explain_output["success"]:
        return explain_output["rows"]  # Return the execution plan rows
    else:
        return f"Error: {explain_output['error']}"


def parse_explain_plan(explain_output):
    """Parse the EXPLAIN plan output to DataFrame"""
    data = []
    for line in explain_output:
        # Assuming tab-delimited output; adjust this if the delimiter is different
        data.append(line.split("\t"))

    df = pd.DataFrame(data, columns=['ID', 'Select Type', 'Table', 'Type', 'Possible Keys', 'Key', 'Key Length', 'Ref', 'Rows', 'Extra'])
    return df

def suggest_based_on_explain_plan(explain_plan_df):
    suggestions = []
    
    # Check for full table scans (type 'ALL') in the EXPLAIN plan
    if "ALL" in explain_plan_df['Type'].values:
        suggestions.append("Consider adding an index on frequently queried columns to avoid full table scans.")
    
    # Check for inefficient join types (e.g., nested loop joins)
    if "NESTED LOOP" in explain_plan_df['Extra'].values:
        suggestions.append("Consider changing the join type or adding an index to improve performance.")
    
    # Check if there are too many rows being scanned, indicating possible performance issues
    high_row_count = explain_plan_df['Rows'].astype(int).max()
    if high_row_count > 10000:  # This is a threshold, can be adjusted
        suggestions.append("The query might scan too many rows. Consider optimizing the WHERE clause or adding filters.")
    
    # If there are 'Using temporary' or 'Using filesort' in 'Extra', suggest index optimization
    if "Using temporary" in explain_plan_df['Extra'].values or "Using filesort" in explain_plan_df['Extra'].values:
        suggestions.append("Consider optimizing the query to avoid using temporary tables or filesort. Indexing could help.")
    
    return suggestions

def convert_markdown_to_pdf(md_content: str) -> bytes:
    """Converts markdown content to a PDF byte object using ReportLab."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter  # Keep the standard letter size
    
    # Write the markdown content line by line
    y_position = height - 40  # Start at top of page
    lines = md_content.splitlines()
    for line in lines:
        c.drawString(40, y_position, line)
        y_position -= 12  # Move down for next line
        if y_position < 40:  # If we reach the bottom of the page
            c.showPage()  # Start a new page
            y_position = height - 40

    c.save()
    buffer.seek(0)
    return buffer.read()

def nl_to_sql(natural_language: str):
    """Converts natural language to SQL using GPT"""
    chat_prompt = f"Convert this natural language query into a SQL query:\n\n{natural_language}"

    response = client.chat.completions.create(
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

    # Add PDF download button
    st.download_button(
            label="📄 Download PDF Report",
            data=convert_markdown_to_pdf(report_md),
            file_name="optiquery_report.pdf",
            mime="application/pdf"
        )

    diff_text = generate_diff(st.session_state.original_query, st.session_state.optimized_sql)
    st.subheader("🔀 Before vs After Diff")
    st.code(diff_text, language='diff')

    explain_plan = explain_sql_with_plan(st.session_state.optimized_sql)
    explain_plan_df = parse_explain_plan(explain_plan)
    st.dataframe(explain_plan_df)  # Show EXPLAIN plan

    suggestions = suggest_based_on_explain_plan(explain_plan_df)
    if suggestions:
        st.subheader("🔧 Suggested Optimizations")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    # Show Performance Comparison
    if st.button("📊 Compare Performance"):
        result_orig = execute_query(db_path, st.session_state.original_query)
        result_opt = execute_query(db_path, st.session_state.optimized_sql)

        if result_orig["success"] and result_opt["success"]:
            perf_data = {
                "Query": ["Original", "Optimized"],
                "Execution Time (s)": [result_orig["execution_time"], result_opt["execution_time"]],
                "Row Count": [result_orig["row_count"], result_opt["row_count"]],
            }
            df_perf = pd.DataFrame(perf_data)
            fig_time = px.bar(df_perf, x="Query", y="Execution Time (s)", color="Query", title="⏱️ Execution Time Comparison")
            st.plotly_chart(fig_time)
            fig_rows = px.bar(df_perf, x="Query", y="Row Count", color="Query", title="📦 Row Count Comparison")
            st.plotly_chart(fig_rows)

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
