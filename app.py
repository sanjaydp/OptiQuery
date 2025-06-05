import streamlit as st
from dotenv import load_dotenv
from optimizer.sql_parser import analyze_sql
from optimizer.llm_optimizer import optimize_query, explain_optimization
from optimizer.report_generator import generate_report
from optimizer.diff_viewer import generate_diff
from optimizer.query_executor import execute_query
from optimizer.line_commenter import add_inline_comments
import os

# Load environment variables
load_dotenv()
st.set_page_config(page_title="OptiQuery – AI SQL Assistant", page_icon="🧠", layout="wide")

st.markdown("<h1 style='color:#4B8BBE;'>🧠 OptiQuery: SQL Optimizer Assistant</h1>", unsafe_allow_html=True)
st.markdown("### 📋 Upload a `.sql` file or paste your SQL query below")

# SQL Query input
uploaded_file = st.file_uploader("Upload SQL File", type=["sql"])
query = ""

if uploaded_file:
    query = uploaded_file.read().decode("utf-8")
else:
    query = st.text_area("Paste your SQL query here", height=200)

optimized_sql = ""

if st.button("🔍 Analyze & Optimize"):
    if not query.strip():
        st.warning("Please upload or paste a SQL query.")
    else:
        with st.spinner("Analyzing query..."):
            issues = analyze_sql(query)
            optimized_sql = optimize_query(query)
            explanation = explain_optimization(query, optimized_sql)

        st.subheader("📌 Identified Issues")
        for issue in issues:
            st.markdown(f"- {issue}")

        st.subheader("✅ Optimized Query")
        st.code(optimized_sql, language='sql')

        # Inline AI reviewer
        if optimized_sql.strip():
            if st.checkbox("🧠 Show Inline AI Review Comments"):
                with st.spinner("Reviewing query line by line..."):
                    reviewed = add_inline_comments(optimized_sql)
                st.subheader("🧾 Inline Comments on Optimized Query")
                st.code(reviewed, language="sql")

        st.subheader("💬 Optimization Explanation")
        st.write(explanation)

        # Download report
        report_md = generate_report(query, issues, optimized_sql, explanation)
        st.download_button(
            label="📥 Download Report as .md",
            data=report_md,
            file_name="optiquery_report.md",
            mime="text/markdown"
        )

        # Before/after diff
        diff_text = generate_diff(query, optimized_sql)
        st.subheader("🔀 Before vs After Diff")
        st.code(diff_text, language='diff')

        # SQLite Test Section
        st.markdown("### 🧪 Optional: Test Queries on a SQLite DB")
        db_file = st.file_uploader("Upload a SQLite `.db` file to test queries", type=["db"])

        if db_file:
            db_path = "temp_db.db"
            with open(db_path, "wb") as f:
                f.write(db_file.getbuffer())

            st.markdown("#### ▶️ Executing Original Query...")
            result_orig = execute_query(db_path, query)
            if result_orig["success"]:
                st.success(f"✅ Returned {result_orig['row_count']} rows in {result_orig['execution_time']} sec.")
                st.write(result_orig["rows"])
            else:
                st.error(f"❌ {result_orig['error']}")

            st.markdown("#### ▶️ Executing Optimized Query...")
            result_opt = execute_query(db_path, optimized_sql)
            if result_opt["success"]:
                st.success(f"✅ Returned {result_opt['row_count']} rows in {result_opt['execution_time']} sec.")
                st.write(result_opt["rows"])
            else:
                st.error(f"❌ {result_opt['error']}")

# GitHub link
st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/yourusername/optiquery)")
