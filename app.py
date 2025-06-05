import streamlit as st
from dotenv import load_dotenv
from optimizer.sql_parser import analyze_sql
from optimizer.llm_optimizer import optimize_query, explain_optimization
from optimizer.report_generator import generate_report
from optimizer.diff_viewer import generate_diff
from optimizer.query_executor import execute_query
from optimizer.line_commenter import add_inline_comments
from optimizer.cost_estimator import estimate_query_cost
from optimizer.auto_fixer import apply_auto_fixes  # You must have this module
import os

# Load env
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="OptiQuery – AI SQL Assistant", page_icon="🧠", layout="wide")
st.markdown("<h1 style='color:#4B8BBE;'>🧠 OptiQuery: SQL Optimizer Assistant</h1>", unsafe_allow_html=True)
db_path = None

# Session states
if "optimized_sql" not in st.session_state:
    st.session_state.optimized_sql = ""
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "issues" not in st.session_state:
    st.session_state.issues = []

# Input area
st.markdown("### 📋 Upload a `.sql` file or paste your SQL query below")
uploaded_file = st.file_uploader("Upload SQL File", type=["sql"])
query = ""

if uploaded_file:
    query = uploaded_file.read().decode("utf-8")
else:
    query = st.text_area("Paste your SQL query here", height=200)

# Optimization mode choice
fix_mode = st.radio("Choose optimization mode:", ["🪄 Auto-Fix First", "🤖 GPT Only"])

# Analyze & Optimize Button
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

                # Save to session
                st.session_state.optimized_sql = optimized_sql
                st.session_state.explanation = explanation
                st.session_state.issues = issues

        else:  # GPT Only
            with st.spinner("Optimizing with GPT..."):
                issues = analyze_sql(query)
                optimized_sql = optimize_query(query)
                explanation = explain_optimization(query, optimized_sql)

            st.session_state.optimized_sql = optimized_sql
            st.session_state.explanation = explanation
            st.session_state.issues = issues

# Results
if st.session_state.optimized_sql.strip():
    st.subheader("📌 Identified Issues")
    for issue in st.session_state.issues:
        st.markdown(f"- {issue}")

    st.subheader("✅ Optimized Query")
    st.code(st.session_state.optimized_sql, language='sql')

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

    # Optional DB Tester
    st.markdown("### 🧪 Optional: Test Queries on a SQLite DB")
    db_file = st.file_uploader("Upload a SQLite `.db` file to test queries", type=["db"])

    if db_file:
        db_path = "temp_db.db"
        with open(db_path, "wb") as f:
            f.write(db_file.getbuffer())

        st.markdown("#### ▶️ Executing Original Query...")
        result_orig = execute_query(db_path, st.session_state.original_query)
        if result_orig["success"]:
            st.success(f"✅ Returned {result_orig['row_count']} rows in {result_orig['execution_time']} sec.")
            st.write(result_orig["rows"])
        else:
            st.error(f"❌ {result_orig['error']}")

        st.markdown("#### ▶️ Executing Optimized Query...")
        result_opt = execute_query(db_path, st.session_state.optimized_sql)
        if result_opt["success"]:
            st.success(f"✅ Returned {result_opt['row_count']} rows in {result_opt['execution_time']} sec.")
            st.write(result_opt["rows"])
        else:
            st.error(f"❌ {result_opt['error']}")

# Footer
st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/yourusername/optiquery)")
