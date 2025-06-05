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
import os

# Load .env for local dev
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="OptiQuery – AI SQL Assistant", page_icon="🧠", layout="wide")
st.markdown("<h1 style='color:#4B8BBE;'>🧠 OptiQuery: SQL Optimizer Assistant</h1>", unsafe_allow_html=True)

# 🔄 SESSION STATE INIT
if "optimized_sql" not in st.session_state:
    st.session_state.optimized_sql = ""
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "issues" not in st.session_state:
    st.session_state.issues = []

# 📥 SQL INPUT
st.markdown("### 📋 Upload a `.sql` file or paste your SQL query below")
uploaded_file = st.file_uploader("Upload SQL File", type=["sql"])
query = ""

if uploaded_file:
    query = uploaded_file.read().decode("utf-8")
else:
    query = st.text_area("Paste your SQL query here", height=200)

# 🔍 ANALYZE & OPTIMIZE
if st.button("🔍 Analyze & Optimize"):
    if not query.strip():
        st.warning("Please upload or paste a SQL query.")
    else:
        # 🪄 Auto-fix toggle
        apply_fixes = st.checkbox("🪄 Auto-fix common SQL issues (SELECT *, no LIMIT, etc.)")

        if apply_fixes:
            query, fixes = apply_auto_fixes(query)
            st.info("✅ Auto-fixes applied:\n" + "\n".join(f"- {fix}" for fix in fixes))

        with st.spinner("Analyzing query..."):
            issues = analyze_sql(query)
            optimized_sql = optimize_query(query)
            explanation = explain_optimization(query, optimized_sql)

        # Save to session
        st.session_state.optimized_sql = optimized_sql
        st.session_state.original_query = query
        st.session_state.explanation = explanation
        st.session_state.issues = issues


# ✅ SHOW OPTIMIZED OUTPUT
if st.session_state.optimized_sql.strip():
    st.subheader("📌 Identified Issues")
    for issue in st.session_state.issues:
        st.markdown(f"- {issue}")

    st.subheader("✅ Optimized Query")
    st.code(st.session_state.optimized_sql, language='sql')

    # 🧠 INLINE COMMENT REVIEW
    if st.checkbox("🧠 Show Inline AI Review Comments"):
        with st.spinner("Reviewing query line by line..."):
            reviewed = add_inline_comments(st.session_state.optimized_sql)
        st.subheader("🧾 Inline Comments on Optimized Query")
        st.code(reviewed, language="sql")

    st.subheader("💬 Optimization Explanation")
    st.write(st.session_state.explanation)

    # 📊 COST ESTIMATION
    cost_result = estimate_query_cost(st.session_state.optimized_sql)
    st.subheader("📊 Query Cost Estimation")
    st.markdown(f"**Score:** `{cost_result['score']} / 100`")
    st.markdown(f"**Risk Level:** `{cost_result['risk']}`")

    if cost_result["issues"]:
        st.markdown("**⚠️ Potential Issues:**")
        for issue in cost_result["issues"]:
            st.markdown(f"- {issue}")
    else:
        st.markdown("✅ No major issues detected.")

    # 📥 DOWNLOAD REPORT
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

    # 🔀 DIFF VIEWER
    diff_text = generate_diff(st.session_state.original_query, st.session_state.optimized_sql)
    st.subheader("🔀 Before vs After Diff")
    st.code(diff_text, language='diff')

    # 🧪 SQLITE QUERY EXECUTION
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

# 📦 GITHUB LINK
st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/yourusername/optiquery)")
