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
import sqlite3

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="OptiQuery – SQL Optimizer Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1fae5;
        border: 1px solid #34d399;
    }
    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #4B8BBE;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:#4B8BBE;'>🧠 OptiQuery: SQL Optimizer Assistant</h1>", unsafe_allow_html=True)

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


def extract_schema_summary(db_path):
    """Create a string summary of all tables and columns for LLM context."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_str = ""
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            col_names = ", ".join([col[1] for col in columns])
            schema_str += f"Table `{table_name}`: columns -> {col_names}\n"
        conn.close()
        return schema_str.strip()
    except Exception as e:
        return f"Error extracting schema: {e}"


def explain_sql_with_plan(query):
    """Run EXPLAIN command on the query to get the execution plan."""
    explain_query = f"EXPLAIN {query}"
    explain_output = execute_query(db_path, explain_query)  # You would need to execute the query in the DB

    if explain_output["success"]:
        return explain_output["rows"]  # Return the execution plan rows
    else:
        return f"Error: {explain_output['error']}"


def parse_explain_plan(explain_output):
    """Parse the EXPLAIN plan output to DataFrame."""
    data = []
    
    for line in explain_output:
        print(f"Parsing line: {line}")  # Debugging: check the line structure
        # Adjust delimiter if needed, this assumes tab-delimited format
        split_line = line.split("\t")
        if len(split_line) == 10:  # Assuming you expect 10 columns
            data.append(split_line)
        else:
            print(f"Skipping line due to unexpected column count: {line}")
    
    # Check if data has valid rows before creating DataFrame
    if data:
        try:
            df = pd.DataFrame(data, columns=['ID', 'Select Type', 'Table', 'Type', 'Possible Keys', 'Key', 'Key Length', 'Ref', 'Rows', 'Extra'])
            return df
        except ValueError as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame()  # Return empty DataFrame if columns don't match
    else:
        print("No valid data found in EXPLAIN output.")
        return pd.DataFrame()  # Return empty DataFrame if no valid data

def suggest_based_on_explain_plan(explain_plan_df):
    """Suggest optimizations based on the EXPLAIN plan."""
    if explain_plan_df.empty:
        print("No valid EXPLAIN data to suggest optimizations.")
        return []

    suggestions = []
    
    # Ensure the 'Type' column exists before proceeding
    if 'Type' not in explain_plan_df.columns:
        print("'Type' column missing in EXPLAIN plan DataFrame.")
        return []

    # Check for full table scans (type 'ALL')
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

# Use this revised parse_explain_plan and suggest_based_on_explain_plan functions in your main app


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

# Natural Language to SQL Section
st.markdown("### 🗣️ Convert Natural Language to SQL Query")
nl_query = st.text_area(
    "Enter your query in natural language here:",
    height=100,
    help="Describe what you want to query in plain English"
)

if st.button("🔍 Convert to SQL", key="convert_nl"):
    if nl_query.strip():
        with st.spinner("Generating SQL..."):
            sql_query = nl_to_sql(nl_query)
        st.subheader("✅ Generated SQL Query:")
        st.code(sql_query, language="sql")
        
        # Add button to use this query
        if st.button("📝 Use this Query"):
            st.session_state.current_query = sql_query
    else:
        st.warning("Please enter a natural language query.")

# Database Connection Section
st.markdown("### 📊 Database Connection")
uploaded_file = st.file_uploader(
    "Upload your SQLite Database File",
    type=["db"],
    help="Upload a SQLite database file to analyze"
)

if uploaded_file:
    try:
        # Save uploaded database
        db_path = f"temp_{uploaded_file.name}"
        with open(db_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Validate database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            if not version:
                raise Exception("Invalid SQLite database file")
            conn.close()
            
            # Store database path in session state
            st.session_state.db_path = db_path
            
            # Show success message
            st.markdown(
                '<div class="success-message">✅ Database uploaded successfully!</div>',
                unsafe_allow_html=True
            )
            
            # Extract and display schema
            schema_info = extract_schema_summary(db_path)
            if schema_info:
                st.session_state.schema_summary = schema_info
                
                # Schema Section
                st.markdown("#### 📑 Database Schema")
                schema_expander = st.expander("View Schema", expanded=True)
                with schema_expander:
                    for line in schema_info.split('\n'):
                        st.markdown(f"- `{line}`")
                
                # Sample Data Preview Section
                st.markdown("#### 📊 Sample Data Preview")
                preview_expander = st.expander("View Sample Data")
                with preview_expander:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    if tables:
                        selected_table = st.selectbox(
                            "Select a table to preview:",
                            [table[0] for table in tables]
                        )
                        
                        if selected_table:
                            df = pd.read_sql_query(
                                f"SELECT * FROM {selected_table} LIMIT 5",
                                conn
                            )
                            st.dataframe(df)
                            
                            # Show table statistics
                            cursor.execute(f"SELECT COUNT(*) FROM {selected_table}")
                            row_count = cursor.fetchone()[0]
                            st.markdown(f"**Total rows:** {row_count:,}")
                    else:
                        st.warning("No tables found in database")
                    conn.close()
                
                # SQL Query Input Section
                st.markdown("### 📝 SQL Query Input")
                query_input_method = st.radio(
                    "Choose input method:",
                    ["Upload SQL File", "Paste SQL Query"]
                )
                
                query = st.session_state.get('current_query', '')
                
                if query_input_method == "Upload SQL File":
                    sql_file = st.file_uploader("Upload SQL File", type=["sql"])
                    if sql_file:
                        query = sql_file.read().decode("utf-8")
                else:
                    query = st.text_area(
                        "Enter your SQL query:",
                        value=query,
                        height=150,
                        help="Write your SQL query here. The schema information is shown above for reference."
                    )

                if query.strip():
                    # Analysis Options
                    st.markdown("### 🔍 Analysis Options")
                    analysis_options = st.multiselect(
                        "Choose analysis types:",
                        ["Syntax Check", "Performance Analysis", "Index Recommendations", "Query Plan"],
                        default=["Syntax Check", "Performance Analysis"]
                    )

                    # Add analyze button
                    if st.button("🚀 Analyze & Optimize Query", type="primary", use_container_width=True):
                        run_analysis(query, analysis_options, db_path)

        except Exception as e:
            st.error(f"❌ Invalid database file: {str(e)}")
            if os.path.exists(db_path):
                os.remove(db_path)
            
    except Exception as e:
        st.error(f"❌ Error processing database file: {str(e)}")

# Move the analysis function to a separate function for better organization
def run_analysis(query, analysis_options, db_path):
    with st.spinner("🔍 Analyzing your query..."):
        progress_bar = st.progress(0)
        
        # Collect table statistics
        table_stats = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column statistics
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_stats[table_name] = {
                    "row_count": row_count,
                    "columns": [col[1] for col in columns]
                }
            conn.close()
        except Exception as e:
            st.error(f"Error collecting table statistics: {str(e)}")

        # Show progress
        progress_bar.progress(20)
        
        # 1. Syntax Check
        if "Syntax Check" in analysis_options:
            progress_bar.progress(40)
            st.markdown("#### 📝 Query Analysis")
            analysis_result = analyze_sql(query)
            
            # Display issues
            issues = analysis_result.get("issues", [])
            suggestions = analysis_result.get("suggestions", [])
            complexity = analysis_result.get("complexity", {})
            
            if issues:
                st.markdown("**⚠️ Potential Issues:**")
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ No major issues found")
            
            # Display suggestions
            if suggestions:
                st.markdown("**💡 Optimization Suggestions:**")
                for suggestion in suggestions:
                    st.info(suggestion)
            
            # Display complexity analysis
            if complexity:
                st.markdown("**🔍 Query Complexity Analysis:**")
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Complexity Score",
                        f"{complexity.get('score', 0)}/100",
                        delta=None,
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Complexity Level",
                        complexity.get('level', 'N/A'),
                        delta=None
                    )
                
                # Display complexity factors
                if 'factors' in complexity:
                    st.markdown("**Complexity Factors:**")
                    factors = complexity['factors']
                    factor_df = pd.DataFrame({
                        'Factor': ['Joins', 'Where Conditions', 'Subqueries', 'Function Calls'],
                        'Count': [
                            factors.get('joins', 0),
                            factors.get('conditions', 0),
                            factors.get('subqueries', 0),
                            factors.get('functions', 0)
                        ]
                    })
                    st.dataframe(factor_df, hide_index=True)

            progress_bar.progress(60)
            
            # Display complexity analysis
            complexity = analysis_result["complexity"]
            st.markdown("**🔍 Query Complexity Analysis:**")
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Complexity Score",
                    f"{complexity['score']}/100",
                    delta=None,
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Complexity Level",
                    complexity['level'],
                    delta=None
                )
            
            # Display complexity factors
            st.markdown("**Complexity Factors:**")
            factors = complexity['factors']
            factor_df = pd.DataFrame({
                'Factor': ['Joins', 'Where Conditions', 'Subqueries', 'Function Calls'],
                'Count': [
                    factors['joins'],
                    factors['conditions'],
                    factors['subqueries'],
                    factors['functions']
                ]
            })
            st.dataframe(factor_df, hide_index=True)
            
            progress_bar.progress(80)

        # 2. Performance Analysis & Optimization
        if "Performance Analysis" in analysis_options:
            st.markdown("#### 🚀 Query Optimization")
            
            # Get optimization suggestions
            optimization_result = optimize_query(
                query,
                schema_info=st.session_state.schema_summary,
                table_stats=table_stats
            )
            
            progress_bar.progress(100)
            
            # Store optimized query in session state
            st.session_state.optimized_sql = optimization_result["optimized_query"]
            
            # Display optimized query
            st.markdown("**Optimized Query:**")
            st.code(optimization_result["optimized_query"], language="sql")
            
            # Show optimization reasoning
            st.markdown("**Optimization Details:**")
            st.info(optimization_result["optimization_reasoning"])
            
            # Show estimated improvement
            st.metric(
                label="Estimated Performance Improvement",
                value=optimization_result["estimated_improvement"]
            )

        progress_bar.progress(100)
        st.success("✅ Analysis Complete!")

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
st.markdown("📦 [View Source Code on GitHub](https://github.com/sanjaydp/optiquery)")
