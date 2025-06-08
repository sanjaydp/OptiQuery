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
from optimizer.complexity_analyzer import calculate_query_complexity
import os
import plotly.express as px
import pandas as pd
import sqlite3
from openai import OpenAI
import re

# Initialize session state
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

def nl_to_sql(natural_language: str):
    """Converts natural language to SQL using GPT"""
    chat_prompt = f"Convert this natural language query into a SQL query:\n\n{natural_language}"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": chat_prompt}]
    )
    return response.choices[0].message.content.strip()

def run_analysis(query, analysis_options, db_path):
    """
    Run the selected analysis options on the given query.
    
    Args:
        query (str): The SQL query to analyze
        analysis_options (list): List of selected analysis types
        db_path (str): Path to the SQLite database file
    """
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
                
                # Get existing indexes
                cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'")
                indexes = cursor.fetchall()
                
                table_stats[table_name] = {
                    "row_count": row_count,
                    "columns": [col[1] for col in columns],
                    "indexes": {idx[0]: idx[1] for idx in indexes}
                }
            conn.close()
        except Exception as e:
            st.error(f"Error collecting table statistics: {str(e)}")
            return

        # 1. Syntax Check
        if "Syntax Check" in analysis_options:
            progress_bar.progress(20)
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
            
            progress_bar.progress(40)
            
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

        # 2. Performance Analysis & Optimization
        if "Performance Analysis" in analysis_options:
            st.markdown("#### 🚀 Query Optimization")
            
            # Get optimization suggestions
            optimization_result = optimize_query(
                query,
                schema_info=st.session_state.get("schema_summary", ""),
                table_stats=table_stats
            )
            
            progress_bar.progress(80)
            
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

        # 3. Index Recommendations
        if "Index Recommendations" in analysis_options:
            st.markdown("#### 📊 Index Analysis")
            
            try:
                # Get existing indexes
                existing_indexes = {}
                for table, stats in table_stats.items():
                    if "indexes" in stats:
                        existing_indexes[table] = stats["indexes"]
                
                st.markdown("**Current Indexes:**")
                if existing_indexes:
                    for table, indexes in existing_indexes.items():
                        st.markdown(f"*Table: {table}*")
                        for idx_name, idx_sql in indexes.items():
                            st.code(idx_sql, language="sql")
                else:
                    st.info("No existing indexes found")
                
                # Analyze query for potential indexes
                st.markdown("**Recommended Indexes:**")
                with st.spinner("Analyzing for index recommendations..."):
                    # Extract tables and columns from the query
                    tables_in_query = re.findall(r"FROM\s+(\w+)|JOIN\s+(\w+)", query, re.IGNORECASE)
                    tables_in_query = [t[0] or t[1] for t in tables_in_query if t[0] or t[1]]
                    
                    # Extract WHERE conditions
                    where_conditions = re.findall(r"WHERE\s+(\w+\.\w+|\w+)\s*[=<>]", query, re.IGNORECASE)
                    
                    # Extract JOIN conditions
                    join_conditions = re.findall(r"ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", query, re.IGNORECASE)
                    
                    recommended_indexes = []
                    
                    # Recommend indexes for WHERE conditions
                    for cond in where_conditions:
                        if "." in cond:
                            table, column = cond.split(".")
                            if table in tables_in_query:
                                recommended_indexes.append(f"CREATE INDEX idx_{table}_{column} ON {table}({column});")
                    
                    # Recommend indexes for JOIN conditions
                    for left, right in join_conditions:
                        left_table, left_col = left.split(".")
                        right_table, right_col = right.split(".")
                        if left_table in tables_in_query:
                            recommended_indexes.append(f"CREATE INDEX idx_{left_table}_{left_col} ON {left_table}({left_col});")
                        if right_table in tables_in_query:
                            recommended_indexes.append(f"CREATE INDEX idx_{right_table}_{right_col} ON {right_table}({right_col});")
                    
                    if recommended_indexes:
                        for idx in recommended_indexes:
                            st.code(idx, language="sql")
                    else:
                        st.info("No additional indexes recommended")
            except Exception as e:
                st.error(f"Error analyzing indexes: {str(e)}")

        # 4. Query Plan
        if "Query Plan" in analysis_options:
            st.markdown("#### 📈 Query Execution Plan")
            
            try:
                with st.spinner("Analyzing query execution plan..."):
                    # Execute EXPLAIN QUERY PLAN
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                    plan_rows = cursor.fetchall()
                    conn.close()
                    
                    if plan_rows:
                        # Create a DataFrame for the execution plan
                        plan_df = pd.DataFrame(plan_rows, columns=['id', 'parent', 'notused', 'detail'])
                        st.dataframe(plan_df, hide_index=True)
                        
                        # Add explanation of the query plan
                        st.markdown("**Plan Analysis:**")
                        
                        # Look for full table scans
                        full_scans = [row for row in plan_rows if 'SCAN TABLE' in row[3] and 'SEARCH TABLE' not in row[3]]
                        if full_scans:
                            st.warning("⚠️ Query contains full table scan(s):")
                            for scan in full_scans:
                                st.markdown(f"- {scan[3]}")
                        
                        # Look for index usage
                        index_usage = [row for row in plan_rows if 'SEARCH' in row[3] or 'INDEX' in row[3]]
                        if index_usage:
                            st.success("✅ Query uses indexes:")
                            for idx in index_usage:
                                st.markdown(f"- {idx[3]}")
                    else:
                        st.warning("No execution plan available for this query")
            except Exception as e:
                st.error(f"Error analyzing query plan: {str(e)}")

        progress_bar.progress(100)
        st.success("✅ Analysis Complete!")

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

# Footer
st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/sanjaydp/optiquery)")
