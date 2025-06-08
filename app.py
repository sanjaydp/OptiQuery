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
from optimizer.enterprise_features import (
    analyze_enterprise_features,
    display_enterprise_analysis,
    add_enterprise_sidebar
)
from optimizer.advanced_analysis import QueryAnalyzer
from config import (
    ANALYSIS_SETTINGS,
    SECURITY_SETTINGS,
    PERFORMANCE_SETTINGS,
    DATA_QUALITY_SETTINGS,
    BEST_PRACTICES,
    QUERY_CATEGORIES,
    OPTIMIZATION_STRATEGIES
)
import os
import plotly.express as px
import pandas as pd
import sqlite3
from openai import OpenAI
import re
import time
from datetime import datetime
import json
from typing import Dict

# Initialize session state
if "debug" not in st.session_state:
    st.session_state.debug = True  # Enable debugging

def init_session_state():
    """Initialize all session state variables in one place"""
    if "initialized" not in st.session_state:
        state_vars = {
            "optimized_sql": "",
            "original_query": "",
            "explanation": "",
            "issues": [],
            "complexity_score": 0,
            "complexity_label": "",
            "query_history": [],
            "benchmark_results": {},
            "saved_queries": {},
            "current_schema_version": None,
            "query_category": "operational",
            "optimization_strategy": "balanced",
            "advanced_settings": {},
            "chat_history": [],
            "analysis_results": {
                "syntax": {},
                "performance": {},
                "index_recommendations": {},
                "query_plan": {}
            },
            "query_results": {
                "original": {
                    "data": None,
                    "execution_time": None,
                    "row_count": None,
                    "error": None
                },
                "optimized": {
                    "data": None,
                    "execution_time": None,
                    "row_count": None,
                    "error": None
                }
            },
            "last_analysis_id": None,  # Track the last analysis
            "initialized": True
        }
        
        for key, value in state_vars.items():
            if key not in st.session_state:
                st.session_state[key] = value

def debug_state(location: str):
    """Print debug information about the session state"""
    if st.session_state.debug:
        st.sidebar.markdown(f"### 🔍 Debug Info: {location}")
        st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
        st.sidebar.write("Has Query Results:", bool(st.session_state.query_results["original"]["data"] or st.session_state.query_results["optimized"]["data"]))
        st.sidebar.write("Analysis Results Keys:", list(st.session_state.analysis_results.keys()))
        st.sidebar.write("Last Analysis ID:", st.session_state.last_analysis_id)

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

def analyze_security_risks(query):
    """Analyze query for security risks and best practices."""
    risks = []
    
    # Check for SQL injection vulnerabilities
    if "EXECUTE" in query.upper() or "EXEC" in query.upper():
        risks.append("⚠️ Dynamic SQL execution detected - potential SQL injection risk")
    
    # Check for sensitive data exposure
    sensitive_patterns = ["password", "credit_card", "ssn", "secret", "token"]
    for pattern in sensitive_patterns:
        if pattern in query.lower():
            risks.append(f"⚠️ Possible sensitive data exposure: {pattern}")
    
    # Check for proper schema usage
    if "information_schema" in query.lower():
        risks.append("⚠️ Direct information_schema access - consider using proper access controls")
    
    return risks

def analyze_data_quality(query, db_path):
    """Analyze potential data quality impacts."""
    impacts = []
    
    # Check for NULL handling
    if "IS NULL" not in query.upper() and "IS NOT NULL" not in query.upper():
        impacts.append("⚠️ No NULL checks found - consider handling NULL values")
    
    # Check for data truncation risks
    if "CAST" in query.upper() or "CONVERT" in query.upper():
        impacts.append("⚠️ Type conversion detected - verify no data truncation risks")
    
    # Check for proper JOIN conditions
    if "JOIN" in query.upper() and "ON" not in query.upper():
        impacts.append("⚠️ JOIN without ON clause - risk of cartesian product")
    
    return impacts

def benchmark_query(query, db_path):
    """Run query benchmarks with different data volumes."""
    results = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Run query multiple times to get average performance
        iterations = 3
        total_time = 0
        
        for i in range(iterations):
            start_time = time.time()
            cursor.execute(query)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_time = total_time / iterations
        
        # Get result set size
        cursor.execute(query)
        result_count = len(cursor.fetchall())
        
        # Get estimated rows processed
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = cursor.fetchall()
        estimated_rows = sum([int(row[3].split()[-1]) for row in plan if 'SCAN' in row[3]])
        
        results = {
            "average_execution_time": avg_time,
            "result_set_size": result_count,
            "estimated_rows_processed": estimated_rows,
            "iterations": iterations
        }
        
        conn.close()
    except Exception as e:
        results = {"error": str(e)}
    
    return results

def generate_documentation(query, analysis_results):
    """Generate comprehensive query documentation."""
    doc = "# Query Documentation\n\n"
    
    # Basic Information
    doc += "## Overview\n"
    doc += f"```sql\n{query}\n```\n\n"
    
    # Purpose and Usage
    doc += "## Purpose\n"
    doc += "This query is designed to...\n\n"
    
    # Performance Characteristics
    doc += "## Performance Characteristics\n"
    if "complexity" in analysis_results:
        doc += f"- Complexity Score: {analysis_results['complexity'].get('score', 'N/A')}/100\n"
        doc += f"- Complexity Level: {analysis_results['complexity'].get('level', 'N/A')}\n"
    
    # Dependencies
    doc += "\n## Dependencies\n"
    tables = re.findall(r"FROM\s+(\w+)|JOIN\s+(\w+)", query, re.IGNORECASE)
    doc += "### Tables:\n"
    for table in tables:
        doc += f"- {table[0] or table[1]}\n"
    
    # Security Considerations
    doc += "\n## Security Considerations\n"
    security_risks = analyze_security_risks(query)
    for risk in security_risks:
        doc += f"- {risk}\n"
    
    return doc

def store_analysis_results(analysis_type: str, results: dict):
    """Store analysis results in session state"""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    st.session_state.analysis_results[analysis_type] = results

def store_query_results(result_type: str, results: dict):
    """Store query results in session state with debugging"""
    if st.session_state.debug:
        st.sidebar.markdown(f"### 💾 Storing {result_type} Query Results")
        st.sidebar.write("Results:", results)
    
    if results["success"]:
        st.session_state.query_results[result_type] = {
            "data": results["rows"],
            "execution_time": results["execution_time"],
            "row_count": results["row_count"],
            "error": None
        }
    else:
        st.session_state.query_results[result_type] = {
            "data": None,
            "execution_time": None,
            "row_count": None,
            "error": results.get("error", "Unknown error occurred")
        }
    
    # Force Streamlit to recognize the state change
    st.session_state.last_analysis_id = datetime.now().isoformat()

def display_query_results():
    """Display query results and comparison with debugging"""
    debug_state("Before Displaying Results")
    
    if st.session_state.query_results["original"]["data"] is not None:
        st.markdown("#### 📊 Original Query Results")
        st.dataframe(pd.DataFrame(st.session_state.query_results["original"]["data"]))
        st.metric(
            "Original Execution Time",
            f"{st.session_state.query_results['original']['execution_time']}s"
        )

    if st.session_state.query_results["optimized"]["data"] is not None:
        st.markdown("#### 🚀 Optimized Query Results")
        st.dataframe(pd.DataFrame(st.session_state.query_results["optimized"]["data"]))
        
        # Calculate improvement percentage
        orig_time = st.session_state.query_results["original"]["execution_time"]
        opt_time = st.session_state.query_results["optimized"]["execution_time"]
        if orig_time and opt_time:
            improvement = ((orig_time - opt_time) / orig_time * 100)
            st.metric(
                "Optimized Execution Time",
                f"{opt_time}s",
                delta=f"{improvement:.1f}%",
                delta_color="inverse"
            )
    
    debug_state("After Displaying Results")

def run_analysis(query, analysis_options, db_path):
    """Enhanced run_analysis function with enterprise features"""
    debug_state("Starting Analysis")
    
    with st.spinner("🔍 Analyzing your query..."):
        progress_bar = st.progress(0)
        
        # Store original query
        st.session_state.original_query = query
        
        try:
            # Execute original query and store results
            original_results = execute_query(db_path, query)
            store_query_results("original", original_results)
            debug_state("After Original Query")
            
            # Run enterprise analysis first
            enterprise_results = analyze_enterprise_features(query)
            display_enterprise_analysis(query, enterprise_results)
            store_analysis_results("enterprise", enterprise_results)
            progress_bar.progress(30)
            
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
            
            progress_bar.progress(50)
            
            # Run selected analysis options
            if "Syntax Check" in analysis_options:
                st.markdown("#### 📝 Query Analysis")
                analysis_result = analyze_sql(query)
                store_analysis_results("syntax", analysis_result)
                
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
                    st.session_state.complexity_score = complexity.get('score', 0)
                    st.session_state.complexity_label = complexity.get('level', 'N/A')
                    
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
            
            progress_bar.progress(70)
            
            # Performance Analysis & Optimization
            if "Performance Analysis" in analysis_options:
                st.markdown("#### 🚀 Query Optimization")
                
                # Get optimization suggestions
                optimization_result = optimize_query(
                    query,
                    schema_info=st.session_state.get("schema_summary", ""),
                    table_stats=table_stats
                )
                store_analysis_results("performance", optimization_result)
                
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
            
            progress_bar.progress(90)
            
            # Index Recommendations
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
            
            # Query Plan
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
            
            # After optimization, execute optimized query if available
            if st.session_state.optimized_sql:
                optimized_results = execute_query(db_path, st.session_state.optimized_sql)
                store_query_results("optimized", optimized_results)
                debug_state("After Optimized Query")
            
            # Display results
            display_query_results()
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            if st.session_state.debug:
                st.exception(e)
            return

def generate_comprehensive_report(query: str, analysis_results: Dict) -> str:
    """Generate a comprehensive analysis report"""
    report = f"""# SQL Query Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Query Overview
```sql
{query}
```

## Analysis Summary
- Complexity Score: {analysis_results['complexity']['score']}/100
- Security Risk Level: {analysis_results['security']['risk_level'].upper()}
- Optimization Priority: {analysis_results['optimization_plan']['priority'].upper()}

## Detailed Analysis

### 1. Complexity Analysis
- Score: {analysis_results['complexity']['score']}/100
- Level: {analysis_results['complexity']['level']}

Factors:
{json.dumps(analysis_results['complexity']['factors'], indent=2)}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['complexity']['recommendations'])}

### 2. Security Analysis
Risk Level: {analysis_results['security']['risk_level'].upper()}

Vulnerabilities:
{chr(10).join(f"- {vuln}" for vuln in analysis_results['security']['vulnerabilities'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['security']['recommendations'])}

### 3. Performance Analysis
Issues:
{chr(10).join(f"- {issue}" for issue in analysis_results['performance']['issues'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['performance']['recommendations'])}

Estimated Impact: {analysis_results['performance']['estimated_impact']}

### 4. Data Quality Analysis
Issues:
{chr(10).join(f"- {issue}" for issue in analysis_results['quality']['issues'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis_results['quality']['recommendations'])}

## Optimization Plan
Priority: {analysis_results['optimization_plan']['priority'].upper()}

Recommended Optimizations:
{chr(10).join(f"- {opt}" for opt in analysis_results['optimization_plan']['optimizations'])}

Expected Improvements:
{json.dumps(analysis_results['optimization_plan']['estimated_improvement'], indent=2)}
"""
    return report

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

# Add this after the title section
with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    
    # Query Category Selection
    st.session_state.query_category = st.selectbox(
        "Query Category",
        list(QUERY_CATEGORIES.keys()),
        help="Select the type of query you're optimizing"
    )
    
    # Optimization Strategy
    st.session_state.optimization_strategy = st.selectbox(
        "Optimization Strategy",
        ["balanced", "performance", "readability", "security"],
        help="Choose optimization priority"
    )
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.session_state.advanced_settings = {
            "benchmark_iterations": st.slider("Benchmark Iterations", 1, 10, 3),
            "security_level": st.select_slider(
                "Security Level",
                ["low", "medium", "high"],
                value="medium"
            ),
            "enable_ai_suggestions": st.checkbox("Enable AI Suggestions", True),
            "show_query_metrics": st.checkbox("Show Query Metrics", True)
        }

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

# Footer and Chat Assistant
if st.session_state.get("optimized_sql") and isinstance(st.session_state.optimized_sql, str) and st.session_state.optimized_sql.strip():
    debug_state("Before Chat Assistant")
    
    # Add Chat Assistant in sidebar
    with st.sidebar:
        st.markdown("## 💬 Ask OptiQuery Assistant")
        st.info("Ask any questions about the query, its optimization, or SQL best practices!")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])
        
        # Get user question
        user_question = st.chat_input("Ask about the query...")
        
        if user_question:
            debug_state("Processing User Question")
            
            # Add user message to chat
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Generate context for the AI including full analysis results and query results
            context = f"""
Analysis Context:
- Original Query: {st.session_state.get("original_query", "")}
- Optimized Query: {st.session_state.get("optimized_sql", "")}
- Complexity Score: {st.session_state.get("complexity_score", 0)}
- Complexity Level: {st.session_state.get("complexity_label", "")}
- Identified Issues: {', '.join(st.session_state.get("issues", [])) if st.session_state.get("issues") else 'None'}
- Schema: {st.session_state.get("schema_summary", "Not available")}

Query Results:
Original Query:
- Execution Time: {st.session_state.query_results["original"].get("execution_time")}s
- Row Count: {st.session_state.query_results["original"].get("row_count")}
- Error: {st.session_state.query_results["original"].get("error")}

Optimized Query:
- Execution Time: {st.session_state.query_results["optimized"].get("execution_time")}s
- Row Count: {st.session_state.query_results["optimized"].get("row_count")}
- Error: {st.session_state.query_results["optimized"].get("error")}

Analysis Results:
{json.dumps(st.session_state.get("analysis_results", {}), indent=2)}

User Question: {user_question}

Provide a clear, concise answer focusing on the specific question. If relevant, reference the query analysis results and actual query performance metrics. If there were any errors in query execution, mention those as well.
"""
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a SQL expert assistant helping users understand query optimization. Be concise but thorough."},
                            {"role": "user", "content": context}
                        ],
                        temperature=0.7
                    )
                    answer = response.choices[0].message.content.strip()
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            debug_state("After Processing User Question")

# Footer
st.markdown("---")
st.markdown("📦 [View Source Code on GitHub](https://github.com/sanjaydp/optiquery)")

# Add Query History View in Sidebar
with st.sidebar:
    if st.session_state.query_history:
        st.markdown("## 📜 Query History")
        for idx, hist in enumerate(st.session_state.query_history):
            with st.expander(f"Query {idx + 1} - {hist['timestamp']}"):
                st.code(hist['query'], language="sql")
                if "benchmark_results" in hist:
                    st.markdown("**Performance:**")
                    st.markdown(f"- Execution Time: {hist['benchmark_results'].get('average_execution_time', 'N/A')}s")
                    st.markdown(f"- Result Size: {hist['benchmark_results'].get('result_set_size', 'N/A')} rows")

# Initialize session state at startup
init_session_state()
debug_state("App Startup")
