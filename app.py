import streamlit as st
from dotenv import load_dotenv
from optimizer.sql_parser import analyze_sql
from optimizer.llm_optimizer import optimize_query, explain_optimization, format_sql_query
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
import statistics
import openai
from database import get_database_connection, measure_query_execution_time, initialize_sqlite_database, get_schema_summary
from optimization import optimize_query

# Try importing PostgreSQL support
try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    st.warning("PostgreSQL support not available. Using SQLite as fallback.")

# Page Configuration must be the first Streamlit command
st.set_page_config(
    page_title="OptiQuery – SQL Optimizer Assistant",
    page_icon="🧠",
    layout="wide"
)

# Function definitions first
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
    if not st.session_state.debug:
        return
        
    st.sidebar.markdown(f"### 🔍 Debug Info: {location}")
    st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
    st.sidebar.write("Has Query Results:", bool(st.session_state.query_results["original"]["data"] or st.session_state.query_results["optimized"]["data"]))
    st.sidebar.write("Analysis Results Keys:", list(st.session_state.analysis_results.keys()))
    st.sidebar.write("Last Analysis ID:", st.session_state.last_analysis_id)

# Initialize debug flag first
if "debug" not in st.session_state:
    st.session_state.debug = False  # Disable debugging by default

# Now initialize the rest of the session state
if "initialized" not in st.session_state:
    init_session_state()
debug_state("App Startup")

# Initialize session state for database settings
if 'use_sqlite' not in st.session_state:
    st.session_state.use_sqlite = True

if 'sqlite_path' not in st.session_state:
    # Create default SQLite database
    temp_dir = "temp_db"
    os.makedirs(temp_dir, exist_ok=True)
    default_db_path = os.path.join(temp_dir, "optiquery_default.db")
    
    if not os.path.exists(default_db_path):
        try:
            # Create new database with a sample table
            conn = sqlite3.connect(default_db_path)
            cursor = conn.cursor()
            
            # Create a sample table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sample_data (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                )
            """)
            
            # Insert some sample data
            sample_data = [
                (1, 'Example A', 100),
                (2, 'Example B', 200),
                (3, 'Example C', 300)
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO sample_data (id, name, value) VALUES (?, ?, ?)",
                sample_data
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.error(f"Error creating default database: {str(e)}")
    
    st.session_state.sqlite_path = default_db_path

def get_database_connection():
    """Get a database connection with proper error handling."""
    try:
        # Check if using SQLite
        if st.session_state.get('use_sqlite', True):  # Default to SQLite if no preference
            db_path = st.session_state.get('sqlite_path')
            if not db_path:
                st.error("⚠️ SQLite database path not configured.")
                return None
            return sqlite3.connect(db_path)
        
        # Try PostgreSQL connection if selected
        if HAS_POSTGRES:
            # Get connection parameters from session state or environment
            db_params = st.session_state.get('db_params', {})
            if not db_params:
                st.error("⚠️ Database connection not configured. Please set up your database connection first.")
                return None
                
            # Create connection
            connection = psycopg2.connect(
                host=db_params.get('host', 'localhost'),
                port=db_params.get('port', 5432),
                database=db_params.get('database'),
                user=db_params.get('user'),
                password=db_params.get('password')
            )
            
            return connection
        else:
            st.error("⚠️ PostgreSQL support not available. Please use SQLite instead.")
            return None
            
    except Exception as e:
        st.error("❌ Database connection error:")
        st.error(str(e))
        return None

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
    if results["success"]:
        st.session_state.query_results[result_type] = {
            "data": results["rows"],
            "execution_time": results["execution_time"],
            "row_count": results["row_count"],
            "min_time": results.get("min_time"),
            "max_time": results.get("max_time"),
            "error": None
        }
    else:
        st.session_state.query_results[result_type] = {
            "data": None,
            "execution_time": None,
            "row_count": None,
            "min_time": None,
            "max_time": None,
            "error": results.get("error", "Unknown error occurred")
        }
    
    # Force Streamlit to recognize the state change
    st.session_state.last_analysis_id = datetime.now().isoformat()

def display_query_results():
    """Display query results and comparison with debugging"""
    debug_state("Before Displaying Results")
    
    def format_time(time_value):
        """Format time value with appropriate units and precision"""
        if time_value is None:
            return "N/A"
        if time_value < 0.001:  # Less than 1ms
            return f"{time_value * 1000000:.2f}μs"
        elif time_value < 1:  # Less than 1s
            return f"{time_value * 1000:.2f}ms"
        else:
            return f"{time_value:.3f}s"
    
    if st.session_state.query_results["original"]["data"] is not None:
        st.markdown("#### 📊 Original Query Results")
        st.dataframe(pd.DataFrame(st.session_state.query_results["original"]["data"]))
        
        # Show detailed timing metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            exec_time = st.session_state.query_results['original']['execution_time']
            st.metric(
                "Original Execution Time",
                format_time(exec_time),
                help="Median execution time over multiple runs"
            )
        with col2:
            min_time = st.session_state.query_results['original'].get('min_time')
            st.metric(
                "Min Time",
                format_time(min_time)
            )
        with col3:
            max_time = st.session_state.query_results['original'].get('max_time')
            st.metric(
                "Max Time",
                format_time(max_time)
            )

    if st.session_state.query_results["optimized"]["data"] is not None:
        st.markdown("#### 🚀 Optimized Query Results")
        st.dataframe(pd.DataFrame(st.session_state.query_results["optimized"]["data"]))
        
        # Show detailed timing metrics for optimized query
        col1, col2, col3 = st.columns(3)
        with col1:
            # Calculate improvement percentage
            orig_time = st.session_state.query_results["original"]["execution_time"]
            opt_time = st.session_state.query_results["optimized"]["execution_time"]
            if orig_time and opt_time:
                improvement = ((orig_time - opt_time) / orig_time * 100)
                # Only show improvement if optimized query is actually faster
                delta = f"{improvement:.1f}%" if improvement > 0 else f"{-improvement:.1f}% slower"
                delta_color = "normal" if improvement > 0 else "inverse"
                st.metric(
                    "Optimized Execution Time",
                    format_time(opt_time),
                    delta=delta,
                    delta_color=delta_color
                )
        with col2:
            min_time = st.session_state.query_results['optimized'].get('min_time')
            st.metric(
                "Min Time",
                format_time(min_time)
            )
        with col3:
            max_time = st.session_state.query_results['optimized'].get('max_time')
            st.metric(
                "Max Time",
                format_time(max_time)
            )
        
        # Show optimization confidence and warnings
        performance_results = st.session_state.analysis_results.get("performance", {})
        if performance_results:
            st.markdown("#### 📈 Optimization Analysis")
            col1, col2 = st.columns(2)
            with col1:
                confidence = performance_results.get("confidence", "N/A")
                confidence_color = {
                    "high": "🟢",
                    "medium": "🟡",
                    "low": "🔴"
                }.get(confidence.lower(), "⚪")
                st.markdown(f"**Confidence Level:** {confidence_color} {confidence}")
            
            with col2:
                # Compare estimated vs actual improvement
                estimated = performance_results.get("estimated_improvement", "N/A")
                actual = f"{improvement:.1f}%" if orig_time and opt_time else "N/A"
                st.markdown(f"**Estimated Improvement:** {estimated}")
                st.markdown(f"**Actual Improvement:** {actual}")
            
            # Show changes made
            changes = performance_results.get("changes_made", [])
            if changes:
                st.markdown("**Changes Made:**")
                for change in changes:
                    st.markdown(f"- {change}")
            
            # Show warnings
            warnings = performance_results.get("warnings", [])
            if warnings:
                st.markdown("**⚠️ Warnings:**")
                for warning in warnings:
                    st.warning(warning)
                
            # Add warning if actual performance is worse than original
            if orig_time and opt_time and opt_time > orig_time:
                st.warning("⚠️ The optimized query is currently performing slower than the original query. This might be due to:\n" +
                          "- Small data set where optimization overhead exceeds benefits\n" +
                          "- Database caching effects\n" +
                          "- Need for index creation\n" +
                          "Consider running the analysis multiple times or with a larger dataset.")
    
    debug_state("After Displaying Results")

def format_execution_time(microseconds: float) -> str:
    """Format execution time in appropriate units."""
    if microseconds >= 1_000_000:  # >= 1 second
        return f"{microseconds/1_000_000:.2f}s"
    elif microseconds >= 1_000:  # >= 1 millisecond
        return f"{microseconds/1_000:.2f}ms"
    else:
        return f"{microseconds:.2f}μs"

def measure_query_execution_time(connection, query: str, runs: int = 5) -> list:
    """Measure query execution time over multiple runs."""
    execution_times = []
    
    cursor = connection.cursor()
    try:
        # Warm up the cache with one execution
        cursor.execute("EXPLAIN " + query)
        cursor.fetchall()
        
        # Perform multiple runs
        for _ in range(runs):
            start_time = time.perf_counter()
            
            cursor.execute(query)
            cursor.fetchall()  # Ensure query is fully executed
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return []
    finally:
        cursor.close()
        
    return execution_times

def get_query_plan(connection, query: str) -> str:
    """Get the query execution plan."""
    cursor = connection.cursor()
    try:
        if st.session_state.get('use_sqlite', True):
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        else:
            cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
        
        plan = cursor.fetchall()
        return str(plan)
    except Exception as e:
        st.error(f"Error getting query plan: {str(e)}")
        return "Could not get query plan"
    finally:
        cursor.close()

def optimize_query(query: str) -> str:
    """Get optimization suggestions for the query."""
    try:
        # Get database connection
        connection = get_database_connection()
        if not connection:
            return "Could not connect to database"
            
        try:
            # Get query plan
            plan = get_query_plan(connection, query)
            
            # Measure original query performance
            original_times = measure_query_execution_time(connection, query)
            if not original_times:
                return "Could not measure query performance"
            
            # Calculate statistics
            median_time = statistics.median(original_times)
            min_time = min(original_times)
            max_time = max(original_times)
            
            # Format execution time with appropriate units
            def format_time(seconds):
                if seconds < 0.000001:  # Less than 1 microsecond
                    return f"{seconds * 1000000000:.2f} ns"
                elif seconds < 0.001:  # Less than 1 millisecond
                    return f"{seconds * 1000000:.2f} μs"
                elif seconds < 1:  # Less than 1 second
                    return f"{seconds * 1000:.2f} ms"
                else:
                    return f"{seconds:.2f} s"
            
            # Prepare optimization context
            context = {
                "query": query,
                "execution_plan": plan,
                "median_time": format_time(median_time),
                "min_time": format_time(min_time),
                "max_time": format_time(max_time),
                "run_count": len(original_times)
            }
            
            # Get schema information
            cursor = connection.cursor()
            try:
                if st.session_state.get('use_sqlite', True):
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    schema_info = []
                    for (table_name,) in tables:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = cursor.fetchall()
                        schema_info.append(f"Table: {table_name}")
                        schema_info.append(f"Columns: {', '.join(f'{col[1]} ({col[2]})' for col in columns)}")
                        schema_info.append("")
                else:
                    cursor.execute("""
                        SELECT 
                            table_schema || '.' || table_name as table_name,
                            string_agg(column_name || ' ' || data_type, ', ' ORDER BY ordinal_position) as columns
                        FROM information_schema.columns
                        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                        GROUP BY table_schema, table_name
                    """)
                    schema_info = []
                    for table_name, columns in cursor.fetchall():
                        schema_info.append(f"Table: {table_name}")
                        schema_info.append(f"Columns: {columns}")
                        schema_info.append("")
                
                context["schema"] = "\n".join(schema_info)
            finally:
                cursor.close()
            
            # Get optimization suggestions from OpenAI
            optimization_prompt = f"""
            As a SQL optimization expert, analyze this query and its execution metrics:
            
            Original Query:
            {query}
            
            Execution Plan:
            {plan}
            
            Performance Metrics:
            - Median Time: {context['median_time']}
            - Min Time: {context['min_time']}
            - Max Time: {context['max_time']}
            - Number of Runs: {context['run_count']}
            
            Schema Information:
            {context.get('schema', 'No schema information available')}
            
            Please provide:
            1. Analysis of the current query performance
            2. Specific optimization suggestions
            3. An optimized version of the query
            4. Expected performance impact of the optimizations
            
            Focus on practical improvements that will have the most significant impact on performance.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a SQL optimization expert. Provide clear, practical advice for query optimization."},
                    {"role": "user", "content": optimization_prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        finally:
            connection.close()
            
    except Exception as e:
        return f"Error during optimization: {str(e)}"

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

def format_sql_for_display(query: str) -> str:
    """Format SQL query for display with proper syntax highlighting."""
    return format_sql_query(query)

def create_copy_button(text: str, button_text: str = "📋", help_text: str = "Copy to clipboard") -> None:
    """Create a copy button for text content."""
    if st.button(button_text, help=help_text):
        st.session_state.clipboard = text
        st.success("Copied!")

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

# Chat Assistant Section
def render_chat_assistant():
    """Render the chat assistant if optimized query is available"""
    if not st.session_state.get("optimized_sql"):
        return
        
    debug_state("Before Chat Assistant")
    
    with st.sidebar:
        st.markdown("## 💬 Ask OptiQuery Assistant")
        st.info("Ask any questions about the query, its optimization, or SQL best practices!")
        
        # Display chat history
        for msg in st.session_state.get("chat_history", []):
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
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
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

# Add database connection setup UI
def show_db_connection_form():
    """Show database connection setup form."""
    st.sidebar.markdown("### 🔌 Database Connection")
    
    # Show current database status
    if st.session_state.get('sqlite_path'):
        st.sidebar.success(f"✅ Using SQLite database: {os.path.basename(st.session_state.sqlite_path)}")
    
    # Database type selector
    db_type = st.sidebar.selectbox(
        "Database Type",
        ["SQLite", "PostgreSQL"] if HAS_POSTGRES else ["SQLite"],
        key="db_type",
        help="SQLite is always available. PostgreSQL requires additional setup."
    )
    
    if db_type == "PostgreSQL" and HAS_POSTGRES:
        st.session_state.use_sqlite = False
        
        # Initialize session state for db_params if not exists
        if 'db_params' not in st.session_state:
            st.session_state.db_params = {
                'host': 'localhost',
                'port': 5432,
                'database': '',
                'user': '',
                'password': ''
            }
        
        # Create form for PostgreSQL connection
        with st.sidebar.form("db_connection_form"):
            st.text_input(
                "Host",
                value=st.session_state.db_params.get('host', 'localhost'),
                key="db_host"
            )
            st.number_input(
                "Port",
                value=st.session_state.db_params.get('port', 5432),
                key="db_port"
            )
            st.text_input(
                "Database",
                value=st.session_state.db_params.get('database', ''),
                key="db_name"
            )
            st.text_input(
                "Username",
                value=st.session_state.db_params.get('user', ''),
                key="db_user"
            )
            st.text_input(
                "Password",
                value=st.session_state.db_params.get('password', ''),
                type="password",
                key="db_password"
            )
            
            if st.form_submit_button("Connect"):
                # Update connection parameters
                st.session_state.db_params = {
                    'host': st.session_state.db_host,
                    'port': st.session_state.db_port,
                    'database': st.session_state.db_name,
                    'user': st.session_state.db_user,
                    'password': st.session_state.db_password
                }
                
                # Test connection
                connection = get_database_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        try:
                            cursor.execute("SELECT version();")
                            version = cursor.fetchone()[0]
                            st.sidebar.success(f"✅ Connected to PostgreSQL {version}")
                            
                            # Get and store schema information
                            cursor.execute("""
                                SELECT 
                                    table_schema || '.' || table_name as table_name,
                                    string_agg(column_name || ' ' || data_type, ', ' ORDER BY ordinal_position) as columns
                                FROM information_schema.columns
                                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                                GROUP BY table_schema, table_name
                            """)
                            schema_info = []
                            for table_name, columns in cursor.fetchall():
                                schema_info.append(f"Table: {table_name}")
                                schema_info.append(f"Columns: {columns}")
                                schema_info.append("")
                            
                            st.session_state.schema_summary = "\n".join(schema_info)
                        finally:
                            cursor.close()
                            
                    except Exception as e:
                        st.sidebar.error(f"Error fetching database info: {str(e)}")
                    finally:
                        connection.close()
    else:
        st.session_state.use_sqlite = True
        
        # SQLite file uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload SQLite Database",
            type=['db', 'sqlite', 'sqlite3'],
            key="sqlite_file",
            help="Upload an existing SQLite database file"
        )
        
        # Option to create new database
        create_new = st.sidebar.checkbox(
            "Create New Database",
            key="create_new_db",
            help="Create a new SQLite database if you don't have one"
        )
        
        if create_new:
            db_name = st.sidebar.text_input(
                "Database Name",
                value="optiquery.db",
                key="new_db_name",
                help="Name for the new database file"
            )
            
            if st.sidebar.button("Create Database"):
                try:
                    temp_dir = "temp_db"
                    os.makedirs(temp_dir, exist_ok=True)
                    db_path = os.path.join(temp_dir, db_name)
                    
                    # Create new database
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    try:
                        # Create a sample table
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS sample_data (
                                id INTEGER PRIMARY KEY,
                                name TEXT,
                                value INTEGER
                            )
                        """)
                        conn.commit()
                    finally:
                        cursor.close()
                        conn.close()
                    
                    st.session_state.sqlite_path = db_path
                    st.sidebar.success(f"✅ Created new database: {db_name}")
                    
                except Exception as e:
                    st.sidebar.error(f"Error creating database: {str(e)}")
        
        elif uploaded_file:
            # Save the uploaded file
            try:
                temp_dir = "temp_db"
                os.makedirs(temp_dir, exist_ok=True)
                db_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(db_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.session_state.sqlite_path = db_path
                
                # Test connection
                connection = get_database_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        try:
                            cursor.execute("SELECT sqlite_version();")
                            version = cursor.fetchone()[0]
                            st.sidebar.success(f"✅ Connected to SQLite {version}")
                            
                            # Get schema information
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = cursor.fetchall()
                            
                            schema_info = []
                            for (table_name,) in tables:
                                cursor.execute(f"PRAGMA table_info({table_name})")
                                columns = cursor.fetchall()
                                schema_info.append(f"Table: {table_name}")
                                schema_info.append(f"Columns: {', '.join(f'{col[1]} ({col[2]})' for col in columns)}")
                                schema_info.append("")
                            
                            st.session_state.schema_summary = "\n".join(schema_info)
                        finally:
                            cursor.close()
                            
                    except Exception as e:
                        st.sidebar.error(f"Error fetching database info: {str(e)}")
                    finally:
                        connection.close()
                        
            except Exception as e:
                st.sidebar.error(f"Error uploading database: {str(e)}")
        else:
            st.sidebar.info("Using default database. You can upload a different database file or create a new one.")
            
            # Show schema of default database
            connection = get_database_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    try:
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        
                        schema_info = []
                        for (table_name,) in tables:
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()
                            schema_info.append(f"Table: {table_name}")
                            schema_info.append(f"Columns: {', '.join(f'{col[1]} ({col[2]})' for col in columns)}")
                            schema_info.append("")
                        
                        st.session_state.schema_summary = "\n".join(schema_info)
                    finally:
                        cursor.close()
                        
                except Exception as e:
                    st.sidebar.error(f"Error fetching database info: {str(e)}")
                finally:
                    connection.close()

# Main app code
def update_query_history(query: str, performance_metrics: dict = None):
    """Update the query history in session state."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Create history entry
    history_entry = {
        'query': query,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Add performance metrics if available
    if performance_metrics:
        history_entry['performance'] = performance_metrics
    
    # Add to history (keep last 10 queries)
    st.session_state.query_history.insert(0, history_entry)
    st.session_state.query_history = st.session_state.query_history[:10]

def analyze_query(query: str, run_syntax_check: bool = True, run_performance: bool = True) -> bool:
    """Analyze and optimize the SQL query."""
    try:
        # Syntax check if enabled
        if run_syntax_check:
            st.markdown("#### 🔍 Syntax Check")
            try:
                connection = get_database_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        try:
                            # Try parsing the query without executing
                            if st.session_state.get('use_sqlite', True):
                                cursor.execute(f"EXPLAIN {query}")
                            else:
                                cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
                            cursor.fetchall()
                            st.success("✅ SQL syntax is valid")
                        finally:
                            cursor.close()
                    finally:
                        connection.close()
            except Exception as e:
                st.error("❌ SQL syntax error:")
                st.error(str(e))
                return False
        
        # Performance analysis if enabled
        performance_metrics = {}
        if run_performance:
            st.markdown("#### 🚀 Performance Analysis")
            
            # Get optimization suggestions
            optimization_result = optimize_query(query)
            st.markdown(optimization_result)
            
            # Collect performance metrics
            connection = get_database_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    try:
                        # Get execution time
                        execution_times = measure_query_execution_time(connection, query)
                        if execution_times:
                            median_time = statistics.median(execution_times)
                            performance_metrics['median_time'] = format_time(median_time)
                        
                        # Get result size
                        cursor.execute(query)
                        results = cursor.fetchall()
                        performance_metrics['row_count'] = len(results)
                        
                        # Get column names for display
                        if st.session_state.get('use_sqlite', True):
                            columns = [description[0] for description in cursor.description]
                        else:
                            columns = [desc.name for desc in cursor.description]
                        
                        # Show results
                        df = pd.DataFrame(results, columns=columns)
                        with st.expander("🔍 Query Results", expanded=True):
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True
                            )
                            st.caption(f"Showing {len(df)} rows")
                    finally:
                        cursor.close()
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
                    return False
                finally:
                    connection.close()
        
        # Update query history
        update_query_history(query, performance_metrics)
        return True
        
    except Exception as e:
        st.error("❌ Error during analysis:")
        st.error(str(e))
        return False

def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.000001:  # Less than 1 microsecond
        return f"{seconds * 1000000000:.2f} ns"
    elif seconds < 0.001:  # Less than 1 millisecond
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:  # Less than 1 second
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def main():
    st.title("OptiQuery: SQL Optimizer Assistant")
    
    # Show database connection form in sidebar
    show_db_connection_form()
    
    # Main query input area
    st.markdown("### 🔍 Query Optimization")
    
    # Query input method selection
    query_input_method = st.radio(
        "Choose input method:",
        ["Paste SQL Query", "Upload SQL File"],
        horizontal=True,
        help="Select how you want to input your SQL query"
    )
    
    query = None
    
    if query_input_method == "Upload SQL File":
        uploaded_file = st.file_uploader(
            "Upload SQL file",
            type=['sql'],
            help="Upload a .sql file containing your query"
        )
        if uploaded_file:
            try:
                query = uploaded_file.getvalue().decode('utf-8')
                with st.expander("📄 SQL File Contents", expanded=True):
                    st.code(query, language='sql')
            except Exception as e:
                st.error(f"Error reading SQL file: {str(e)}")
    else:
        # Sample query template
        sample_query = """SELECT *
FROM orders
WHERE customer_id IN (
    SELECT customer_id 
    FROM customers 
    WHERE region = 'North'
)
AND order_date >= '2024-01-01'
AND order_total > 1000;"""
        
        query = st.text_area(
            "Enter your SQL query:",
            value=sample_query if not st.session_state.get('query') else st.session_state.get('query'),
            height=200,
            help="Paste your SQL query here for optimization analysis"
        )
    
    # Analysis options
    with st.expander("⚙️ Analysis Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            run_syntax_check = st.checkbox(
                "Syntax Check",
                value=True,
                help="Check SQL syntax before execution"
            )
        with col2:
            run_performance = st.checkbox(
                "Performance Analysis",
                value=True,
                help="Analyze query performance and suggest optimizations"
            )
    
    # Only show optimization button if we have a query
    if query:
        # Store query in session state
        st.session_state.query = query
        
        col1, col2 = st.columns([1, 4])
        with col1:
            run_optimization = st.button(
                "Analyze & Optimize",
                type="primary",
                help="Run selected analysis on the query"
            )
            
        if run_optimization:
            with st.spinner("Analyzing query..."):
                analyze_query(query, run_syntax_check, run_performance)
    
    # Add Query History View in Sidebar
    with st.sidebar:
        if st.session_state.get('query_history', []):
            st.markdown("### 📜 Query History")
            for idx, hist in enumerate(st.session_state.query_history):
                with st.expander(f"Query {idx + 1} - {hist['timestamp']}", expanded=False):
                    st.code(hist['query'], language="sql")
                    if "performance" in hist:
                        st.markdown("**Performance:**")
                        st.markdown(f"- Execution Time: {hist['performance'].get('median_time', 'N/A')}")
                        st.markdown(f"- Result Size: {hist['performance'].get('row_count', 'N/A')} rows")

if __name__ == "__main__":
    # Initialize session state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'query' not in st.session_state:
        st.session_state.query = None
        
    main()
