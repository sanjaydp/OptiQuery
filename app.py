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

def display_sql_with_copy(sql_code: str, key_suffix: str = ""):
    """Display a SQL code block with a copy button at the bottom left."""
    st.code(sql_code, language='sql')
    col1, col2, col3 = st.columns([1, 15, 1])
    with col1:
        button_key = f"copy_{key_suffix}_{hash(sql_code)}"
        copy_status_key = f"copy_status_{key_suffix}_{hash(sql_code)}"
        
        # Initialize the copy status in session state if not present
        if copy_status_key not in st.session_state:
            st.session_state[copy_status_key] = False
            
        if st.button("📋 Copy SQL", key=button_key, help="Copy to query editor"):
            st.session_state.query = sql_code
            st.session_state[copy_status_key] = True
        
        # Show success message if copy was performed
        if st.session_state[copy_status_key]:
            st.success("Copied to editor!")
            # Reset the status after showing the message
            st.session_state[copy_status_key] = False

def optimize_query(query: str) -> str:
    """Get optimization suggestions for the query using OpenAI."""
    try:
        if not st.session_state.get('openai_api_key'):
            return "⚠️ Please provide an OpenAI API key in the sidebar to get AI-powered optimization suggestions."
        
        openai.api_key = st.session_state.openai_api_key
        
        # Get database schema for context
        connection = get_database_connection()
        context = {"schema": ""}
        
        if connection:
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
                connection.close()
        
        # Prepare the prompt
        system_prompt = """You are an expert SQL optimizer. Analyze the given SQL query and provide optimization suggestions.
Focus on:
1. Performance improvements
2. Best practices
3. Potential issues
4. Index recommendations

When providing SQL examples, format them exactly like this:

[SQL]
SELECT * FROM table;
[/SQL]

For index recommendations, format them like this:

[SQL]
CREATE INDEX idx_name ON table(column);
[/SQL]

Database Schema:
{schema}

Provide your response in markdown format with clear sections."""

        user_prompt = f"""Analyze this SQL query for optimization opportunities:

{query}

Please provide specific, actionable recommendations with example queries where relevant."""
        
        # Call OpenAI API with the new client format
        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt.format(schema=context["schema"])},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Process the response
        content = response.choices[0].message.content
        
        # Split content and process each part
        parts = re.split(r'\[SQL\](.*?)\[/SQL\]', content, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text content
                st.markdown(part.strip())
            else:  # SQL content
                sql_code = part.strip()
                display_sql_with_copy(sql_code, f"opt_{i}")
                st.markdown("---")  # Add a separator between suggestions
        
        return ""
        
    except Exception as e:
        return f"❌ Error getting optimization suggestions: {str(e)}"

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

def analyze_query(query: str, perform_syntax_check: bool = True, perform_analysis: bool = True) -> str:
    """Analyze the SQL query for syntax and performance."""
    try:
        # Store the current query in session state to preserve it
        if 'current_analysis_query' not in st.session_state:
            st.session_state.current_analysis_query = query
        
        connection = get_database_connection()
        if not connection:
            return "❌ No database connection available. Please connect to a database first."

        cursor = connection.cursor()

        try:
            if perform_syntax_check:
                st.markdown("### Syntax Check")
                try:
                    cursor.execute(f"EXPLAIN {query}")
                    st.success("✅ Query syntax is valid")
                    display_sql_with_copy(query, "syntax_valid")
                except Exception as e:
                    st.error(f"❌ Syntax error: {str(e)}")
                    return ""

            if perform_analysis:
                st.markdown("### Performance Analysis")
                
                # Get the execution plan
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                plan = cursor.fetchall()
                
                if plan:
                    st.markdown("#### Execution Plan")
                    plan_text = "\n".join([str(row) for row in plan])
                    st.code(plan_text)
                    
                    # Display the original query with copy button
                    st.markdown("#### Original Query")
                    display_sql_with_copy(query, "perf_original")

                    # Get optimization suggestions
                    st.markdown("#### Optimization Suggestions")
                    optimize_query(query)

        finally:
            cursor.close()
            connection.close()

        return ""

    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def show_db_connection_form():
    """Show database connection settings in the sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Database Connection")
        
        # Database type selection
        db_type = st.radio(
            "Select Database Type:",
            ["SQLite", "PostgreSQL"],
            index=0 if st.session_state.get('use_sqlite', True) else 1,
            help="Choose your database type"
        )
        
        st.session_state.use_sqlite = (db_type == "SQLite")
        
        if st.session_state.use_sqlite:
            # SQLite connection options
            sqlite_option = st.radio(
                "Choose SQLite Option:",
                ["Upload Database File", "Use Sample Database", "Specify Database Path"],
                help="Select how you want to connect to SQLite"
            )
            
            if sqlite_option == "Upload Database File":
                uploaded_file = st.file_uploader(
                    "Upload SQLite Database",
                    type=['db', 'sqlite', 'sqlite3'],
                    help="Upload your SQLite database file"
                )
                if uploaded_file:
                    # Save the uploaded file
                    import os
                    save_path = os.path.join(os.getcwd(), "uploaded_database.db")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state.sqlite_path = save_path
                    st.session_state.db_initialized = True
                    st.session_state.schema_summary = get_schema_summary()
                    st.success("✅ Database file uploaded successfully!")
                    
            elif sqlite_option == "Use Sample Database":
                if st.button("Initialize Sample Database"):
                    with st.spinner("Initializing database..."):
                        initialize_sqlite_database()
                        st.session_state.sqlite_path = 'optiquery.db'
                        st.session_state.schema_summary = get_schema_summary()
                        st.session_state.db_initialized = True
                        st.success("✅ Sample database initialized!")
                        st.rerun()
            
            else:  # Specify Database Path
                sqlite_path = st.text_input(
                    "SQLite Database Path",
                    value=st.session_state.get('sqlite_path', 'optiquery.db'),
                    help="Path to your SQLite database file"
                )
                st.session_state.sqlite_path = sqlite_path
                if st.button("Connect to Database"):
                    try:
                        conn = get_database_connection()
                        if conn:
                            conn.close()
                            st.session_state.db_initialized = True
                            st.session_state.schema_summary = get_schema_summary()
                            st.success("✅ Successfully connected to database!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to connect to database: {str(e)}")
                
        else:
            # PostgreSQL settings
            col1, col2 = st.columns(2)
            with col1:
                pg_host = st.text_input(
                    "Host",
                    value=st.session_state.get('pg_host', 'localhost'),
                    help="PostgreSQL host"
                )
                pg_database = st.text_input(
                    "Database",
                    value=st.session_state.get('pg_database', ''),
                    help="PostgreSQL database name"
                )
                pg_user = st.text_input(
                    "User",
                    value=st.session_state.get('pg_user', ''),
                    help="PostgreSQL username"
                )
            
            with col2:
                pg_port = st.number_input(
                    "Port",
                    value=st.session_state.get('pg_port', 5432),
                    help="PostgreSQL port"
                )
                pg_password = st.text_input(
                    "Password",
                    type="password",
                    value=st.session_state.get('pg_password', ''),
                    help="PostgreSQL password"
                )
            
            # Store PostgreSQL settings in session state
            st.session_state.pg_host = pg_host
            st.session_state.pg_port = pg_port
            st.session_state.pg_database = pg_database
            st.session_state.pg_user = pg_user
            st.session_state.pg_password = pg_password
            
            # Test PostgreSQL connection
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    connection = get_database_connection()
                    if connection:
                        try:
                            cursor = connection.cursor()
                            cursor.execute("SELECT version();")
                            version = cursor.fetchone()[0]
                            st.success(f"✅ Connected to PostgreSQL {version}")
                            st.session_state.db_initialized = True
                            st.session_state.schema_summary = get_schema_summary()
                            cursor.close()
                            connection.close()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                            st.session_state.db_initialized = False
        
        # OpenAI API key for AI-powered suggestions
        st.markdown("### 🤖 AI Settings")
        openai_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="Enter your OpenAI API key for AI-powered optimization suggestions"
        )
        st.session_state.openai_api_key = openai_key
        
        # Show current database schema if available
        if st.session_state.get('schema_summary') and st.session_state.db_initialized:
            with st.expander("📚 Database Schema", expanded=False):
                st.text(st.session_state.schema_summary)

def main():
    
    # Initialize database connection state if not exists
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Show database connection form in sidebar
    show_db_connection_form()
    
    # Only show interface if database is initialized
    if not st.session_state.db_initialized:
        st.info("👆 Please connect to a database using the sidebar options first.")
        return
        
    # Create tabs for different functionalities
    query_tab, nl_tab, chat_tab = st.tabs(["🔍 SQL Query Optimization", "🤖 Natural Language to SQL", "💬 Chat"])
    
    with query_tab:
        # Query input method selection
        query_input_method = st.radio(
            "Choose input method:",
            ["Paste SQL Query", "Upload SQL File"],
            horizontal=True,
            help="Select how you want to input your SQL query"
        )
        
        query = None
        
        # Handle query input based on selected method
        if query_input_method == "Upload SQL File":
            # File upload section
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
            # Text input section
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
                value=st.session_state.get('query', sample_query),
                height=200,
                help="Paste your SQL query here for optimization analysis"
            )
        
        # Analysis options
        if query:  # Only show options if we have a query
            st.markdown("### ⚙️ Analysis Options")
            
            # Syntax check option
            run_syntax_check = st.checkbox(
                "✓ Syntax Check",
                value=True,
                help="Check SQL syntax before execution"
            )
            
            # Performance analysis option
            run_performance = st.checkbox(
                "🚀 Performance Analysis",
                value=True,
                help="Analyze query performance and suggest optimizations"
            )
            
            # Store query in session state
            st.session_state.query = query
            
            # Add analyze button
            if st.button("Analyze Query", type="primary"):
                if not run_syntax_check and not run_performance:
                    st.warning("Please select at least one analysis option.")
                else:
                    with st.spinner("Analyzing query..."):
                        # Run syntax check if selected
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
                                return
                        
                        # Run performance analysis if selected
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
                                            performance_metrics = {'median_time': format_time(median_time)}
                                        
                                        # Get result size
                                        cursor.execute(query)
                                        results = cursor.fetchall()
                                        performance_metrics['row_count'] = len(results)
                                        
                                        # Get column names and handle duplicates
                                        if st.session_state.get('use_sqlite', True):
                                            # For SQLite, get table info from the query plan
                                            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                                            plan = cursor.fetchall()
                                            tables = {}
                                            for row in plan:
                                                if 'SCAN' in row[3]:
                                                    table_name = row[3].split()[-1].strip('[]')
                                                    tables[table_name] = True
                                            
                                            # Get column descriptions
                                            cursor.execute(query)
                                            descriptions = cursor.description
                                            columns = []
                                            seen_columns = set()
                                            
                                            for desc in descriptions:
                                                col_name = desc[0]
                                                if col_name in seen_columns:
                                                    # If duplicate, try to find the table name
                                                    for table in tables:
                                                        qualified_name = f"{table}.{col_name}"
                                                        if qualified_name not in seen_columns:
                                                            columns.append(qualified_name)
                                                            seen_columns.add(qualified_name)
                                                            break
                                                    else:
                                                        # If no table found, append with a unique suffix
                                                        suffix = 1
                                                        while f"{col_name}_{suffix}" in seen_columns:
                                                            suffix += 1
                                                        columns.append(f"{col_name}_{suffix}")
                                                        seen_columns.add(f"{col_name}_{suffix}")
                                                else:
                                                    columns.append(col_name)
                                                    seen_columns.add(col_name)
                                        else:
                                            # For PostgreSQL, column names already include table names
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
                                            
                                        # Update query history
                                        update_query_history(query, performance_metrics)
                                    finally:
                                        cursor.close()
                                except Exception as e:
                                    st.error(f"Error executing query: {str(e)}")
                                finally:
                                    connection.close()

    with nl_tab:
        st.markdown("### Convert Natural Language to SQL")
        nl_query = st.text_area(
            "Enter your query in natural language:",
            placeholder="Example: Show me all orders from customers in the North region after January 2024 with total over $1000",
            help="Describe what you want to query in plain English"
        )
        
        if nl_query:
            if st.button("Convert to SQL", type="primary"):
                if not st.session_state.get('openai_api_key'):
                    st.error("⚠️ Please provide an OpenAI API key in the sidebar to use this feature")
                else:
                    with st.spinner("Converting to SQL..."):
                        try:
                            # Get schema information for context
                            schema_info = st.session_state.get('schema_summary', '')
                            
                            # Call OpenAI API with the new client format
                            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": f"You are a SQL expert. Convert natural language queries to SQL based on this schema:\n{schema_info}"},
                                    {"role": "user", "content": f"Convert this to SQL:\n{nl_query}"}
                                ]
                            )
                            
                            # Extract SQL from response
                            sql_query = response.choices[0].message.content
                            
                            # Display the result
                            st.code(sql_query, language='sql')
                            
                            # Add button to use this query
                            if st.button("Use this query"):
                                st.session_state.query = sql_query
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error converting to SQL: {str(e)}")
    
    with chat_tab:
        st.markdown("### SQL Assistant Chat")
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a follow-up question about your queries..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get context from query history
            query_context = ""
            if st.session_state.query_history:
                recent_queries = st.session_state.query_history[:3]  # Get 3 most recent queries
                query_context = "\n\n".join([
                    f"Query {i+1}:\n{q['query']}" 
                    for i, q in enumerate(recent_queries)
                ])
            
            # Get schema context
            schema_context = st.session_state.get('schema_summary', '')
            
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Call OpenAI API
                        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"""You are a SQL expert assistant. Help users understand and improve their SQL queries.
                                
Current database schema:
{schema_context}

Recent queries:
{query_context}

Provide clear, specific answers and include example SQL queries when relevant."""},
                                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
                            ]
                        )
                        
                        # Get and display response
                        assistant_response = response.choices[0].message.content
                        st.markdown(assistant_response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "API key" in str(e):
                    st.error("Please provide an OpenAI API key in the sidebar to use the chat feature.")
        
        # Add a clear chat button
        if st.session_state.chat_messages and st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()

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
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'query' not in st.session_state:
        st.session_state.query = ""
        
    main()
