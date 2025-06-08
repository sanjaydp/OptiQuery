import sqlite3
import psycopg2
import time
import statistics
import streamlit as st
from typing import List, Optional, Union, Tuple

def get_database_connection():
    """Get a database connection based on the current configuration."""
    if st.session_state.get('use_sqlite', True):
        # Default to SQLite
        db_path = st.session_state.get('sqlite_path', 'optiquery.db')
        try:
            return sqlite3.connect(db_path)
        except sqlite3.Error as e:
            st.error(f"❌ SQLite database error: {str(e)}")
            return None
    else:
        # PostgreSQL connection
        try:
            conn = psycopg2.connect(
                host=st.session_state.get('pg_host', 'localhost'),
                port=st.session_state.get('pg_port', 5432),
                database=st.session_state.get('pg_database', ''),
                user=st.session_state.get('pg_user', ''),
                password=st.session_state.get('pg_password', '')
            )
            return conn
        except psycopg2.Error as e:
            st.error(f"❌ PostgreSQL database error: {str(e)}")
            return None

def measure_query_execution_time(connection, query: str, num_runs: int = 3) -> List[float]:
    """Measure query execution time over multiple runs."""
    execution_times = []
    
    try:
        cursor = connection.cursor()
        for _ in range(num_runs):
            start_time = time.time()
            cursor.execute(query)
            cursor.fetchall()  # Ensure query is fully executed
            end_time = time.time()
            execution_times.append(end_time - start_time)
        cursor.close()
    except (sqlite3.Error, psycopg2.Error) as e:
        st.error(f"Error measuring query execution time: {str(e)}")
        return []
        
    return execution_times

def initialize_sqlite_database():
    """Initialize SQLite database with sample data if it doesn't exist."""
    connection = sqlite3.connect('optiquery.db')
    cursor = connection.cursor()
    
    try:
        # Create sample tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            region TEXT,
            join_date DATE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date DATE,
            order_total DECIMAL(10,2),
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
        """)
        
        # Add sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                "INSERT INTO customers (name, email, region, join_date) VALUES (?, ?, ?, ?)",
                [
                    ("John Doe", "john@example.com", "North", "2024-01-01"),
                    ("Jane Smith", "jane@example.com", "South", "2024-01-02"),
                    ("Bob Wilson", "bob@example.com", "East", "2024-01-03"),
                    ("Alice Brown", "alice@example.com", "West", "2024-01-04")
                ]
            )
            
            cursor.executemany(
                "INSERT INTO orders (customer_id, order_date, order_total, status) VALUES (?, ?, ?, ?)",
                [
                    (1, "2024-01-15", 1500.00, "completed"),
                    (1, "2024-02-01", 2000.00, "completed"),
                    (2, "2024-01-20", 1200.00, "completed"),
                    (3, "2024-02-05", 800.00, "pending"),
                    (4, "2024-02-10", 3000.00, "completed")
                ]
            )
            
        connection.commit()
        
    except sqlite3.Error as e:
        st.error(f"Error initializing database: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def get_schema_summary() -> str:
    """Get a summary of the database schema."""
    connection = get_database_connection()
    if not connection:
        return "Unable to connect to database"
        
    try:
        cursor = connection.cursor()
        schema_info = []
        
        if st.session_state.get('use_sqlite', True):
            # SQLite schema
            cursor.execute("""
                SELECT name, sql 
                FROM sqlite_master 
                WHERE type='table'
            """)
            tables = cursor.fetchall()
            
            for table_name, table_sql in tables:
                schema_info.append(f"Table: {table_name}")
                schema_info.append(table_sql)
                schema_info.append("")
                
        else:
            # PostgreSQL schema
            cursor.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                ORDER BY table_name, ordinal_position
            """)
            current_table = None
            for table, column, dtype in cursor.fetchall():
                if current_table != table:
                    current_table = table
                    schema_info.append(f"\nTable: {table}")
                schema_info.append(f"  {column}: {dtype}")
                
        return "\n".join(schema_info)
        
    except (sqlite3.Error, psycopg2.Error) as e:
        return f"Error getting schema: {str(e)}"
    finally:
        cursor.close()
        connection.close() 