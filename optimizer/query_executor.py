import sqlite3  # or your preferred database module
import time  # Add missing time import
import statistics  # For calculating median execution time

def execute_query(db_path, query, num_runs=5):
    """Execute a query and measure its performance accurately.
    
    Args:
        db_path: Path to the SQLite database
        query: SQL query to execute
        num_runs: Number of times to run the query for timing (default: 5)
    """
    conn = None
    execution_times = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First run to warm up cache and get rows
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Multiple timing runs
        for _ in range(num_runs):
            # Clear any SQLite statement cache between runs
            cursor.execute("PRAGMA schema_version")
            
            start_time = time.time()
            cursor.execute(query)
            cursor.fetchall()  # Must fetch to complete the query
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        # Use median time as it's more stable than mean
        median_time = statistics.median(execution_times)
        
        return {
            "success": True,
            "rows": rows,
            "row_count": len(rows),
            "execution_time": round(median_time, 4),
            "min_time": round(min(execution_times), 4),
            "max_time": round(max(execution_times), 4)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()
