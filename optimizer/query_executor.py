import sqlite3  # or your preferred database module

def execute_query(db_path, query):
    """Execute the SQL query and return the results."""
    conn = None  # Initialize conn variable
    try:
        # Establish connection to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return {"success": True, "rows": rows, "row_count": len(rows), "execution_time": 1.23}  # Simulated execution time
        
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    finally:
        if conn:
            conn.close()  # Ensure connection is closed after query execution
