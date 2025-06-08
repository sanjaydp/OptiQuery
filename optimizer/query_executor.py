import sqlite3  # or your preferred database module
import time  # Add missing time import

def execute_query(db_path, query):
    conn = None  # ✅ Ensure it's defined before try block
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        start_time = time.time()
        cursor.execute(query)
        rows = cursor.fetchall()
        execution_time = time.time() - start_time

        return {
            "success": True,
            "rows": rows,
            "row_count": len(rows),
            "execution_time": round(execution_time, 4)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()
