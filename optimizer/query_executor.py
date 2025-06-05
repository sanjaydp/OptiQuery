import sqlite3
import time

def execute_query(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        start_time = time.time()
        cursor.execute(query)
        rows = cursor.fetchall()
        execution_time = time.time() - start_time

        return {
            "success": True,
            "rows": rows[:10],  # limit preview
            "row_count": len(rows),
            "execution_time": round(execution_time, 4)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        conn.close()
