from app.models.database import DatabaseManager
import sqlite3

def check_schema():
    db = DatabaseManager.get_instance()
    
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(events)")
        columns = cursor.fetchall()
        
        print("Events table schema:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")

if __name__ == '__main__':
    check_schema()
