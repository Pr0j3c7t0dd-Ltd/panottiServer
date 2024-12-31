import sqlite3

from app.models.database import DatabaseManager


def check_schema() -> None:
    db = DatabaseManager.get_instance()

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()

        # Get table info
        cursor.execute("PRAGMA table_info(events)")
        columns: list[tuple] = cursor.fetchall()

        print("Events table schema:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")


if __name__ == "__main__":
    check_schema()
