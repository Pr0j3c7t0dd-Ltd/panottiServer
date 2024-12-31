import os
import sqlite3
from pathlib import Path


def migrate():
    # Get the database path
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    db_path = str(root_dir / "data" / "panotti.db")

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create a new events table with the updated schema
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                recording_id TEXT NOT NULL,
                recording_datetime TEXT,
                event_title TEXT,
                event_provider_id TEXT,
                event_provider TEXT,
                event_attendees TEXT,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Copy data from the old table to the new one
        cursor.execute(
            """
            INSERT INTO events_new (
                id, type, timestamp, recording_id, metadata_json, created_at
            )
            SELECT
                id, type, timestamp,
                json_extract(data, '$.recordingId') as recording_id,
                data as metadata_json,
                created_at
            FROM events
        """
        )

        # Drop the old table
        cursor.execute("DROP TABLE events")

        # Rename the new table to events
        cursor.execute("ALTER TABLE events_new RENAME TO events")

        # Commit the changes
        conn.commit()
        print("Migration completed successfully")

    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
