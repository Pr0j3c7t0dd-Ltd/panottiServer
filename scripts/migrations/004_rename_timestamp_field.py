import sqlite3
import os
from pathlib import Path
import json

def migrate():
    """Rename timestamp field to recording_timestamp in events table"""
    # Get the project root directory (three levels up from this file)
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_dir = root_dir / 'data'
    db_path = str(data_dir / 'panotti.db')

    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        try:
            # Create new table with renamed field
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    recording_timestamp DATETIME NOT NULL,
                    recording_id TEXT NOT NULL,
                    event_title TEXT,
                    event_provider_id TEXT,
                    event_provider TEXT,
                    event_attendees TEXT,  -- JSON array
                    system_label TEXT,     -- Label for system audio source
                    microphone_label TEXT, -- Label for microphone audio source
                    recording_started TEXT, -- ISO8601 timestamp when recording started
                    recording_ended TEXT,   -- ISO8601 timestamp when recording ended
                    metadata_json TEXT,    -- Full JSON payload
                    system_audio_path TEXT,
                    microphone_audio_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Copy data to new table with renamed field
            cursor.execute('''
                INSERT INTO events_new (
                    id, type, recording_timestamp, recording_id,
                    event_title, event_provider_id, event_provider,
                    event_attendees, system_label, microphone_label,
                    recording_started, recording_ended,
                    metadata_json, system_audio_path, microphone_audio_path,
                    created_at
                )
                SELECT 
                    id, type, timestamp, recording_id,
                    event_title, event_provider_id, event_provider,
                    event_attendees, system_label, microphone_label,
                    recording_started, recording_ended,
                    metadata_json, system_audio_path, microphone_audio_path,
                    created_at
                FROM events
            ''')

            # Drop the old table
            cursor.execute('DROP TABLE IF EXISTS events')

            # Rename the new table to events
            cursor.execute('ALTER TABLE events_new RENAME TO events')

            # Commit the changes
            conn.commit()
            print("Migration completed successfully")

        except Exception as e:
            print(f"Error during migration: {e}")
            conn.rollback()
            raise

if __name__ == '__main__':
    migrate()