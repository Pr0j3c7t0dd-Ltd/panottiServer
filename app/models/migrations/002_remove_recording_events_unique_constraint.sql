-- Drop the old table and recreate it without the unique constraint
DROP TABLE IF EXISTS recording_events_old;
ALTER TABLE recording_events RENAME TO recording_events_old;

CREATE TABLE recording_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_timestamp TEXT NOT NULL,
    system_audio_path TEXT,
    microphone_audio_path TEXT,
    metadata TEXT,  -- JSON object with event metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Copy data from old table
INSERT INTO recording_events 
SELECT * FROM recording_events_old;

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_recording_events_recording_id
ON recording_events(recording_id);

-- Drop the old table
DROP TABLE recording_events_old;
