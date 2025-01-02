-- Add audio path columns to recording_events table if they don't exist
ALTER TABLE recording_events ADD COLUMN system_audio_path TEXT DEFAULT NULL;
ALTER TABLE recording_events ADD COLUMN microphone_audio_path TEXT DEFAULT NULL;

-- Add audio path columns to recordings table if they don't exist
ALTER TABLE recordings ADD COLUMN system_audio_path TEXT DEFAULT NULL;
ALTER TABLE recordings ADD COLUMN microphone_audio_path TEXT DEFAULT NULL;
ALTER TABLE recordings ADD COLUMN processed_audio_path TEXT DEFAULT NULL;

-- Now that columns exist, update recordings with data from events
UPDATE recordings
SET 
    system_audio_path = (
        SELECT re.system_audio_path 
        FROM recording_events re
        WHERE re.recording_id = recordings.recording_id 
        AND re.event_type = 'recording.ended'
        ORDER BY re.event_timestamp DESC 
        LIMIT 1
    ),
    microphone_audio_path = (
        SELECT re.microphone_audio_path 
        FROM recording_events re
        WHERE re.recording_id = recordings.recording_id 
        AND re.event_type = 'recording.ended'
        ORDER BY re.event_timestamp DESC 
        LIMIT 1
    )
WHERE EXISTS (
    SELECT 1 
    FROM recording_events re 
    WHERE re.recording_id = recordings.recording_id
);
