-- Drop existing plugin_tasks table and recreate with composite primary key
DROP TABLE IF EXISTS plugin_tasks;

CREATE TABLE plugin_tasks (
    recording_id TEXT NOT NULL,
    plugin_name TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'processing', 'completed', 'failed'
    input_paths TEXT,  -- Comma-separated list of input file paths
    output_paths TEXT,  -- Comma-separated list of output file paths
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (recording_id, plugin_name),
    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
);

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_plugin_tasks_recording_id ON plugin_tasks(recording_id);
CREATE INDEX IF NOT EXISTS idx_plugin_tasks_plugin_name ON plugin_tasks(plugin_name); 