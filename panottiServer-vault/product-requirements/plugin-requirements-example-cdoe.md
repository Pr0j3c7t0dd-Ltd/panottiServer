# app/plugins/events/persistence.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import sqlite3
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

class StoredEvent(BaseModel):
    """Model for stored events"""
    id: int
    name: str
    payload: Dict[str, Any]
    context: Dict[str, Any]
    priority: str
    timestamp: datetime
    processed: bool = False
    error_count: int = 0
    last_error: Optional[str] = None

class EventStore:
    """Persistent storage for events"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize event store database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    context TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_processed 
                ON events(processed, timestamp)
            """)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection asynchronously"""
        def _get_conn():
            return sqlite3.connect(self.db_path)
        
        conn = await asyncio.get_event_loop().run_in_executor(None, _get_conn)
        try:
            yield conn
        finally:
            await asyncio.get_event_loop().run_in_executor(None, conn.close)
    
    async def store_event(self, event: Event) -> int:
        """Store an event"""
        async with self.get_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(
                    """
                    INSERT INTO events (name, payload, context, priority, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.name,
                        json.dumps(event.payload),
                        json.dumps(event.context.dict()),
                        event.priority.name,
                        event.context.timestamp.isoformat()
                    )
                )
            )
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
            return cursor.lastrowid
    
    async def mark_processed(self, event_id: int, success: bool = True, error: Optional[str] = None):
        """Mark an event as processed"""
        async with self.get_connection() as conn:
            if success:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: conn.execute(
                        "UPDATE events SET processed = TRUE WHERE id = ?",
                        (event_id,)
                    )
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: conn.execute(
                        """
                        UPDATE events 
                        SET error_count = error_count + 1,
                            last_error = ?
                        WHERE id = ?
                        """,
                        (error, event_id)
                    )
                )
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)

    async def get_unprocessed_events(
        self,
        batch_size: int = 100,
        max_retries: int = 3
    ) -> List[StoredEvent]:
        """Get unprocessed events"""
        async with self.get_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(
                    """
                    SELECT * FROM events 
                    WHERE processed = FALSE 
                    AND error_count < ? 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                    """,
                    (max_retries, batch_size)
                )
            )
            rows = await asyncio.get_event_loop().run_in_executor(None, cursor.fetchall)
            
            return [
                StoredEvent(
                    id=row[0],
                    name=row[1],
                    payload=json.loads(row[2]),
                    context=json.loads(row[3]),
                    priority=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    processed=bool(row[6]),
                    error_count=row[7],
                    last_error=row[8]
                )
                for row in rows
            ]

# app/plugins/events/replay.py
class EventReplay:
    """Handles event replay functionality"""
    def __init__(self, event_bus: EventBus, event_store: EventStore):
        self.event_bus = event_bus
        self.event_store = event_store
        self._running = False
        self._replay_task: Optional[asyncio.Task] = None
    
    async def start_replay(self, 
                          batch_size: int = 100,
                          interval: float = 1.0,
                          max_retries: int = 3):
        """Start event replay process"""
        if self._running:
            return
        
        self._running = True
        self._replay_task = asyncio.create_task(
            self._replay_loop(batch_size, interval, max_retries)
        )
    
    async def stop_replay(self):
        """Stop event replay process"""
        self._running = False
        if self._replay_task:
            await self._replay_task
            self._replay_task = None
    
    async def _replay_loop(self, batch_size: int, interval: float, max_retries: int):
        """Main replay loop"""
        while self._running:
            try:
                events = await self.event_store.get_unprocessed_events(
                    batch_size,
                    max_retries
                )
                
                if not events:
                    await asyncio.sleep(interval)
                    continue
                
                for stored_event in events:
                    try:
                        event = Event(
                            name=stored_event.name,
                            payload=stored_event.payload,
                            context=EventContext(**stored_event.context),
                            priority=EventPriority[stored_event.priority]
                        )
                        
                        await self.event_bus.emit(event)
                        await self.event_store.mark_processed(stored_event.id)
                        
                    except Exception as e:
                        await self.event_store.mark_processed(
                            stored_event.id,
                            success=False,
                            error=str(e)
                        )
            
            except Exception as e:
                print(f"Error in replay loop: {e}")
                await asyncio.sleep(interval)

# Enhanced EventBus with persistence and replay
class EnhancedEventBus(EventBus):
    """EventBus with persistence and replay capabilities"""
    def __init__(self, event_store: EventStore):
        super().__init__()
        self.event_store = event_store
        self.replay_manager = EventReplay(self, event_store)
    
    async def emit(self, event: Event) -> None:
        """Emit event with persistence"""
        # Store event first
        event_id = await self.event_store.store_event(event)
        
        try:
            # Process event
            await super().publish(event)
            # Mark as processed
            await self.event_store.mark_processed(event_id)
        except Exception as e:
            # Mark as failed
            await self.event_store.mark_processed(
                event_id,
                success=False,
                error=str(e)
            )
            raise

# Example usage with persistence and replay
class RecordingPlugin(PluginBase):
    async def initialize(self, app: FastAPI, db_conn: Connection):
        await super().initialize(app, db_conn)
        
        @self.router.post("/replay-events")
        async def start_replay():
            """Start replaying unprocessed events"""
            await self.event_bus.replay_manager.start_replay()
            return {"status": "replay_started"}
        
        @self.router.post("/stop-replay")
        async def stop_replay():
            """Stop replaying events"""
            await self.event_bus.replay_manager.stop_replay()
            return {"status": "replay_stopped"}

# Example implementation
"""
# Initialize components
event_store = EventStore("events.db")
event_bus = EnhancedEventBus(event_store)

# Create and initialize plugin
recording_plugin = RecordingPlugin(event_bus)
await recording_plugin.initialize(app, db_conn)

# Start event replay
await event_bus.replay_manager.start_replay(
    batch_size=50,
    interval=2.0,
    max_retries=3
)

# Emit an event
await recording_plugin.emit_event(
    "recording_started",
    {"recording_id": "rec_123"},
    EventPriority.HIGH
)

# Stop replay when done
await event_bus.replay_manager.stop_replay()

# Query unprocessed events
unprocessed = await event_store.get_unprocessed_events()
"""