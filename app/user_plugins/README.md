# User Plugins Directory

This directory is where you should add your custom plugins for PanottiServer. Each plugin should be in its own directory to keep functionality isolated.

## Directory Structure

```
user_plugins/
├── my_plugin/              # Your plugin directory
│   ├── __init__.py        # Plugin initialization
│   ├── plugin.py          # Main plugin code
│   └── requirements.txt    # Plugin-specific dependencies
└── another_plugin/         # Another plugin directory
    ├── __init__.py
    ├── plugin.py
    └── requirements.txt
```

## Creating a New Plugin

1. Create a new directory for your plugin under `user_plugins/`
2. Create an `__init__.py` file to make it a Python package
3. Create your plugin implementation in `plugin.py`
4. Add any plugin-specific dependencies to `requirements.txt`

## Example Plugin Structure

Here's an example of a minimal plugin:

```python
# plugin.py
import logging
import pluggy
from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)
hookimpl = pluggy.HookimplMarker("panotti")

class MyPlugin:
    @hookimpl
    async def on_startup(self, app: FastAPI) -> None:
        logger.info("MyPlugin: Starting up")

    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        logger.info(f"MyPlugin: Recording {recording_id} is starting")
```

## Available Hooks

Your plugin can implement any of these hooks:

- `on_startup(app: FastAPI) -> None`
- `on_shutdown(app: FastAPI) -> None`
- `before_request(request: Request) -> None`
- `after_request(response: Response) -> None`
- `before_recording_start(recording_id: str) -> None`
- `after_recording_end(recording_id: str) -> None`

## Threading and Async Operations

PanottiServer uses FastAPI and supports both synchronous and asynchronous operations in plugins. Here are examples of how to handle different scenarios:

### Async Operations

Most hooks in PanottiServer are asynchronous. Here's how to implement them:

```python
import asyncio
import logging
import pluggy
from fastapi import FastAPI

logger = logging.getLogger(__name__)
hookimpl = pluggy.HookimplMarker("panotti")

class AsyncPlugin:
    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        # Perform async operations
        await self.async_setup()
        
    async def async_setup(self):
        """Example async operation."""
        await asyncio.sleep(1)  # Simulating async work
        logger.info("Async setup completed")

    @hookimpl
    async def after_recording_end(self, recording_id: str) -> None:
        # Run multiple async operations concurrently
        tasks = [
            self.process_audio(recording_id),
            self.save_metadata(recording_id)
        ]
        await asyncio.gather(*tasks)
    
    async def process_audio(self, recording_id: str):
        """Example async audio processing."""
        await asyncio.sleep(2)  # Simulating processing
        logger.info(f"Processed audio for {recording_id}")
    
    async def save_metadata(self, recording_id: str):
        """Example async metadata saving."""
        await asyncio.sleep(1)  # Simulating DB operation
        logger.info(f"Saved metadata for {recording_id}")
```

### Long-Running Operations

For long-running operations, you should run them in a separate thread to avoid blocking the event loop:

```python
import asyncio
import threading
import time
import logging
import pluggy
from fastapi import FastAPI

logger = logging.getLogger(__name__)
hookimpl = pluggy.HookimplMarker("panotti")

class LongRunningPlugin:
    def __init__(self):
        self.processing_thread = None
    
    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        # Start a background thread for long-running work
        self.processing_thread = threading.Thread(
            target=self.long_running_task,
            args=(recording_id,)
        )
        self.processing_thread.start()
    
    def long_running_task(self, recording_id: str):
        """Example of a long-running task."""
        logger.info(f"Starting long process for {recording_id}")
        time.sleep(10)  # Simulating long work
        logger.info(f"Completed long process for {recording_id}")
    
    @hookimpl
    async def after_recording_end(self, recording_id: str) -> None:
        # Wait for background thread to complete if needed
        if self.processing_thread and self.processing_thread.is_alive():
            # Convert thread.join() to async operation
            await asyncio.to_thread(self.processing_thread.join)
```

### Combining Async and Thread-based Operations

Here's an example combining both approaches:

```python
import asyncio
import threading
import time
import logging
import pluggy
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
hookimpl = pluggy.HookimplMarker("panotti")

class HybridPlugin:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        # Run CPU-intensive task in thread pool
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.cpu_intensive_task,
            recording_id
        )
        
        # Run async tasks directly
        await self.async_task(recording_id)
    
    def cpu_intensive_task(self, recording_id: str):
        """CPU-intensive operation runs in thread pool."""
        time.sleep(2)  # Simulating CPU work
        logger.info(f"CPU task completed for {recording_id}")
    
    async def async_task(self, recording_id: str):
        """I/O-bound operation runs asynchronously."""
        await asyncio.sleep(1)  # Simulating I/O
        logger.info(f"Async task completed for {recording_id}")
    
    @hookimpl
    async def after_recording_end(self, recording_id: str) -> None:
        # Clean up resources
        self.executor.shutdown(wait=False)
```

### Best Practices

1. **Use async for I/O-bound operations**
   - Network requests
   - Database operations
   - File I/O

2. **Use threads for CPU-bound operations**
   - Audio processing
   - Image processing
   - Complex calculations

3. **Resource Management**
   - Always clean up threads and executors
   - Use context managers when possible
   - Handle exceptions in both async and threaded code

4. **Avoid Blocking the Event Loop**
   - Don't use `time.sleep()` in async functions
   - Use `await asyncio.sleep()` instead
   - Move CPU-intensive work to threads

5. **Thread Safety**
   - Use proper synchronization for shared resources
   - Consider using `asyncio.Lock()` for async operations
   - Use `threading.Lock()` for threaded operations

Remember that all hook implementations should be async, even if they're running synchronous code internally. Use `asyncio.to_thread()` or thread pools to wrap synchronous operations when needed.

## Database Access

Plugins can access the application's SQLite database for persistent storage. Here's how to use it:

```python
from app.models.database import get_db

class MyPlugin:
    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        # Access the database using the thread-safe connection manager
        with get_db().get_connection() as conn:
            cursor = conn.cursor()
            
            # Example: Store plugin-specific data
            cursor.execute(
                'INSERT INTO events (type, timestamp, data) VALUES (?, ?, ?)',
                ('plugin_event', datetime.now().isoformat(), 
                 json.dumps({'plugin': 'MyPlugin', 'recording_id': recording_id}))
            )
            
            # Example: Query existing events
            cursor.execute(
                'SELECT * FROM events WHERE type = ?',
                ('Recording Started',)
            )
            events = cursor.fetchall()
```

The database connection is thread-safe and managed automatically. Each plugin gets its own connection, and connections are properly cleaned up when they're no longer needed.

Key points about database usage:
- Always use the connection manager with a `with` statement
- Connections are automatically committed on success and rolled back on error
- The database is SQLite, so complex queries should be kept to a minimum
- Each plugin should handle its own database errors appropriately

## Installing Dependencies

If your plugin has dependencies:

1. List them in your plugin's `requirements.txt`
2. Install them using: `pip install -r user_plugins/your_plugin/requirements.txt`

## Registering Your Plugin

Your plugin will be automatically discovered and loaded if it's properly placed in this directory.
