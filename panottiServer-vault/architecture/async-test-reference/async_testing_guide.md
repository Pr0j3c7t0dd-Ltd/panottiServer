# Async Testing Best Practices

## Overview
This guide outlines best practices for writing async tests in our codebase, particularly focusing on avoiding common pitfalls with asyncio and pytest.

## Key Components

### 1. pytest Configuration
In `pytest.ini`, use these recommended settings:
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
log_cli = true  # Helpful for debugging async tests
log_cli_level = INFO
```

Do NOT use deprecated settings like `asyncio_fixture_timeout`.

### 2. Fixture Best Practices

#### Modern Approach (Preferred)
```python
@pytest_asyncio.fixture
async def event_bus():
    """Create and start an event bus instance."""
    # Mock long-running operations
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = None
        
        # Setup
        bus = EventBus()
        await bus.start()
        
        # Cancel any background tasks
        if bus._cleanup_events_task:
            bus._cleanup_events_task.cancel()
            try:
                await bus._cleanup_events_task
            except asyncio.CancelledError:
                pass
            bus._cleanup_events_task = None
        
        yield bus
        
        # Cleanup
        await bus.stop()
        await bus.shutdown()
```

#### Global Test Fixtures
In `conftest.py`, set up global async mocks:
```python
@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Mock asyncio.sleep globally."""
    async def immediate_sleep(*args, **kwargs):
        return None
    
    with patch("asyncio.sleep", side_effect=immediate_sleep):
        yield

@pytest.fixture(autouse=True)
def mock_threadpool():
    """Mock ThreadPoolExecutor to prevent shutdown issues."""
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance._shutdown = False
        
        # Create a real Future object for submit
        def submit_side_effect(*args, **kwargs):
            future = asyncio.Future()
            future.set_result(None)
            return future
            
        mock_executor_instance.submit.side_effect = submit_side_effect
        yield mock_executor_instance
```

### 3. Test Class Structure
For complex async tests, use `IsolatedAsyncioTestCase`:
```python
class TestComplexAsyncFeature(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup async resources
        self.mock_resource = AsyncMock()
        
    async def asyncTearDown(self):
        # Cleanup async resources
        await self.resource.cleanup()
        
    async def test_async_operation(self):
        # Test implementation
        pass
```

### 4. Handling Background Tasks
```python
@pytest.fixture
async def cleanup_tasks():
    yield
    # Get all tasks except current task
    tasks = [t for t in asyncio.all_tasks() 
             if t is not asyncio.current_task()]
    if tasks:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        # Wait with timeout for tasks to cancel
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=1.0
            )
        except asyncio.TimeoutError:
            pass  # Some tasks may be stuck, continue cleanup
```

### 5. Mocking Async Database Operations
```python
@pytest.fixture(autouse=True)
def mock_db():
    """Mock database fixture"""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    mock_db.initialize = AsyncMock()
    mock_db._init_db = MagicMock()
    mock_db._run_migrations = MagicMock()
    mock_db.get_connection = MagicMock()

    with patch("app.models.database.DatabaseManager") as mock_manager:
        mock_manager._instance = None
        mock_manager._lock = asyncio.Lock()
        mock_manager.get_instance = AsyncMock(return_value=mock_db)
        yield mock_db
```

### 6. Testing Event-Based Systems
```python
@pytest.mark.asyncio
async def test_event_handling(event_bus):
    # Setup test event handler
    async def test_handler(event):
        # Handler logic
        pass
    
    # Subscribe to events
    await event_bus.subscribe("test_event", test_handler)
    
    # Publish event
    await event_bus.publish({
        "event": "test_event", 
        "data": "test"
    })
    
    # Allow event processing
    await asyncio.sleep(0)
    
    # Assert results
    assert some_condition
```

### 7. Testing Long-Running Tasks
```python
@pytest.mark.asyncio
async def test_long_running_task():
    # Add run_once parameter for cleanup tasks
    await service._cleanup_old_events(run_once=True)
    
    # Or mock the long-running task
    with patch.object(service, '_cleanup_old_events', 
                     new_callable=AsyncMock) as mock_cleanup:
        await service.start()
        mock_cleanup.assert_called_once()
```

#### Handling Infinite Loops in Error Cases
When testing error conditions in long-running tasks, ensure the task can exit even in error cases:

```python
# In the implementation:
async def _cleanup_old_events(self, run_once: bool = False) -> None:
    while True:
        try:
            if not run_once:
                await asyncio.sleep(3600)
            # ... cleanup logic ...
            if run_once:
                break
        except Exception as e:
            logger.error("Error in cleanup")
            if run_once:  # Important: Also break in error case when run_once=True
                break

# In the test:
@pytest.mark.asyncio
async def test_cleanup_error_handling():
    # Create a mock that raises an exception
    mock_lock = AsyncMock()
    mock_lock.__aenter__.side_effect = [Exception("Lock error")]
    service._lock = mock_lock

    # The task should complete even with error when run_once=True
    cleanup_task = asyncio.create_task(
        service._cleanup_old_events(run_once=True)
    )
    
    try:
        await asyncio.wait_for(cleanup_task, timeout=1.0)
    except (asyncio.TimeoutError, Exception):
        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):
            pass
```

This pattern prevents infinite error logging loops in tests while still allowing proper error handling in production code.

### 8. Testing Long Delays and Cleanup Tasks
```python
class TestEventBus(IsolatedAsyncioTestCase):
    async def test_cleanup_old_events(self):
        bus = EventBus()
        
        # Simulate old processed events
        now = datetime.now(UTC)
        old_event_id = "old_event"
        recent_event_id = "recent_event"
        
        bus._processed_events = {
            old_event_id: now - timedelta(hours=2),  # Older than 1 hour
            recent_event_id: now - timedelta(minutes=30),  # Recent
        }
        
        # Method 1: Use run_once parameter
        await bus._cleanup_old_events(run_once=True)
        
        # Method 2: Mock sleep and test cleanup logic
        with patch("asyncio.sleep", return_value=asyncio.Future()) as mock_sleep:
            mock_sleep.return_value.set_result(None)
            await bus._cleanup_old_events()
            
        # Method 3: For background tasks, use controlled cancellation
        bus._cleanup_events_task = asyncio.create_task(bus._cleanup_old_events())
        await asyncio.sleep(0)  # Let task start
        bus._cleanup_events_task.cancel()
        try:
            await bus._cleanup_events_task
        except asyncio.CancelledError:
            pass
            
        # Assert results
        assert old_event_id not in bus._processed_events
        assert recent_event_id in bus._processed_events
```

#### Best Practices for Long-Running Tests

1. **Use `run_once` Parameters**
   - Add a `run_once=True` parameter to long-running tasks
   - Makes tests deterministic and fast
   ```python
   async def _cleanup_old_events(self, run_once=False):
       while True:
           # Cleanup logic
           if run_once:
               break
           await asyncio.sleep(3600)
   ```

2. **Mock Time-Based Operations**
   - Mock `asyncio.sleep` for immediate return
   - Use `AsyncMock` for other async time operations
   ```python
   with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
       mock_sleep.return_value = None
       await long_running_task()
   ```

3. **Handle Background Tasks**
   - Use task cancellation for cleanup
   - Add timeouts to prevent hanging
   - Always handle CancelledError
   ```python
   if task:
       task.cancel()
       try:
           await asyncio.wait_for(task, timeout=1.0)
       except (asyncio.CancelledError, asyncio.TimeoutError):
           pass
   ```

4. **Test Task States**
   - Verify task creation and cancellation
   - Check task status transitions
   - Test error handling
   ```python
   task = asyncio.create_task(long_running_operation())
   assert not task.done()
   task.cancel()
   await asyncio.sleep(0)
   assert task.cancelled()
   ```

## Common Mistakes to Avoid

### 1. Deprecated Client Initialization
❌ **Don't use deprecated parameters:**
```python
client = AsyncClient(app=app, base_url="http://test")  # Deprecated
```

✅ **Do use explicit transport:**
```python
from httpx import AsyncClient, ASGITransport
client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
```

### 2. Improper Task Cleanup
❌ **Don't leave tasks uncancelled:**
```python
task = asyncio.create_task(long_running())
await some_operation()  # Task might still be running
```

✅ **Do ensure proper task cleanup:**
```python
task = asyncio.create_task(long_running())
try:
    await some_operation()
finally:
    if not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
```

### 3. Missing Mock Sleep
❌ **Don't test long-running tasks directly:**
```python
async def test_cleanup():
    await service._cleanup_old_events()  # Will run forever
```

✅ **Do mock sleep or add run_once parameter:**
```python
@pytest.mark.asyncio
async def test_cleanup():
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = None
        await service._cleanup_old_events()
```

### 4. Improper Event Loop Management
❌ **Don't create your own event loops:**
```python
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
```

✅ **Do use pytest-asyncio's loop management:**
```python
@pytest.mark.asyncio
async def test_async_operation():
    # pytest-asyncio manages the loop
    await async_operation()
```

### 5. Incorrect AsyncMock Usage
❌ **Don't mix MagicMock with async code:**
```python
mock_obj = MagicMock()
mock_obj.async_method  # Won't work properly with await
```

✅ **Do use AsyncMock for async methods:**
```python
mock_obj = AsyncMock()
await mock_obj.async_method()  # Works correctly
```

### 6. Missing Async Context Manager Handling
❌ **Don't forget to handle async context managers:**
```python
client = AsyncClient()  # Resource leak
await client.get("/test")
```

✅ **Do properly handle async context managers:**
```python
async with AsyncClient() as client:
    await client.get("/test")
```

### 7. Improper Background Task Testing
❌ **Don't test background tasks without cancellation handling:**
```python
task = asyncio.create_task(background_job())
assert task.done()  # Unreliable
```

✅ **Do implement proper task testing:**
```python
task = asyncio.create_task(background_job())
try:
    await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
except asyncio.TimeoutError:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

### 8. Missing Error Handling in Cleanup
❌ **Don't assume cleanup will always succeed:**
```python
async def asyncTearDown(self):
    await self.resource.cleanup()  # Might fail
```

✅ **Do handle cleanup errors gracefully:**
```python
async def asyncTearDown(self):
    try:
        await self.resource.cleanup()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
        # Consider if you need to raise or can continue
```

### 9. Incorrect Test Isolation
❌ **Don't share state between tests:**
```python
class TestClass:
    resource = None  # Shared state!
    
    async def test_1(self):
        self.resource = await setup()
```

✅ **Do ensure proper test isolation:**
```python
class TestClass:
    async def asyncSetUp(self):
        self.resource = await setup()
        
    async def asyncTearDown(self):
        await self.resource.cleanup()
```

### 10. Missing Timeout Handling
❌ **Don't leave operations without timeouts:**
```python
await long_running_operation()  # Might hang indefinitely
```

✅ **Do use timeouts for all operations:**
```python
try:
    await asyncio.wait_for(long_running_operation(), timeout=5.0)
except asyncio.TimeoutError:
    logger.error("Operation timed out")
    raise
```

Remember:
- Always use `pytest.mark.asyncio` for async tests
- Clean up resources in `asyncTearDown` or `finally` blocks
- Mock long-running operations
- Handle task cancellation properly
- Use timeouts for all async operations
- Properly initialize and clean up async context managers
- Use the appropriate mock types (AsyncMock vs MagicMock)
- Follow explicit over implicit patterns
- Test both success and error cases
- Ensure proper test isolation

## Testing Tools

1. **pytest-asyncio**: Primary tool for async testing
2. **AsyncMock**: For mocking coroutines
3. **pytest-timeout**: For preventing hanging tests
4. **pytest-cov**: For coverage reporting

## References

- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html) 