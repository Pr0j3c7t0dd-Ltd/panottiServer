# Event System Architecture

## Overview

The event system is a core component of PanottiServer, providing a robust, type-safe, and asynchronous event processing pipeline. It enables loose coupling between components while maintaining system consistency and reliability.

## Components

### 1. Event Bus
```python
class EventBus:
    """Central event dispatcher with async support."""
    
    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers."""
        
    async def subscribe(self, handler: EventHandler) -> None:
        """Register event handler."""
        
    async def unsubscribe(self, handler: EventHandler) -> None:
        """Remove event handler."""
```

### 2. Event Models
```python
class Event(BaseModel):
    """Base event model with required fields."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any]
    correlation_id: str | None = None
    source_plugin: str | None = None
```

### 3. Event Context
```python
class EventContext(BaseModel):
    """Context information for event processing."""
    correlation_id: str
    source_plugin: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

## Event Flow

1. **Event Creation**
   - Event instantiation
   - Context attachment
   - Validation
   - Correlation ID generation

2. **Event Publishing**
   - Event bus distribution
   - Handler notification
   - Error handling
   - Event persistence

3. **Event Processing**
   - Handler execution
   - State updates
   - Response events
   - Error recovery

4. **Event Completion**
   - Result recording
   - Cleanup tasks
   - State finalization
   - Metrics update

## Best Practices

### 1. Event Design
```python
# Good Practice
class RecordingCompleted(Event):
    """Event emitted when recording processing is complete."""
    name: str = "recording.completed"
    data: RecordingCompletedData
    
class RecordingCompletedData(BaseModel):
    """Structured data for recording completion."""
    recording_id: str
    duration: float
    file_path: str
    metadata: dict[str, Any]
```

### 2. Error Handling
```python
async def handle_event(self, event: Event, context: EventContext) -> None:
    try:
        await self.process_event(event)
    except TemporaryError:
        # Retry logic
        await self.retry_event(event, context)
    except PermanentError:
        # Dead letter queue
        await self.move_to_dlq(event, context)
    except Exception as e:
        # Unexpected error
        self.logger.error("Unexpected error", error=str(e), event=event)
        raise
```

### 3. Event Correlation
```python
def create_child_event(self, parent_event: Event, name: str, data: dict) -> Event:
    """Create a new event that maintains correlation with parent."""
    return Event(
        name=name,
        data=data,
        correlation_id=parent_event.correlation_id,
        source_plugin=self.name
    )
```

## Event Patterns

### 1. Event Sourcing
- Event as source of truth
- State reconstruction
- Event replay capability
- Version management

### 2. Event Validation
- Schema validation
- Business rule validation
- Context validation
- Data integrity checks

### 3. Event Persistence
- Event store implementation
- Replay capability
- Event versioning
- Data retention

## Performance Considerations

### 1. Batch Processing
```python
async def process_events(self, events: list[Event]) -> None:
    """Process multiple events efficiently."""
    chunks = [events[i:i + 100] for i in range(0, len(events), 100)]
    for chunk in chunks:
        await asyncio.gather(*[self.process_event(e) for e in chunk])
```

### 2. Backpressure
```python
class EventBus:
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=1000)
        
    async def publish(self, event: Event) -> None:
        try:
            await asyncio.wait_for(
                self._queue.put(event),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # Handle backpressure
            pass
```

### 3. Caching
```python
class EventCache:
    def __init__(self):
        self._cache: dict[str, Event] = {}
        self._lock = asyncio.Lock()
        
    async def get_or_create(self, key: str, factory: Callable) -> Event:
        async with self._lock:
            if key not in self._cache:
                self._cache[key] = await factory()
            return self._cache[key]
```

## Monitoring and Debugging

### 1. Event Logging
```python
class EventLogger:
    async def log_event(self, event: Event, context: EventContext) -> None:
        self.logger.info(
            "Event processed",
            event_id=event.id,
            event_name=event.name,
            correlation_id=context.correlation_id,
            duration=time.time() - context.timestamp.timestamp()
        )
```

### 2. Metrics Collection
```python
class EventMetrics:
    async def record_metrics(self, event: Event, duration: float) -> None:
        await self.metrics.histogram(
            "event_processing_duration",
            duration,
            tags={"event_type": event.name}
        )
```

### 3. Debugging Tools
```python
class EventDebugger:
    async def capture_event(self, event: Event) -> None:
        """Capture event for debugging."""
        await self.debug_store.save(
            event_id=event.id,
            event_data=event.dict(),
            timestamp=datetime.now(UTC)
        )
```

## Security Considerations

### 1. Event Validation
- Input sanitization
- Schema validation
- Access control
- Rate limiting

### 2. Sensitive Data
- Data encryption
- PII handling
- Audit logging
- Data retention

### 3. Error Handling
- Safe error messages
- Secure logging
- Resource cleanup
- Failure isolation
