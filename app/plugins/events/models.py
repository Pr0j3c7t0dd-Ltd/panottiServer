"""Event system models."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class EventContext:
    """Context for an event."""
    metadata: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
