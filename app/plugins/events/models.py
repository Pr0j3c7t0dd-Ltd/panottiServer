"""Event system models."""
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class EventContext:
    """Context for an event."""

    metadata: dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
