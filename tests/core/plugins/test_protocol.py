"""Tests for core plugin protocol."""

import pytest
from unittest.mock import Mock, AsyncMock

from app.core.events import EventBus
from app.core.plugins.protocol import PluginBase


def create_test_plugin(config: dict = None, event_bus: EventBus = None) -> PluginBase:
    """Create a test plugin implementation."""
    class TestPluginImplementation(PluginBase):
        """Test implementation of PluginBase for testing."""
        
        async def _initialize(self) -> None:
            """Test initialize implementation."""
            pass
            
        async def _shutdown(self) -> None:
            """Test shutdown implementation."""
            pass
    
    return TestPluginImplementation(config or {}, event_bus)


@pytest.mark.asyncio
class TestPluginBase:
    """Test suite for PluginBase class."""

    async def test_init(self):
        """Test plugin initialization with config and event bus."""
        config = {"test": "config"}
        event_bus = Mock(spec=EventBus)
        
        plugin = create_test_plugin(config, event_bus)
        
        assert plugin.config == config
        assert plugin.event_bus == event_bus
        assert plugin.logger is None

    async def test_initialize(self):
        """Test initialize method calls _initialize."""
        plugin = create_test_plugin()
        
        # Create spy on _initialize using AsyncMock
        plugin._initialize = AsyncMock()  # type: ignore
        
        await plugin.initialize()
        
        plugin._initialize.assert_called_once()

    async def test_shutdown(self):
        """Test shutdown method calls _shutdown."""
        plugin = create_test_plugin()
        
        # Create spy on _shutdown using AsyncMock
        plugin._shutdown = AsyncMock()  # type: ignore
        
        await plugin.shutdown()
        
        plugin._shutdown.assert_called_once() 