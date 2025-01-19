"""Tests for core plugin system interfaces."""

import pytest
from unittest.mock import Mock

from app.core.plugins import PluginBase
from app.core.events import EventBus


class ConcretePlugin(PluginBase):
    """Concrete implementation of PluginBase for testing."""
    
    async def _initialize(self) -> None:
        """Test implementation of initialize."""
        pass

    async def _shutdown(self) -> None:
        """Test implementation of shutdown."""
        pass


@pytest.fixture
def event_bus():
    """Fixture providing a mock event bus."""
    return Mock(spec=EventBus)


@pytest.fixture
def plugin_config():
    """Fixture providing test plugin config."""
    return {"test_key": "test_value"}


@pytest.fixture
def concrete_plugin(plugin_config, event_bus):
    """Fixture providing a concrete plugin instance."""
    return ConcretePlugin(config=plugin_config, event_bus=event_bus)


async def test_plugin_initialization_with_event_bus(concrete_plugin, plugin_config, event_bus):
    """Test plugin initialization with event bus."""
    assert concrete_plugin.config == plugin_config
    assert concrete_plugin.event_bus == event_bus
    assert concrete_plugin.logger is None


async def test_plugin_initialization_without_event_bus(plugin_config):
    """Test plugin initialization without event bus."""
    plugin = ConcretePlugin(config=plugin_config)
    assert plugin.config == plugin_config
    assert plugin.event_bus is None
    assert plugin.logger is None


async def test_plugin_initialize_calls_internal_initialize(concrete_plugin):
    """Test that initialize() calls _initialize()."""
    await concrete_plugin.initialize()
    # No assertion needed as we're just verifying it doesn't raise an exception


async def test_plugin_shutdown_calls_internal_shutdown(concrete_plugin):
    """Test that shutdown() calls _shutdown()."""
    await concrete_plugin.shutdown()
    # No assertion needed as we're just verifying it doesn't raise an exception 