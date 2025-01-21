from unittest.mock import AsyncMock

import pytest

from tests.conftest import _TestPluginImpl


async def test_initialize_already_initialized(test_plugin_impl):
    """Test initialize when plugin is already initialized"""
    test_plugin_impl._initialized = True
    await test_plugin_impl.initialize()
    # Should log but not call _initialize
    assert test_plugin_impl.is_initialized


async def test_initialize_success(test_plugin_impl):
    """Test successful initialization"""
    assert not test_plugin_impl.is_initialized
    await test_plugin_impl.initialize()
    assert test_plugin_impl.is_initialized


async def test_initialize_failure(test_plugin_impl):
    """Test initialization failure"""

    async def failing_initialize():
        raise ValueError("Test error")

    test_plugin_impl._initialize = failing_initialize
    with pytest.raises(ValueError, match="Test error"):
        await test_plugin_impl.initialize()
    assert not test_plugin_impl.is_initialized


async def test_shutdown_success(test_plugin_impl):
    """Test successful shutdown"""
    test_plugin_impl._initialized = True
    await test_plugin_impl.shutdown()
    assert not test_plugin_impl.is_initialized


async def test_shutdown_failure(test_plugin_impl):
    """Test shutdown failure"""

    async def failing_shutdown():
        raise ValueError("Shutdown error")

    test_plugin_impl._shutdown = failing_shutdown
    test_plugin_impl._initialized = True
    with pytest.raises(ValueError, match="Shutdown error"):
        await test_plugin_impl.shutdown()


def test_get_config_with_default(test_plugin_impl):
    """Test get_config with default value when key doesn't exist"""
    value = test_plugin_impl.get_config("nonexistent", default="default")
    assert value == "default"


def test_get_config_with_none_config(plugin_config, event_bus):
    """Test get_config when config is None"""
    plugin_config.config = None
    plugin = _TestPluginImpl(plugin_config, event_bus)
    value = plugin.get_config("test_key", default="default")
    assert value == "default"


async def test_subscribe_with_event_bus(test_plugin_impl, event_bus):
    """Test subscribe with event bus"""
    callback = AsyncMock()
    await test_plugin_impl.subscribe("test_event", callback)
    event_bus.subscribe.assert_awaited_once_with("test_event", callback)


async def test_subscribe_without_event_bus(plugin_config):
    """Test subscribe without event bus"""
    plugin = _TestPluginImpl(plugin_config, None)
    callback = AsyncMock()
    # Should not raise an error
    await plugin.subscribe("test_event", callback)


async def test_unsubscribe_with_event_bus(test_plugin_impl, event_bus):
    """Test unsubscribe with event bus"""
    callback = AsyncMock()
    await test_plugin_impl.unsubscribe("test_event", callback)
    event_bus.unsubscribe.assert_awaited_once_with("test_event", callback)


async def test_unsubscribe_without_event_bus(plugin_config):
    """Test unsubscribe without event bus"""
    plugin = _TestPluginImpl(plugin_config, None)
    callback = AsyncMock()
    # Should not raise an error
    await plugin.unsubscribe("test_event", callback)


async def test_publish_with_event_bus(test_plugin_impl, event_bus):
    """Test publish with event bus"""
    event = {"test": "event"}
    await test_plugin_impl.publish(event)
    event_bus.publish.assert_awaited_once_with(event)


async def test_publish_without_event_bus(plugin_config):
    """Test publish without event bus"""
    plugin = _TestPluginImpl(plugin_config, None)
    event = {"test": "event"}
    # Should not raise an error
    await plugin.publish(event)


async def test_emit_event_with_event_bus(test_plugin_impl, event_bus):
    """Test emit_event with event bus"""
    await test_plugin_impl.emit_event("test_event", {"data": "test"}, "correlation-123")
    expected_event = {
        "name": "test_event",
        "data": {"data": "test"},
        "correlation_id": "correlation-123",
        "source_plugin": "test_plugin",
    }
    event_bus.publish.assert_awaited_once_with(expected_event)


async def test_emit_event_without_event_bus(plugin_config):
    """Test emit_event without event bus"""
    plugin = _TestPluginImpl(plugin_config, None)
    # Should log warning but not raise error
    await plugin.emit_event("test_event", {"data": "test"})
