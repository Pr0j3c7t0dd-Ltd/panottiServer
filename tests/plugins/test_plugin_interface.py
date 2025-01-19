import pytest

from app.core.events import ConcreteEventBus as EventBus
from app.plugins.base import PluginBase, PluginConfig


class BasePluginTest:
    """Base class that all plugin tests must inherit from"""

    @pytest.fixture
    def event_bus(self):
        """Mock event bus fixture"""
        return EventBus()

    @pytest.fixture
    def plugin_config(self):
        """Base plugin config fixture - override in specific tests"""
        return PluginConfig(
            name="test_plugin",
            version="1.0.0",
            enabled=True,
            dependencies=[],
            config={},
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Plugin instance fixture - override in specific tests"""
        raise NotImplementedError("Plugin fixture must be implemented by subclass")

    async def test_plugin_interface(self, plugin):
        """Verify plugin implements required interface"""
        assert isinstance(plugin, PluginBase)
        assert hasattr(plugin, "_initialize")
        assert hasattr(plugin, "_shutdown")
        assert hasattr(plugin, "name")
        assert isinstance(plugin.name, str)
        assert plugin.version == plugin.config.version
        assert not plugin.is_initialized

    async def test_plugin_initialization(self, plugin):
        """Verify plugin initializes correctly"""
        assert not plugin.is_initialized
        await plugin.initialize()
        assert plugin.is_initialized

    async def test_plugin_shutdown(self, plugin):
        """Verify plugin shuts down correctly"""
        await plugin.initialize()
        assert plugin.is_initialized
        await plugin.shutdown()
        assert not plugin.is_initialized

    async def test_get_config(self, plugin):
        """Test config retrieval"""
        test_key = "test_key"
        test_value = "test_value"
        plugin.config.config = {test_key: test_value}
        assert plugin.get_config(test_key) == test_value
        assert plugin.get_config("nonexistent", "default") == "default"

    async def test_event_bus_methods(self, plugin, event_bus):
        """Test event bus integration methods"""
        test_event = {"name": "test_event", "data": {}}

        # Test subscribe/unsubscribe
        callback_called = False

        async def test_callback(event):
            nonlocal callback_called
            callback_called = True

        await plugin.subscribe("test_event", test_callback)
        await plugin.publish(test_event)
        assert callback_called

        # Test unsubscribe
        callback_called = False
        await plugin.unsubscribe("test_event", test_callback)
        await plugin.publish(test_event)
        assert not callback_called

    async def test_emit_event(self, plugin):
        """Test event emission"""
        test_name = "test_event"
        test_data = {"key": "value"}
        await plugin.emit_event(test_name, test_data)
        # Note: actual event emission verification should be done in specific plugin tests
