from unittest.mock import patch

import pytest
import yaml

from app.core.events import ConcreteEventBus
from app.core.plugins.interface import PluginBase, PluginConfig
from app.core.plugins.manager import PluginManager


@pytest.fixture
def event_bus():
    return ConcreteEventBus()


@pytest.fixture
def plugin_manager(event_bus, tmp_plugin_dir):
    return PluginManager(plugin_dir=tmp_plugin_dir, event_bus=event_bus)


@pytest.fixture
def tmp_plugin_dir(tmp_path):
    """Create a temporary plugin directory with proper structure."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir(parents=True)
    return plugin_dir


@pytest.fixture
def TestPluginClass():
    """Fixture providing a base test plugin class."""

    class TestPlugin(PluginBase):
        """Base test plugin class with tracking of lifecycle methods."""

        def __init__(self, config, event_bus=None):
            super().__init__(config, event_bus)
            self.initialize_called = False
            self.shutdown_called = False

        async def _initialize(self):
            self.initialize_called = True

        async def _shutdown(self):
            self.shutdown_called = True

        async def initialize(self):
            await super().initialize()

        async def shutdown(self):
            await super().shutdown()

    return TestPlugin


async def test_plugin_dependency_sorting(plugin_manager):
    """Test that plugins are sorted correctly based on dependencies."""
    # Create test configs
    plugin1_config = PluginConfig(
        name="plugin1", version="1.0.0", dependencies=["plugin2"]
    )
    plugin2_config = PluginConfig(name="plugin2", version="1.0.0", dependencies=[])

    plugin_manager.configs = {"plugin1": plugin1_config, "plugin2": plugin2_config}

    sorted_plugins = plugin_manager._sort_by_dependencies(plugin_manager.configs)
    assert len(sorted_plugins) == 2
    assert sorted_plugins[0].name == "plugin2"  # plugin2 should come first
    assert sorted_plugins[1].name == "plugin1"  # plugin1 depends on plugin2


async def test_plugin_initialization_order(plugin_manager, TestPluginClass):
    """Test that plugins are initialized in the correct order based on dependencies."""

    class ConcreteOrderedTestPlugin(TestPluginClass):
        pass

    # Create test configs
    plugin1_config = PluginConfig(
        name="plugin1", version="1.0.0", dependencies=["plugin2"]
    )
    plugin2_config = PluginConfig(name="plugin2", version="1.0.0", dependencies=[])

    plugin_manager.configs = {"plugin1": plugin1_config, "plugin2": plugin2_config}

    # Create a mock module class
    class MockModule:
        def __init__(self, name):
            self.ConcreteOrderedTestPlugin = ConcreteOrderedTestPlugin
            self.__file__ = f"{name}/plugin.py"
            self.__name__ = f"app.plugins.{name}"

    with patch(
        "importlib.import_module",
        side_effect=lambda name: MockModule(name.split(".")[-1]),
    ) as mock_import:
        await plugin_manager.initialize_plugins()

        # Verify import order
        assert mock_import.call_count == 2
        calls = mock_import.call_args_list
        assert (
            calls[0][0][0] == "app.plugins.plugin2"
        )  # plugin2 should be imported first
        assert (
            calls[1][0][0] == "app.plugins.plugin1"
        )  # plugin1 should be imported second


async def test_plugin_lifecycle(plugin_manager, TestPluginClass):
    """Test plugin initialization and shutdown lifecycle."""
    config = PluginConfig(name="test", version="1.0.0")
    plugin_manager.configs["test"] = config

    # Create a concrete plugin class
    class ConcreteTestPlugin(TestPluginClass):
        pass

    # Create a mock module
    class MockModule:
        def __init__(self):
            self.ConcreteTestPlugin = ConcreteTestPlugin
            self.__file__ = "test_plugin.py"
            self.__name__ = "app.plugins.test"

    with patch("importlib.import_module", return_value=MockModule()):
        # Initialize plugins
        await plugin_manager.initialize_plugins()

        # Verify plugin was initialized
        assert "test" in plugin_manager.plugins
        test_plugin = plugin_manager.plugins["test"]
        assert test_plugin.initialize_called

        # Shutdown plugins
        await plugin_manager.shutdown_plugins()

        # Verify plugin was shutdown
        assert test_plugin.shutdown_called


async def test_plugin_config_loading(plugin_manager, tmp_plugin_dir):
    """Test loading plugin configurations from files."""
    # Create test plugin directories
    plugin1_dir = tmp_plugin_dir / "plugin1"
    plugin2_dir = tmp_plugin_dir / "plugin2"
    plugin1_dir.mkdir()
    plugin2_dir.mkdir()

    # Create plugin config files
    plugin1_config = {
        "name": "plugin1",
        "version": "1.0.0",
        "enabled": True,
        "dependencies": ["plugin2"],
    }
    plugin2_config = {
        "name": "plugin2",
        "version": "1.0.0",
        "enabled": True,
        "dependencies": [],
    }

    with open(plugin1_dir / "plugin.yaml", "w") as f:
        yaml.dump(plugin1_config, f)
    with open(plugin2_dir / "plugin.yaml", "w") as f:
        yaml.dump(plugin2_config, f)

    # Set plugin directory in manager
    plugin_manager.plugin_dir = tmp_plugin_dir

    # Discover plugins
    await plugin_manager.discover_plugins()

    # Verify configs were loaded correctly
    assert "plugin1" in plugin_manager.configs
    assert "plugin2" in plugin_manager.configs
    assert plugin_manager.configs["plugin1"].dependencies == ["plugin2"]
    assert not plugin_manager.configs["plugin2"].dependencies
