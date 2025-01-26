from unittest.mock import patch

import pytest
import yaml
import asyncio

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


async def test_discover_plugins_yaml_error(plugin_manager, tmp_plugin_dir):
    """Test error handling when loading invalid YAML config."""
    plugin_dir = tmp_plugin_dir / "invalid_plugin"
    plugin_dir.mkdir()
    
    # Create invalid YAML file
    with open(plugin_dir / "plugin.yaml", "w") as f:
        f.write("invalid: yaml: content: {")
    
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_initialize_plugins_import_error(plugin_manager):
    """Test error handling when plugin module import fails."""
    config = PluginConfig(name="nonexistent", version="1.0.0")
    plugin_manager.configs["nonexistent"] = config
    
    await plugin_manager.initialize_plugins()
    assert "nonexistent" not in plugin_manager.plugins


async def test_circular_dependency_sorting(plugin_manager):
    """Test sorting of plugins with circular dependencies."""
    plugin1_config = PluginConfig(
        name="plugin1", version="1.0.0", dependencies=["plugin2"]
    )
    plugin2_config = PluginConfig(
        name="plugin2", version="1.0.0", dependencies=["plugin1"]
    )
    
    plugin_manager.configs = {
        "plugin1": plugin1_config,
        "plugin2": plugin2_config,
    }
    
    # The current implementation will handle circular dependencies by visiting each node once
    sorted_plugins = plugin_manager._sort_by_dependencies(plugin_manager.configs)
    assert len(sorted_plugins) == 2
    # The order will depend on which plugin is visited first, but both should be included
    assert all(p.name in ["plugin1", "plugin2"] for p in sorted_plugins)


async def test_plugin_initialization_error(plugin_manager, TestPluginClass):
    """Test error handling during plugin initialization."""
    class FailingPlugin(TestPluginClass):
        async def _initialize(self):
            raise RuntimeError("Initialization failed")

    class MockModule:
        def __init__(self):
            self.ConcreteTestPlugin = FailingPlugin
            self.__file__ = "failing_plugin.py"
            self.__name__ = "app.plugins.failing"

    config = PluginConfig(name="failing", version="1.0.0")
    plugin_manager.configs["failing"] = config

    with patch("importlib.import_module", return_value=MockModule()):
        await plugin_manager.initialize_plugins()
        assert "failing" not in plugin_manager.plugins


async def test_plugin_shutdown_error(plugin_manager, TestPluginClass):
    """Test error handling during plugin shutdown."""
    class FailingShutdownPlugin(TestPluginClass):
        async def _shutdown(self):
            raise RuntimeError("Shutdown failed")

    class MockModule:
        def __init__(self):
            self.ConcreteTestPlugin = FailingShutdownPlugin
            self.__file__ = "failing_plugin.py"
            self.__name__ = "app.plugins.failing"

    config = PluginConfig(name="failing", version="1.0.0")
    plugin_manager.configs["failing"] = config

    with patch("importlib.import_module", return_value=MockModule()):
        await plugin_manager.initialize_plugins()
        assert "failing" in plugin_manager.plugins
        await plugin_manager.shutdown_plugins()


def test_get_plugin_nonexistent(plugin_manager):
    """Test getting a non-existent plugin."""
    assert plugin_manager.get_plugin("nonexistent") is None


async def test_missing_plugin_class(plugin_manager):
    """Test error handling when plugin class is missing."""
    class MockModule:
        def __init__(self):
            self.__file__ = "missing_plugin.py"
            self.__name__ = "app.plugins.missing"
            # No ConcreteTestPlugin class defined

    config = PluginConfig(name="missing", version="1.0.0")
    plugin_manager.configs["missing"] = config

    with patch("importlib.import_module", return_value=MockModule()):
        await plugin_manager.initialize_plugins()
        assert "missing" not in plugin_manager.plugins


async def test_plugin_manager_no_event_bus():
    """Test plugin manager initialization without event bus."""
    plugin_manager = PluginManager(plugin_dir="plugins")
    assert plugin_manager.event_bus is None
    
    # Test discover_plugins without event bus
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_discover_plugins_no_plugin_dir(tmp_path):
    """Test plugin discovery with non-existent plugin directory."""
    nonexistent_dir = tmp_path / "nonexistent"
    plugin_manager = PluginManager(plugin_dir=str(nonexistent_dir))
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_discover_plugins_invalid_plugin_dir(tmp_path):
    """Test plugin discovery with invalid plugin directory (file instead of dir)."""
    invalid_dir = tmp_path / "invalid"
    invalid_dir.touch()  # Create a file instead of a directory
    plugin_manager = PluginManager(plugin_dir=str(invalid_dir))
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_plugin_initialization_missing_module(plugin_manager):
    """Test plugin initialization with missing module."""
    config = PluginConfig(name="missing_module", version="1.0.0")
    plugin_manager.configs["missing_module"] = config
    
    with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'missing_module'")):
        await plugin_manager.initialize_plugins()
        assert "missing_module" not in plugin_manager.plugins


async def test_plugin_shutdown_with_error_and_continue(plugin_manager, TestPluginClass):
    """Test plugin shutdown continues after error."""
    class FailingShutdownPlugin(TestPluginClass):
        async def _shutdown(self):
            raise RuntimeError("Shutdown failed")

    class SuccessfulShutdownPlugin(TestPluginClass):
        pass

    class MockModuleFailing:
        def __init__(self):
            self.ConcreteTestPlugin = FailingShutdownPlugin
            self.__file__ = "failing_plugin.py"
            self.__name__ = "app.plugins.failing"

    class MockModuleSuccessful:
        def __init__(self):
            self.ConcreteTestPlugin = SuccessfulShutdownPlugin
            self.__file__ = "successful_plugin.py"
            self.__name__ = "app.plugins.successful"

    failing_config = PluginConfig(name="failing", version="1.0.0")
    successful_config = PluginConfig(name="successful", version="1.0.0")
    plugin_manager.configs = {
        "failing": failing_config,
        "successful": successful_config
    }

    with patch("importlib.import_module", side_effect=[MockModuleFailing(), MockModuleSuccessful()]):
        await plugin_manager.initialize_plugins()
        assert "failing" in plugin_manager.plugins
        assert "successful" in plugin_manager.plugins
        
        await plugin_manager.shutdown_plugins()
        # Plugins remain in the manager even if shutdown fails
        assert "failing" in plugin_manager.plugins
        assert "successful" in plugin_manager.plugins


async def test_discover_plugins_with_empty_config(plugin_manager, tmp_plugin_dir):
    """Test plugin discovery with empty config file."""
    plugin_dir = tmp_plugin_dir / "empty_plugin"
    plugin_dir.mkdir()
    
    # Create empty YAML file
    with open(plugin_dir / "plugin.yaml", "w") as f:
        f.write("")
    
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_discover_plugins_with_missing_required_fields(plugin_manager, tmp_plugin_dir):
    """Test plugin discovery with missing required fields."""
    plugin_dir = tmp_plugin_dir / "invalid_plugin"
    plugin_dir.mkdir()
    
    # Create YAML file missing required fields
    with open(plugin_dir / "plugin.yaml", "w") as f:
        f.write("version: 1.0.0")  # Missing name field
    
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_plugin_initialization_with_invalid_module(plugin_manager):
    """Test plugin initialization with invalid module."""
    config = PluginConfig(name="invalid", version="1.0.0")
    plugin_manager.configs["invalid"] = config
    
    class InvalidModule:
        def __init__(self):
            self.__file__ = "invalid_plugin.py"
            self.__name__ = "app.plugins.invalid"
            # No plugin class and invalid attribute access should raise AttributeError
    
    with patch("importlib.import_module", return_value=InvalidModule()):
        await plugin_manager.initialize_plugins()
        assert "invalid" not in plugin_manager.plugins


async def test_plugin_shutdown_cleanup(plugin_manager, TestPluginClass):
    """Test plugin shutdown with cleanup."""
    class CleanupPlugin(TestPluginClass):
        async def _shutdown(self):
            await super()._shutdown()
            # Simulate some cleanup work that might fail
            raise RuntimeError("Cleanup failed")

    class MockModule:
        def __init__(self):
            self.ConcreteTestPlugin = CleanupPlugin
            self.__file__ = "cleanup_plugin.py"
            self.__name__ = "app.plugins.cleanup"

    config = PluginConfig(name="cleanup", version="1.0.0")
    plugin_manager.configs["cleanup"] = config

    with patch("importlib.import_module", return_value=MockModule()):
        await plugin_manager.initialize_plugins()
        assert "cleanup" in plugin_manager.plugins
        
        # Test shutdown with cleanup error
        await plugin_manager.shutdown_plugins()
        assert "cleanup" in plugin_manager.plugins  # Plugin remains in manager after failed shutdown


async def test_discover_plugins_with_invalid_example(plugin_manager, tmp_plugin_dir):
    """Test plugin discovery with invalid example file."""
    plugin_dir = tmp_plugin_dir / "example_plugin"
    plugin_dir.mkdir()
    
    # Create invalid example file
    with open(plugin_dir / "plugin.yaml.example", "w") as f:
        f.write("invalid: yaml: content")
    
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


async def test_plugin_initialization_with_module_error(plugin_manager):
    """Test plugin initialization with module that raises error."""
    config = PluginConfig(name="error_module", version="1.0.0")
    plugin_manager.configs["error_module"] = config
    
    class ErrorModule:
        def __init__(self):
            raise ImportError("Module import failed")
    
    with patch("importlib.import_module", side_effect=ErrorModule):
        await plugin_manager.initialize_plugins()
        assert "error_module" not in plugin_manager.plugins


async def test_plugin_shutdown_with_task_error(plugin_manager, TestPluginClass):
    """Test plugin shutdown with task error."""
    class TaskErrorPlugin(TestPluginClass):
        async def _shutdown(self):
            # Simulate a task error during shutdown
            await asyncio.sleep(0)  # Allow other tasks to run
            raise asyncio.CancelledError("Task cancelled")

    class MockModule:
        def __init__(self):
            self.ConcreteTestPlugin = TaskErrorPlugin
            self.__file__ = "task_error_plugin.py"
            self.__name__ = "app.plugins.task_error"

    config = PluginConfig(name="task_error", version="1.0.0")
    plugin_manager.configs["task_error"] = config

    with patch("importlib.import_module", return_value=MockModule()):
        await plugin_manager.initialize_plugins()
        assert "task_error" in plugin_manager.plugins
        
        # Test shutdown with task error
        try:
            await plugin_manager.shutdown_plugins()
        except asyncio.CancelledError:
            pass  # Expected error
        assert "task_error" in plugin_manager.plugins


def test_get_plugin_with_error():
    """Test get_plugin with error."""
    plugin_manager = PluginManager(plugin_dir="plugins")
    assert plugin_manager.get_plugin(None) is None  # Test with invalid plugin name
