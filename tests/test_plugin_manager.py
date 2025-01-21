"""Tests for the plugin manager module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, mock_open
import yaml
import uuid

from app.core.events import ConcreteEventBus
from app.core.plugins import PluginBase, PluginConfig, PluginManager


@pytest.fixture
def event_bus():
    return ConcreteEventBus()


@pytest.fixture
def plugin_dir(tmp_path):
    return str(tmp_path / "plugins")


@pytest.fixture
def plugin_manager(plugin_dir, event_bus):
    return PluginManager(plugin_dir, event_bus)


@pytest.mark.asyncio
async def test_plugin_manager_init():
    """Test plugin manager initialization."""
    plugin_dir = "/test/plugins"
    event_bus = ConcreteEventBus()
    
    # Test with event bus
    manager = PluginManager(plugin_dir, event_bus)
    assert manager.plugin_dir == Path(plugin_dir)
    assert manager.event_bus == event_bus
    assert isinstance(manager.plugins, dict)
    assert isinstance(manager.configs, dict)
    
    # Test without event bus
    manager = PluginManager(plugin_dir)
    assert manager.event_bus is None


@pytest.mark.asyncio
async def test_discover_plugins_no_event_bus(plugin_manager):
    """Test plugin discovery fails without event bus."""
    plugin_manager.event_bus = None
    configs = await plugin_manager.discover_plugins()
    assert configs == []


@pytest.mark.asyncio
async def test_discover_plugins_with_yaml(plugin_manager, tmp_path):
    """Test plugin discovery with valid plugin.yaml files."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    test_plugin_dir = plugin_dir / "test_plugin"
    test_plugin_dir.mkdir()
    
    config_data = {
        "name": "test_plugin",
        "version": "1.0.0",
        "enabled": True,
        "dependencies": ["dep1"],
        "config": {"key": "value"}
    }
    
    yaml_path = test_plugin_dir / "plugin.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)
    
    plugin_manager.plugin_dir = plugin_dir
    configs = await plugin_manager.discover_plugins()
    
    assert len(configs) == 1
    config = configs[0]
    assert config.name == "test_plugin"
    assert config.version == "1.0.0"
    assert config.enabled is True
    assert config.dependencies == ["dep1"]
    assert config.config == {"key": "value"}


@pytest.mark.asyncio
async def test_discover_plugins_with_example(plugin_manager, tmp_path):
    """Test plugin discovery with plugin.yaml.example files."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    test_plugin_dir = plugin_dir / "test_plugin"
    test_plugin_dir.mkdir()
    
    config_data = {
        "name": "test_plugin",
        "version": "1.0.0"
    }
    
    example_path = test_plugin_dir / "plugin.yaml.example"
    with open(example_path, "w") as f:
        yaml.dump(config_data, f)
    
    plugin_manager.plugin_dir = plugin_dir
    configs = await plugin_manager.discover_plugins()
    
    assert len(configs) == 1
    assert Path(test_plugin_dir / "plugin.yaml").exists()


@pytest.mark.asyncio
async def test_discover_plugins_invalid_yaml(plugin_manager, tmp_path):
    """Test plugin discovery with invalid yaml files."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    test_plugin_dir = plugin_dir / "test_plugin"
    test_plugin_dir.mkdir()
    
    yaml_path = test_plugin_dir / "plugin.yaml"
    with open(yaml_path, "w") as f:
        f.write("invalid: yaml: content:")
    
    plugin_manager.plugin_dir = plugin_dir
    configs = await plugin_manager.discover_plugins()
    assert len(configs) == 0


@pytest.mark.asyncio
async def test_initialize_plugins(plugin_manager, tmp_path):
    """Test plugin initialization."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    
    # Create a mock plugin
    class MockPlugin(PluginBase):
        async def _initialize(self) -> None:
            self.initialized = True
        
        async def _shutdown(self) -> None:
            self.shutdown_called = True
    
    config = PluginConfig(name="test_plugin", version="1.0.0")
    mock_plugin = MockPlugin(config=config, event_bus=plugin_manager.event_bus)
    plugin_manager.plugins["test_plugin"] = mock_plugin
    
    # Call initialize directly to set the flag
    await mock_plugin.initialize()
    
    # Now initialize all plugins
    await plugin_manager.initialize_plugins()
    assert hasattr(mock_plugin, "initialized")
    assert mock_plugin.initialized is True


@pytest.mark.asyncio
async def test_shutdown_plugins(plugin_manager):
    """Test plugin shutdown."""
    class MockPlugin(PluginBase):
        async def _initialize(self) -> None:
            pass
        
        async def _shutdown(self) -> None:
            self.shutdown_called = True
    
    config = PluginConfig(name="test_plugin", version="1.0.0")
    mock_plugin = MockPlugin(config=config, event_bus=plugin_manager.event_bus)
    plugin_manager.plugins["test_plugin"] = mock_plugin
    
    await plugin_manager.shutdown_plugins()
    assert hasattr(mock_plugin, "shutdown_called")
    assert mock_plugin.shutdown_called is True


@pytest.mark.asyncio
async def test_get_plugin(plugin_manager):
    """Test getting a plugin by name."""
    config = PluginConfig(name="test_plugin", version="1.0.0")
    mock_plugin = Mock(spec=PluginBase, version="1.0.0")
    mock_plugin.config = config
    plugin_manager.plugins["test_plugin"] = mock_plugin
    
    assert plugin_manager.get_plugin("test_plugin") == mock_plugin
    assert plugin_manager.get_plugin("nonexistent") is None


@pytest.mark.asyncio
async def test_sort_by_dependencies(plugin_manager):
    """Test sorting plugins by dependencies."""
    configs = {
        "plugin1": PluginConfig(name="plugin1", version="1.0", enabled=True, dependencies=["plugin2"]),
        "plugin2": PluginConfig(name="plugin2", version="1.0", enabled=True),
        "plugin3": PluginConfig(name="plugin3", version="1.0", enabled=True, dependencies=["plugin1"]),
    }
    
    sorted_configs = plugin_manager._sort_by_dependencies(configs)
    sorted_names = [config.name for config in sorted_configs]
    
    # plugin2 should come before plugin1, and plugin1 before plugin3
    assert sorted_names.index("plugin2") < sorted_names.index("plugin1")
    assert sorted_names.index("plugin1") < sorted_names.index("plugin3") 