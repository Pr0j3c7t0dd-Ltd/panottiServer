import importlib
from pathlib import Path
import sys
import traceback
import uuid
import yaml
from typing import Optional

from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PluginManager:
    """Manages plugin lifecycle and event routing"""

    def __init__(self, plugin_dir: str, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin manager with the plugin directory"""
        self._req_id = str(uuid.uuid4())  # Add request ID for tracing
        self.plugin_dir = Path(plugin_dir)
        self.plugins: dict[str, PluginBase] = {}
        self.configs: dict[str, PluginConfig] = {}
        if event_bus is None:
            logger.warning(
                "No event bus provided to plugin manager",
                extra={"req_id": self._req_id}
            )
        self.event_bus = event_bus
        logger.info(
            "Plugin manager initialized",
            extra={
                "req_id": self._req_id,
                "plugin_dir": str(plugin_dir)
            }
        )

    async def discover_plugins(self) -> list[PluginConfig]:
        """Discover and load plugin configurations"""
        if self.event_bus is None:
            logger.error(
                "Cannot discover plugins without event bus",
                extra={"req_id": self._req_id}
            )
            return []

        logger.info(
            "Starting plugin discovery",
            extra={
                "req_id": self._req_id,
                "plugin_dir": str(self.plugin_dir)
            }
        )

        logger.debug(
            "Looking for plugins",
            extra={
                "req_id": self._req_id,
                "plugin_dir": str(self.plugin_dir),
                "plugin_dir_exists": self.plugin_dir.exists(),
                "plugin_dir_is_dir": self.plugin_dir.is_dir() if self.plugin_dir.exists() else None,
                "plugin_dir_contents": [str(p) for p in self.plugin_dir.iterdir()] if self.plugin_dir.exists() and self.plugin_dir.is_dir() else [],
            },
        )

        # Search for both plugin.yaml and plugin.yaml.example files
        yaml_files = list(self.plugin_dir.glob("*/plugin.yaml")) + list(self.plugin_dir.glob("*/plugin.yaml.example"))
        logger.debug(
            "Found yaml files",
            extra={
                "req_id": self._req_id,
                "yaml_files": [str(f) for f in yaml_files],
            },
        )
        config_files = []
        
        # For each directory, prefer plugin.yaml over plugin.yaml.example
        plugin_dirs = {f.parent for f in yaml_files}
        logger.debug(
            "Found plugin directories",
            extra={
                "req_id": self._req_id,
                "plugin_dirs": [str(d) for d in plugin_dirs],
            },
        )
        for plugin_dir in plugin_dirs:
            logger.debug(
                "Checking plugin directory",
                extra={
                    "req_id": self._req_id,
                    "plugin_dir": str(plugin_dir),
                    "is_dir": plugin_dir.is_dir(),
                    "contents": [str(p) for p in plugin_dir.iterdir()] if plugin_dir.is_dir() else [],
                },
            )
            
            yaml_path = plugin_dir / "plugin.yaml"
            example_path = plugin_dir / "plugin.yaml.example"
            
            logger.debug(
                "Checking plugin directory",
                extra={
                    "req_id": self._req_id,
                    "plugin_dir": str(plugin_dir),
                    "yaml_exists": (plugin_dir / "plugin.yaml").exists(),
                    "example_exists": (plugin_dir / "plugin.yaml.example").exists(),
                },
            )
            
            if yaml_path.exists():
                logger.debug(
                    "Found plugin config file",
                    extra={
                        "req_id": self._req_id,
                        "config_file": str(yaml_path),
                        "plugin_dir": str(plugin_dir),
                    },
                )
                config_files.append(yaml_path)
            elif example_path.exists():
                logger.debug(
                    "No plugin config file found, using example",
                    extra={
                        "req_id": self._req_id,
                        "plugin_dir": str(plugin_dir),
                        "expected_config": str(yaml_path),
                        "example_path": str(example_path),
                    },
                )
                # Copy example to yaml if it doesn't exist
                logger.debug(
                    "Creating plugin.yaml from example",
                    extra={
                        "req_id": self._req_id,
                        "plugin_dir": str(plugin_dir),
                        "example_path": str(example_path),
                        "yaml_path": str(yaml_path)
                    }
                )
                with open(example_path, "r") as src, open(yaml_path, "w") as dst:
                    dst.write(src.read())
                config_files.append(yaml_path)
            else:
                logger.debug(
                    "No plugin config file found",
                    extra={
                        "req_id": self._req_id,
                        "plugin_dir": str(plugin_dir),
                        "expected_config": str(yaml_path),
                    },
                )
        
        logger.debug(
            "Found plugin config files",
            extra={
                "req_id": self._req_id,
                "config_files": [str(f) for f in config_files],
                "plugin_dir": str(self.plugin_dir),
                "search_pattern": "*/plugin.yaml",
                "plugin_dir_exists": self.plugin_dir.exists(),
                "plugin_dir_contents": [str(p) for p in self.plugin_dir.iterdir()] if self.plugin_dir.exists() else [],
                "example_plugin_dir": str(self.plugin_dir / "example"),
                "example_plugin_dir_exists": (self.plugin_dir / "example").exists(),
                "example_plugin_dir_contents": [str(p) for p in (self.plugin_dir / "example").iterdir()] if (self.plugin_dir / "example").exists() else [],
                "example_plugin_yaml": str(self.plugin_dir / "example" / "plugin.yaml"),
                "example_plugin_yaml_exists": (self.plugin_dir / "example" / "plugin.yaml").exists(),
            },
        )

        # Load plugin configurations
        configs = []
        for config_file in config_files:
            logger.debug(
                "Processing plugin config",
                extra={
                    "req_id": self._req_id,
                    "config_file": str(config_file),
                    "plugin_dir": str(config_file.parent),
                },
            )
            try:
                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f)
                    logger.debug(
                        "Loaded plugin config data",
                        extra={
                            "req_id": self._req_id,
                            "config_file": str(config_file),
                            "config_data": config_data,
                        },
                    )
                    config = PluginConfig(
                        name=config_data["name"],
                        version=config_data["version"],
                        enabled=config_data.get("enabled", True),
                        dependencies=config_data.get("dependencies", []),
                        config=config_data.get("config", {}),
                    )
                    logger.debug(
                        "Validated plugin config",
                        extra={
                            "req_id": self._req_id,
                            "config_file": str(config_file),
                            "config": config.dict(),
                        },
                    )
                    self.configs[config.name] = config  # Store in self.configs
                    configs.append(config)

            except Exception as e:
                logger.error(
                    "Failed to load plugin config",
                    extra={
                        "req_id": self._req_id,
                        "config_file": str(config_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                continue

        return configs

    async def initialize_plugins(self) -> None:
        """Initialize all discovered plugins."""
        logger.info(
            "Starting plugin initialization",
            extra={
                "req_id": self._req_id,
                "plugin_count": len(self.plugins)
            }
        )

        # Sort plugins by dependencies
        sorted_configs = self._sort_by_dependencies(self.configs)
        logger.debug(
            "Sorted plugins by dependencies",
            extra={
                "req_id": self._req_id,
                "plugin_order": [config.name for config in sorted_configs],
                "plugin_configs": {
                    config.name: {
                        "enabled": config.enabled,
                        "dependencies": config.dependencies,
                    }
                    for config in sorted_configs
                },
            },
        )
        
        # Initialize plugins in dependency order
        for config in sorted_configs:
            if not config.enabled:
                logger.info(
                    "Plugin is disabled, skipping",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": config.name,
                        "plugin_version": config.version,
                    },
                )
                continue

            logger.debug(
                "Initializing plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": config.name,
                    "plugin_version": config.version,
                    "plugin_enabled": config.enabled,
                    "plugin_dependencies": config.dependencies,
                    "plugin_dir": str(self.plugin_dir / config.name),
                    "plugin_dir_exists": (self.plugin_dir / config.name).exists(),
                    "plugin_py_exists": (self.plugin_dir / config.name / "plugin.py").exists(),
                },
            )

            try:
                # Import plugin module
                plugin_dir = self.plugin_dir / config.name
                module_path = plugin_dir / "plugin.py"
                plugin_module_name = f"app.plugins.{config.name}"

                # Add plugin parent directory to Python path if not already there
                plugin_parent_dir = str(self.plugin_dir.parent.parent)
                if plugin_parent_dir not in sys.path:
                    logger.debug(
                        "Adding plugin parent directory to Python path",
                        extra={
                            "req_id": self._req_id,
                            "plugin_name": config.name,
                            "plugin_parent_dir": plugin_parent_dir,
                            "sys_path_before": sys.path,
                        },
                    )
                    sys.path.insert(0, plugin_parent_dir)

                logger.debug(
                    "Importing plugin module",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": config.name,
                        "module_name": plugin_module_name,
                        "module_path": str(module_path),
                        "module_exists": module_path.exists(),
                        "sys_path": sys.path,
                    },
                )

                plugin_module = importlib.import_module(plugin_module_name)

                # Find plugin class
                plugin_class = None
                for attr_name in dir(plugin_module):
                    attr = getattr(plugin_module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PluginBase)
                        and attr is not PluginBase
                    ):
                        logger.debug(
                            "Found plugin class candidate",
                            extra={
                                "req_id": self._req_id,
                                "plugin_name": config.name,
                                "class_name": attr_name,
                                "module_name": plugin_module.__name__,
                                "plugin_dir": str(plugin_dir),
                                "plugin_parent_dir": str(plugin_dir.parent),
                                "module_file": str(plugin_module.__file__),
                            },
                        )
                        plugin_class = attr
                        break

                if plugin_class is None:
                    logger.error(
                        "No plugin class found",
                        extra={
                            "req_id": self._req_id,
                            "plugin_name": config.name,
                            "module_name": plugin_module.__name__,
                            "available_attrs": dir(plugin_module),
                        },
                    )
                    continue

                # Create and initialize plugin instance
                plugin = plugin_class(config, event_bus=self.event_bus)
                await plugin.initialize()
                self.plugins[config.name] = plugin

                logger.info(
                    "Plugin initialized successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": config.name,
                        "plugin_version": config.version,
                        "plugin_enabled": config.enabled,
                    },
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize plugin",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": config.name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                continue

    def _sort_by_dependencies(self, configs: dict[str, PluginConfig]) -> list[PluginConfig]:
        """Sort plugins by dependencies."""
        sorted_configs = []
        visited = set()

        def visit(config: PluginConfig):
            if config.name in visited:
                return
            visited.add(config.name)
            for dep in config.dependencies:
                if dep in configs:
                    visit(configs[dep])
            sorted_configs.append(config)

        for config in configs.values():
            visit(config)

        return sorted_configs

    async def shutdown_plugins(self) -> None:
        """Shutdown all plugins in reverse dependency order"""
        logger.info(
            "Shutting down plugins",
            extra={
                "req_id": self._req_id,
                "plugin_count": len(self.plugins)
            }
        )

        # Shutdown in reverse dependency order
        for name, plugin in reversed(list(self.plugins.items())):
            try:
                logger.debug(
                    "Shutting down plugin",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": name
                    }
                )
                await plugin.shutdown()
                logger.info(
                    "Plugin shutdown successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": name
                    }
                )
            except Exception as e:
                logger.error(
                    "Error shutting down plugin",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": name,
                        "error": str(e)
                    }
                )

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name"""
        plugin = self.plugins.get(name)
        if plugin:
            logger.debug(
                "Retrieved plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": name,
                    "plugin_version": plugin.version
                }
            )
        else:
            logger.debug(
                "Plugin not found",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": name
                }
            )
        return plugin

    async def _load_plugin(self, plugin_path: Path) -> None:
        """Load a plugin from the given path."""
        try:
            # Import the plugin module
            module_name = plugin_path.stem
            spec = importlib.util.spec_from_file_location(
                f"app.plugins.{module_name}",
                plugin_path / "plugin.py"
            )
            if not spec or not spec.loader:
                raise ImportError(f"Could not load plugin spec for {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get plugin class from the module
            plugin_class = None
            for item in dir(module):
                if item.endswith("Plugin"):
                    plugin_class = getattr(module, item)
                    break

            if not plugin_class:
                raise ValueError(f"No plugin class found in {module_name}")

            # Initialize plugin
            plugin = plugin_class(self.configs, self.event_bus)
            await plugin._initialize()
            
            self.plugins[module_name] = plugin
            logger.info(
                "Plugin loaded successfully",
                extra={
                    "req_id": self._req_id
                }
            )

        except Exception as e:
            logger.error(
                f"Failed to load plugin module {plugin_path.stem}",
                extra={
                    "req_id": self._req_id
                },
                exc_info=True
            )
            raise
