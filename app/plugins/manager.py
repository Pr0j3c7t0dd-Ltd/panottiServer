import importlib
from pathlib import Path
import sys
import traceback
import yaml

from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class PluginManager:
    """Manages plugin lifecycle and event routing"""

    def __init__(self, plugin_dir: str, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin manager with the plugin directory"""
        self.plugin_dir = Path(plugin_dir)
        self.plugins: dict[str, PluginBase] = {}
        self.configs: dict[str, PluginConfig] = {}
        if event_bus is None:
            logger.warning("No event bus provided to plugin manager")
        self.event_bus = event_bus
        logger.info("Plugin manager initialized", extra={"plugin_dir": str(plugin_dir)})

    async def discover_plugins(self) -> list[PluginConfig]:
        """Discover and load plugin configurations"""
        if self.event_bus is None:
            logger.error("Cannot discover plugins without event bus")
            return []

        logger.info(
            "Starting plugin discovery", extra={"plugin_dir": str(self.plugin_dir)}
        )

        config_files = list(self.plugin_dir.glob("*/plugin.yaml"))
        logger.debug(
            "Found plugin config files",
            extra={
                "config_files": [str(f) for f in config_files],
                "plugin_dir": str(self.plugin_dir),
                "search_pattern": "*/plugin.yaml"
            },
        )

        configs: list[PluginConfig] = []
        for config_file in config_files:
            try:
                plugin_dir = config_file.parent
                logger.debug(
                    "Processing plugin config",
                    extra={
                        "config_file": str(config_file),
                        "plugin_dir": str(plugin_dir),
                        "exists": config_file.exists(),
                        "is_file": config_file.is_file(),
                        "parent_exists": plugin_dir.exists(),
                    },
                )

                if not config_file.exists():
                    logger.error(
                        "Plugin config file does not exist",
                        extra={
                            "config_file": str(config_file),
                            "plugin_dir": str(plugin_dir),
                        },
                    )
                    continue

                with open(config_file) as f:
                    try:
                        config_data = yaml.safe_load(f)
                        logger.debug(
                            "Loaded plugin config data",
                            extra={
                                "config_file": str(config_file),
                                "config_data": config_data,
                                "yaml_version": yaml.__version__,
                            },
                        )
                    except yaml.YAMLError as e:
                        logger.error(
                            "Failed to parse plugin config YAML",
                            extra={
                                "config_file": str(config_file),
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                        continue

                try:
                    config = PluginConfig(**config_data)
                    logger.debug(
                        "Validated plugin config",
                        extra={
                            "config_file": str(config_file),
                            "plugin_name": config.name,
                            "plugin_version": config.version,
                            "plugin_enabled": config.enabled,
                        },
                    )
                except Exception as e:
                    logger.error(
                        "Failed to validate plugin config",
                        extra={
                            "config_file": str(config_file),
                            "config_data": config_data,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    continue

                self.configs[config.name] = config
                configs.append(config)

                if not config.enabled:
                    logger.info(
                        "Plugin is disabled, skipping",
                        extra={
                            "plugin_name": config.name,
                            "plugin_version": config.version,
                        },
                    )
                    continue

                # Load plugin module
                module_path = plugin_dir / "plugin.py"
                if not module_path.exists():
                    logger.error(
                        "Plugin module not found",
                        extra={
                            "plugin_name": config.name,
                            "module_path": str(module_path),
                        },
                    )
                    continue

                try:
                    # Import the plugin package directly
                    plugin_package = f"app.plugins.{config.name}"
                    logger.debug(
                        "Attempting to import plugin package",
                        extra={
                            "plugin_package": plugin_package,
                            "plugin_name": config.name,
                            "plugin_dir": str(plugin_dir),
                            "module_path": str(module_path),
                            "sys_modules": list(sys.modules.keys()),
                        },
                    )
                    module = importlib.import_module(plugin_package)
                    logger.debug(
                        "Successfully imported module",
                        extra={
                            "module_name": module.__name__,
                            "module_file": getattr(module, "__file__", "unknown"),
                            "module_attrs": dir(module),
                        },
                    )

                    # Get plugin class from the module
                    plugin_class = None
                    module_attrs = dir(module)
                    plugin_candidates = [attr for attr in module_attrs if attr.endswith("Plugin")]
                    logger.debug(
                        "Searching for plugin class",
                        extra={
                            "plugin_name": config.name,
                            "module_attrs": module_attrs,
                            "plugin_candidates": plugin_candidates,
                        },
                    )
                    
                    for attr_name in plugin_candidates:
                        attr_value = getattr(module, attr_name)
                        logger.debug(
                            "Examining candidate plugin class",
                            extra={
                                "attr_name": attr_name,
                                "attr_type": type(attr_value).__name__,
                                "is_class": isinstance(attr_value, type),
                                "bases": [base.__name__ for base in getattr(attr_value, "__bases__", ())] if isinstance(attr_value, type) else [],
                            },
                        )
                        if attr_name.endswith("Plugin"):
                            plugin_class = attr_value
                            break

                    if not plugin_class:
                        logger.error(
                            "No plugin class found in module",
                            extra={
                                "plugin_name": config.name,
                                "plugin_module": module.__name__,
                                "available_classes": [attr for attr in dir(module) if not attr.startswith("_")],
                                "module_path": str(module_path),
                            },
                        )
                        continue

                    plugin_name = plugin_class.__name__
                    logger.debug(
                        f"Found plugin class {plugin_name}",
                        extra={
                            "plugin_name": config.name,
                            "plugin_class": plugin_name,
                            "plugin_module": module.__name__,
                            "plugin_base": PluginBase.__name__,
                        },
                    )
                except ImportError as e:
                    logger.error(
                        f"Failed to import plugin module {config.name}",
                        extra={
                            "plugin_name": config.name,
                            "module_path": str(module_path),
                            "error": str(e),
                            "error_type": "ImportError",
                            "sys_path": sys.path,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed to load plugin module {config.name}",
                        extra={
                            "plugin_name": config.name,
                            "module_path": str(module_path),
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc(),
                            "module_attrs": dir(module) if 'module' in locals() else None,
                        },
                    )
                    continue

                # Verify it's a proper class that inherits from PluginBase
                logger.debug(
                    "Verifying plugin class",
                    extra={
                        "plugin_name": config.name,
                        "plugin_class": plugin_name,
                        "plugin_base": PluginBase.__name__,
                        "is_subclass": issubclass(plugin_class, PluginBase) if plugin_class else False,
                    },
                )

                if not plugin_class or not issubclass(plugin_class, PluginBase):
                    logger.error(
                        "Invalid plugin class",
                        extra={
                            "plugin_name": config.name,
                            "plugin_class": plugin_name if plugin_class else None,
                            "plugin_base": PluginBase.__name__,
                            "class_bases": [base.__name__ for base in plugin_class.__bases__] if plugin_class else [],
                            "error": "Plugin class must inherit from PluginBase",
                        },
                    )
                    continue

                # Successfully loaded plugin
                logger.info(
                    "✨ Plugin loaded successfully",
                    extra={
                        "plugin_name": config.name,
                        "plugin_version": config.version,
                        "plugin_enabled": config.enabled,
                        "plugin_dependencies": config.dependencies,
                    },
                )

                # Create plugin instance
                plugin = plugin_class(config, event_bus=self.event_bus)
                self.plugins[config.name] = plugin

            except Exception as e:
                logger.error(
                    "Error loading plugin",
                    extra={"config_file": str(config_file), "error": str(e)},
                )

        return configs

    async def initialize_plugins(self) -> None:
        """Initialize all plugins in dependency order"""
        logger.info("Initializing plugins", extra={"plugin_count": len(self.plugins)})

        # Build dependency graph and detect cycles
        graph = {
            name: {dep for dep in plugin.config.dependencies if dep in self.plugins}
            for name, plugin in self.plugins.items()
            if not plugin.is_initialized  # Only include uninitialized plugins
        }
        logger.debug(
            "Plugin dependency graph",
            extra={
                "graph": str(graph),
                "initialized_plugins": [
                    name for name, plugin in self.plugins.items()
                    if plugin.is_initialized
                ]
            }
        )

        # Check for missing dependencies
        for name, plugin in self.plugins.items():
            if plugin.is_initialized:
                continue
            missing = set(plugin.config.dependencies) - set(self.plugins.keys())
            if missing:
                logger.warning(
                    "Plugin has missing dependencies",
                    extra={
                        "plugin_name": name,
                        "missing_dependencies": list(missing)
                    }
                )

        initialized: set[str] = set()
        while graph:
            # Find plugins with no dependencies or only satisfied dependencies
            ready = {name for name, deps in graph.items() if not deps - initialized}
            if not ready:
                remaining = ", ".join(graph.keys())
                dependency_info = {name: list(deps) for name, deps in graph.items()}
                logger.error(
                    "Circular plugin dependencies detected",
                    extra={
                        "remaining_plugins": remaining,
                        "dependency_info": dependency_info,
                        "initialized_plugins": list(initialized)
                    }
                )
                raise ValueError(f"Circular plugin dependencies detected: {remaining}")

            # Initialize ready plugins
            for name in ready:
                plugin = self.plugins[name]
                if not plugin.is_initialized:  # Double-check initialization state
                    try:
                        logger.debug(
                            "Initializing plugin",
                            extra={
                                "plugin_name": name,
                                "plugin_version": plugin.version
                            }
                        )
                        await plugin.initialize()
                        initialized.add(name)
                    except Exception as e:
                        logger.error(
                            "Failed to initialize plugin",
                            extra={
                                "plugin_name": name,
                                "error": str(e)
                            }
                        )
                        raise
                else:
                    logger.debug(
                        "Plugin already initialized",
                        extra={
                            "plugin_name": name,
                            "plugin_version": plugin.version
                        }
                    )
                    initialized.add(name)
                del graph[name]

            # Update remaining dependencies
            for deps in graph.values():
                deps.difference_update(ready)

        logger.info(
            "All plugins initialized successfully",
            extra={
                "initialized_plugins": list(initialized),
                "total_plugins": len(self.plugins)
            }
        )

    async def shutdown_plugins(self) -> None:
        """Shutdown all plugins in reverse dependency order"""
        logger.info("Shutting down plugins", extra={"plugin_count": len(self.plugins)})

        # Shutdown in reverse dependency order
        for name, plugin in reversed(list(self.plugins.items())):
            try:
                logger.debug("Shutting down plugin", extra={"plugin_name": name})
                await plugin.shutdown()
                logger.info("Plugin shutdown successfully", extra={"plugin_name": name})
            except Exception as e:
                logger.error(
                    "Error shutting down plugin",
                    extra={"plugin_name": name, "error": str(e)},
                )

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get a plugin by name"""
        plugin = self.plugins.get(name)
        if plugin:
            logger.debug(
                "Retrieved plugin",
                extra={"plugin_name": name, "plugin_version": plugin.version},
            )
        else:
            logger.debug("Plugin not found", extra={"plugin_name": name})
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

            # Get plugin class
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
            logger.info("✨ Plugin loaded successfully")

        except Exception as e:
            logger.error(
                f"Failed to load plugin module {plugin_path.stem}",
                exc_info=True
            )
            raise
