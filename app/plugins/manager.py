import importlib
from pathlib import Path

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
            extra={"config_files": [str(f) for f in config_files]},
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
                    },
                )

                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    logger.debug(
                        "Loaded plugin config data", extra={"config_data": config_data}
                    )

                config = PluginConfig(**config_data)
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
                        f"Importing plugin package {plugin_package}",
                        extra={
                            "plugin_package": plugin_package,
                            "plugin_name": config.name,
                        },
                    )
                    module = importlib.import_module(plugin_package)

                    # Get plugin class from the module
                    plugin_class = None
                    for attr_name in dir(module):
                        if attr_name.endswith("Plugin"):
                            plugin_class = getattr(module, attr_name)
                            break

                    plugin_name = plugin_class.__name__ if plugin_class else None
                    logger.debug(
                        f"Found plugin class {plugin_name}",
                        extra={
                            "plugin_name": config.name,
                            "plugin_class": plugin_name,
                            "plugin_module": module.__name__,
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load plugin module {config.name}",
                        extra={
                            "plugin_name": config.name,
                            "module_path": str(module_path),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    continue

                # Verify it's a proper class that inherits from PluginBase
                logger.debug(
                    "Verifying plugin class",
                    extra={
                        "plugin_name": config.name,
                        "plugin_class": plugin_name,
                    },
                )

                if not plugin_class or not issubclass(plugin_class, PluginBase):
                    logger.error(
                        "Invalid plugin class",
                        extra={
                            "plugin_name": config.name,
                            "plugin_class": plugin_name,
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
        }
        logger.debug("Plugin dependency graph", extra={"graph": str(graph)})

        # Check for missing dependencies
        for name, plugin in self.plugins.items():
            missing = set(plugin.config.dependencies) - set(self.plugins.keys())
            if missing:
                logger.warning(
                    "Plugin has missing dependencies",
                    extra={"plugin_name": name, "missing_dependencies": list(missing)},
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
                        "initialized_plugins": list(initialized),
                    },
                )
                raise ValueError(f"Circular plugin dependencies detected: {remaining}")

            # Initialize ready plugins
            for name in ready:
                plugin = self.plugins[name]
                try:
                    logger.debug(
                        "Initializing plugin",
                        extra={"plugin_name": name, "plugin_version": plugin.version},
                    )
                    await plugin.initialize()
                    initialized.add(name)
                except Exception as e:
                    logger.error(
                        "Failed to initialize plugin",
                        extra={"plugin_name": name, "error": str(e)},
                    )
                    raise
                del graph[name]

            # Update remaining dependencies
            for deps in graph.values():
                deps.difference_update(ready)

        logger.info(
            "All plugins initialized successfully",
            extra={"initialized_plugins": list(initialized)},
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
