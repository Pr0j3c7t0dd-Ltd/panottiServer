"""Plugin manager for PanottiServer."""
import logging
from importlib.metadata import entry_points
import pluggy
import asyncio
import os
import importlib.util
from pathlib import Path
from typing import Optional, Any, Callable
from .hookspec import PanottiSpecs

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages the lifecycle and execution of PanottiServer plugins."""

    def __init__(self):
        """Initialize the plugin manager."""
        self._plugin_manager: Optional[pluggy.PluginManager] = None

    def _load_user_plugins(self) -> None:
        """Load plugins from the user_plugins directory."""
        user_plugins_dir = Path(__file__).parent.parent / "user_plugins"
        if not user_plugins_dir.exists():
            logger.info("Creating user_plugins directory")
            user_plugins_dir.mkdir(exist_ok=True)
            return

        # Iterate through directories in user_plugins
        for plugin_dir in user_plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith('_'):
                continue

            # Look for plugin.py in the directory
            plugin_file = plugin_dir / "plugin.py"
            if not plugin_file.exists():
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"app.user_plugins.{plugin_dir.name}",
                    str(plugin_file)
                )
                if spec is None or spec.loader is None:
                    logger.error(f"Failed to load plugin spec from {plugin_file}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find and register plugin classes
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if isinstance(item, type) and any(
                        hasattr(item, hook_name) for hook_name in [
                            "on_startup", "on_shutdown", "before_request",
                            "after_request", "before_recording_start",
                            "after_recording_end"
                        ]
                    ):
                        plugin_instance = item()
                        self._plugin_manager.register(plugin_instance)
                        logger.info(f"Loaded user plugin: {item_name} from {plugin_dir.name}")

            except Exception as e:
                logger.error(f"Failed to load user plugin from {plugin_dir}: {str(e)}")

    def setup(self) -> None:
        """Set up the plugin manager and load plugins."""
        # Create plugin manager
        self._plugin_manager = pluggy.PluginManager("panotti")
        
        # Add hook specifications
        self._plugin_manager.add_hookspecs(PanottiSpecs)
        
        # Load plugins from entry points
        try:
            discovered_plugins = entry_points(group="panotti.plugins")
            for entry_point in discovered_plugins:
                try:
                    plugin = entry_point.load()
                    self._plugin_manager.register(plugin)
                    logger.info(f"Loaded plugin: {entry_point.name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {entry_point.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error discovering plugins: {str(e)}")

        # Load user plugins
        self._load_user_plugins()

    async def call_hook(self, hook_name: str, **kwargs: Any) -> None:
        """Call a hook with the given name and arguments.
        
        This method handles both sync and async hook implementations.
        
        Args:
            hook_name: Name of the hook to call
            **kwargs: Arguments to pass to the hook
        """
        if self._plugin_manager is None:
            raise RuntimeError("Plugin manager not initialized. Call setup() first.")
        
        hook_caller = getattr(self._plugin_manager.hook, hook_name)
        results = hook_caller(**kwargs)
        
        # If results is a list, it means we have multiple implementations
        if isinstance(results, list):
            # Convert any sync results to coroutines
            coroutines = []
            for result in results:
                if asyncio.iscoroutine(result):
                    coroutines.append(result)
                else:
                    # If it's a synchronous result, wrap it in a coroutine
                    coroutines.append(asyncio.create_task(asyncio.to_thread(result)))
            
            # Wait for all coroutines to complete
            if coroutines:
                await asyncio.gather(*coroutines)

    @property
    def plugin_manager(self) -> pluggy.PluginManager:
        """Get the underlying pluggy PluginManager instance.
        
        Returns:
            pluggy.PluginManager: The plugin manager instance
        
        Raises:
            RuntimeError: If setup() hasn't been called
        """
        if self._plugin_manager is None:
            raise RuntimeError("Plugin manager not initialized. Call setup() first.")
        return self._plugin_manager

# Global plugin manager instance
plugin_manager = PluginManager()
