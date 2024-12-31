from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ..utils import get_logger, setup_logging

# Initialize logging when the module is imported
logger = setup_logging()


class PluginConfig(BaseModel):
    """Base configuration model for plugins"""

    name: str
    version: str
    enabled: bool = True
    dependencies: list[str] = []
    config: dict[str, Any] | None = {}


class PluginBase(ABC):
    """Base class for all plugins"""

    def __init__(self, config: PluginConfig, event_bus=None) -> None:
        self.config = config
        self.logger = get_logger(f"plugin.{config.name}")
        self._initialized = False
        self.event_bus = event_bus

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the plugin"""
        try:
            self.logger.info(
                "Initializing plugin",
                extra={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "plugin_config": self.config.dict(),
                },
            )
            await self._initialize()
            self._initialized = True
            self.logger.info(
                "Plugin initialized successfully", extra={"plugin_name": self.name}
            )
        except Exception as e:
            self.logger.error(
                "Failed to initialize plugin",
                extra={
                    "plugin_name": self.name,
                    "error": str(e),
                    "plugin_config": self.config.dict(),
                },
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        try:
            self.logger.info("Shutting down plugin", extra={"plugin_name": self.name})
            await self._shutdown()
            self._initialized = False
            self.logger.info(
                "Plugin shutdown successfully", extra={"plugin_name": self.name}
            )
        except Exception as e:
            self.logger.error(
                "Failed to shutdown plugin",
                extra={"plugin_name": self.name, "error": str(e)},
            )
            raise

    @abstractmethod
    async def _initialize(self) -> None:
        """Custom initialization logic to be implemented by plugins"""
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """Custom shutdown logic to be implemented by plugins"""
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        value = self.config.config.get(key, default)
        self.logger.debug(
            "Retrieved plugin config",
            extra={"plugin_name": self.name, "config_key": key, "config_value": value},
        )
        return value
