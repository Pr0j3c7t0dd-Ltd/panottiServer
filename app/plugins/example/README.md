# Example Plugin

This is a reference implementation of a Panotti Server plugin that demonstrates best practices for plugin development and serves as a template for creating new plugins.

## Overview

The Example Plugin showcases the basic structure and essential components required for a Panotti Server plugin, including:
- Plugin configuration using `plugin.yaml`
- Event handling and processing
- Integration with the Panotti Server plugin system

## Structure

```
example/
├── __init__.py      # Plugin initialization
├── plugin.py        # Main plugin implementation
├── plugin.yaml      # Plugin configuration
└── README.md        # Plugin documentation
```

## Configuration

The plugin is configured through `plugin.yaml`:
- **name**: Unique identifier for the plugin
- **version**: Plugin version following semantic versioning
- **description**: Brief description of the plugin's purpose
- **author**: Plugin creator/maintainer

## Installation

1. Ensure the plugin directory is placed in the `app/plugins/` directory of your Panotti Server installation
2. Copy `plugin.yaml.example` to `plugin.yaml` and update the configuration according to your needs
3. The plugin will be automatically discovered and loaded by the plugin manager
4. Verify the plugin is loaded by checking the server logs on startup

## Development

When developing plugins based on this example:

1. Copy the example plugin structure as a starting point
2. Update `plugin.yaml` with your plugin's information
3. Modify `plugin.py` to implement your plugin's logic
4. Add any necessary dependencies to the main project's requirements
5. Follow the type hints and docstrings for proper integration

## Testing

To test the plugin:
1. Ensure Panotti Server is running
2. The plugin will automatically process relevant events
3. Check server logs for plugin activity and any potential errors

## Best Practices

- Use type hints for all function parameters and return values
- Include comprehensive docstrings for classes and methods
- Follow PEP 8 style guidelines
- Implement proper error handling and logging
- Keep the plugin focused on a single responsibility
- Use asynchronous programming where appropriate

## Contributing

When contributing improvements to this example plugin:
1. Follow the project's coding standards
2. Add/update documentation as needed
3. Ensure all existing functionality remains intact
4. Add relevant tests for new features

## License

This example plugin is released under the same license as the main Panotti Server project.
