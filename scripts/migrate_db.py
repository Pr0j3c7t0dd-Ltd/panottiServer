#!/usr/bin/env python3
"""Script to run database migrations."""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.models.database import DatabaseManager  # noqa: E402


def main() -> None:
    """Run database migrations."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Get database manager instance and run migrations
        DatabaseManager.get_instance()

        # The migrations will run automatically during initialization
        logger.info("Database migrations completed successfully")

    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
