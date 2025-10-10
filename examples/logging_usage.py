"""Example usage of logging utilities.

This demonstrates how to use the logging configuration system with:
- Environment variable configuration
- File logging with timestamps
- Runtime log level changes
- Third-party library log suppression
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app.common.utils.logging import (  # noqa: E402
    disable_logger,
    get_logger,
    set_log_level,
    setup_logging,
)


def example_basic_usage():
    """Demonstrate basic logging setup with console output only."""
    print("\n=== Basic Usage Example ===\n")  # noqa: T201

    # Setup logging with default settings (INFO level, console only)
    setup_logging()

    # Get a logger for your module
    logger = get_logger(__name__)

    # Log messages at different levels
    logger.debug("This won't appear (below INFO level)")
    logger.info("Application started")
    logger.warning("This is a warning")
    logger.error("This is an error")


def example_environment_variable():
    """Demonstrate using environment variable for log level configuration."""
    print("\n=== Environment Variable Example ===\n")  # noqa: T201

    # Set LOG_LEVEL environment variable
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Setup logging - will use DEBUG level from environment
    setup_logging()

    logger = get_logger(__name__)
    logger.debug("Debug messages now appear!")
    logger.info("Info message")

    # Clean upw
    del os.environ["LOG_LEVEL"]


def example_file_logging():
    """Demonstrate logging to file with timestamp-based naming."""
    print("\n=== File Logging Example ===\n")  # noqa: T201

    # Setup logging with file output enabled
    # Files will be saved to project_root/logs/ with timestamp naming
    setup_logging(save_log=True, log_level="INFO")

    # Get module logger
    module_logger = get_logger("my_module")

    # Log messages - will appear in both console and file
    module_logger.info("This goes to console and file")
    module_logger.warning("Warning message saved to log file")
    module_logger.error("Error message with details")

    # The log file path is printed by setup_logging
    print("\nCheck the logs/ directory for the timestamped log file")  # noqa: T201


def example_file_only():
    """Demonstrate logging to file only (no console output)."""
    print("\n=== File-Only Logging Example ===\n")  # noqa: T201

    # Setup logging with file output but no console
    setup_logging(save_log=True, console_output=False, log_level="DEBUG")

    logger = get_logger(__name__)

    # These messages will only appear in the log file
    logger.debug("Debug info saved to file")
    logger.info("Info saved to file")

    print("Logs saved to file only (not shown in console)")  # noqa: T201


def example_runtime_level_change():
    """Demonstrate changing log level at runtime."""
    print("\n=== Runtime Level Change Example ===\n")  # noqa: T201

    # Start with INFO level
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)

    logger.debug("This won't appear")
    logger.info("Info message appears")

    # Change to DEBUG level at runtime
    print("\nChanging log level to DEBUG...")  # noqa: T201
    set_log_level("DEBUG")

    logger.debug("Debug messages now appear!")
    logger.info("Info still appears")


def example_disable_noisy_loggers():
    """Silencing noisy third-party libraries."""
    print("\n=== Disable Noisy Loggers Example ===\n")  # noqa: T201

    # Setup logging
    setup_logging(log_level="DEBUG")

    # Disable noisy third-party libraries
    disable_logger("urllib3")
    disable_logger("httpx")
    disable_logger("httpcore")
    disable_logger("langchain")

    # Your application logs will still work
    logger = get_logger(__name__)
    logger.info("Application logs still work")
    logger.debug("But noisy libraries are silenced")


def example_production_setup():
    """Recommended production logging setup."""
    print("\n=== Production Setup Example ===\n")  # noqa: T201

    # Production configuration:
    # - Log level from environment (default: INFO)
    # - Save logs to file with timestamps
    # - Console output enabled for container logs
    setup_logging(
        save_log=True,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        console_output=True,
    )

    # Disable noisy libraries
    for lib in ["urllib3", "httpx", "httpcore"]:
        disable_logger(lib)

    # Use logging in your application
    logger = get_logger(__name__)
    logger.info("Production application started")
    logger.info("Configuration loaded successfully")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_environment_variable()
    example_file_logging()
    example_runtime_level_change()
    example_disable_noisy_loggers()

    print("\n=== All Examples Complete ===\n")  # noqa: T201
    print(  # noqa: T201
        "Production setup example (not run automatically):\n"  # noqa: T201
        "example_production_setup()\n"
    )
