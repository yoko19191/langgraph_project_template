"""Logging configuration utilities for the application."""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(
    save_log: bool = False,
    log_level: str | None = None,
    log_dir: Path | None = None,
    console_output: bool = True,
) -> logging.Logger:
    """Set up application logging with optional file output.

    Configures logging with:
    - Environment variable LOG_LEVEL support (defaults to INFO)
    - Optional file logging with timestamp-based filenames
    - Consistent formatting across console and file outputs
    - Automatic log directory creation

    Args:
        save_log: Whether to save logs to file (default: False)
        log_level: Log level override. If None, reads from LOG_LEVEL env var
                  (default: INFO). Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_dir: Custom log directory. If None, uses project_root/logs
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured root logger instance

    Example:
        >>> # Basic usage with console output only
        >>> logger = setup_logging()
        >>> logger.info("Application started")
        >>>
        >>> # Enable file logging
        >>> logger = setup_logging(save_log=True)
        >>>
        >>> # Custom log level via environment
        >>> os.environ['LOG_LEVEL'] = 'DEBUG'
        >>> logger = setup_logging(save_log=True)
        >>>
        >>> # Direct log level specification
        >>> logger = setup_logging(save_log=True, log_level='WARNING')
    """
    # Determine log level from parameter or environment variable
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        log_level = log_level.upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        # Use stderr for early logging issues before logger is configured
        import sys

        sys.stderr.write(
            f"Invalid log level '{log_level}'. Using INFO. Valid levels: {', '.join(valid_levels)}\n"
        )
        log_level = "INFO"

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Define log format with colors for console (if terminal supports it)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if save_log is True
    if save_log:
        # Determine log directory
        if log_dir is None:
            # Get project root (assuming this file is at src/app/common/utils/logging.py)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            log_dir = project_root / "logs"

        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"app_{timestamp}.log"
        log_filepath = log_dir / log_filename

        # Create file handler
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_filepath}")

    # Log the configuration
    root_logger.info(f"Logging initialized with level: {log_level}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming across the application.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Logger instance for the specified name

    Example:
        >>> # In your module
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Change the log level for all configured handlers.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> # Change to debug level at runtime
        >>> set_log_level('DEBUG')
    """
    level = level.upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if level not in valid_levels:
        raise ValueError(
            f"Invalid log level '{level}'. Valid levels: {', '.join(valid_levels)}"
        )

    numeric_level = getattr(logging, level)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)

    root_logger.info(f"Log level changed to: {level}")


def disable_logger(name: str) -> None:
    """Disable logging for a specific logger by name.

    Useful for silencing noisy third-party libraries.

    Args:
        name: Name of the logger to disable

    Example:
        >>> # Disable noisy library logs
        >>> disable_logger('urllib3')
        >>> disable_logger('langchain')
    """
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)
