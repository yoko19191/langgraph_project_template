"""Tests for logging configuration utilities."""

import logging
import os

import pytest

from src.app.common.utils.logging import (
    disable_logger,
    get_logger,
    set_log_level,
    setup_logging,
)


class TestLoggingSetup:
    """Test logging setup functionality."""

    def teardown_method(self):
        """Clean up after each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        # Reset log level
        root_logger.setLevel(logging.WARNING)
        # Clear LOG_LEVEL env var if set
        if "LOG_LEVEL" in os.environ:
            del os.environ["LOG_LEVEL"]

    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()

        assert logger is not None
        assert logger.level == logging.INFO
        # Should have console handler
        assert len(logger.handlers) >= 1

    def test_setup_logging_with_env_var(self):
        """Test logging setup with LOG_LEVEL environment variable."""
        os.environ["LOG_LEVEL"] = "DEBUG"

        logger = setup_logging()

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_explicit_level(self):
        """Test logging setup with explicit log level."""
        logger = setup_logging(log_level="WARNING")

        assert logger.level == logging.WARNING

    def test_setup_logging_explicit_overrides_env(self):
        """Test that explicit log level overrides environment variable."""
        os.environ["LOG_LEVEL"] = "DEBUG"

        logger = setup_logging(log_level="ERROR")

        # Explicit parameter should override env var
        assert logger.level == logging.ERROR

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        # Should fall back to INFO
        logger = setup_logging(log_level="INVALID")

        assert logger.level == logging.INFO

    def test_setup_logging_case_insensitive(self):
        """Test that log levels are case-insensitive."""
        logger = setup_logging(log_level="debug")

        assert logger.level == logging.DEBUG

    def test_setup_logging_without_console(self):
        """Test logging setup without console output."""
        logger = setup_logging(console_output=False)

        # Should have no handlers (or only file handler if save_log=True)
        assert len(logger.handlers) == 0

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file output."""
        log_dir = tmp_path / "logs"

        logger = setup_logging(save_log=True, log_dir=log_dir)

        # Should have console and file handlers
        assert len(logger.handlers) >= 2

        # Log directory should be created
        assert log_dir.exists()

        # Should have a log file
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1

        # Log file should have content
        log_file = log_files[0]
        assert log_file.stat().st_size > 0

    def test_setup_logging_file_only(self, tmp_path):
        """Test logging setup with file output only (no console)."""
        log_dir = tmp_path / "logs"

        logger = setup_logging(save_log=True, console_output=False, log_dir=log_dir)

        # Should have only file handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

    def test_log_file_naming(self, tmp_path):
        """Test that log files use timestamp-based naming."""
        log_dir = tmp_path / "logs"

        setup_logging(save_log=True, log_dir=log_dir)

        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1

        # Check filename format: app_YYYYMMDD_HHMMSS.log
        filename = log_files[0].name
        assert filename.startswith("app_")
        assert filename.endswith(".log")
        # Extract timestamp part
        timestamp_part = filename[4:-4]  # Remove 'app_' and '.log'
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        assert "_" in timestamp_part

    def test_multiple_setup_calls(self):
        """Test that multiple setup calls don't create duplicate handlers."""
        setup_logging()
        initial_handler_count = len(logging.getLogger().handlers)

        setup_logging()
        final_handler_count = len(logging.getLogger().handlers)

        # Should not create duplicate handlers
        assert final_handler_count == initial_handler_count


class TestGetLogger:
    """Test get_logger functionality."""

    def test_get_logger(self):
        """Test getting a named logger."""
        logger = get_logger("test_module")

        assert logger is not None
        assert logger.name == "test_module"

    def test_get_logger_with_dunder_name(self):
        """Test getting a logger with __name__."""
        logger = get_logger(__name__)

        assert logger is not None
        assert logger.name == __name__


class TestSetLogLevel:
    """Test set_log_level functionality."""

    def setup_method(self):
        """Setup before each test."""
        setup_logging(log_level="INFO")

    def teardown_method(self):
        """Clean up after each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

    def test_set_log_level(self):
        """Test changing log level at runtime."""
        set_log_level("DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # All handlers should also be updated
        for handler in root_logger.handlers:
            assert handler.level == logging.DEBUG

    def test_set_log_level_invalid(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError):
            set_log_level("INVALID")

    def test_set_log_level_case_insensitive(self):
        """Test that set_log_level is case-insensitive."""
        set_log_level("warning")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING


class TestDisableLogger:
    """Test disable_logger functionality."""

    def test_disable_logger(self):
        """Test disabling a specific logger."""
        logger = get_logger("noisy_library")
        disable_logger("noisy_library")

        # Logger should be effectively disabled
        assert logger.level == logging.CRITICAL + 1


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def teardown_method(self):
        """Clean up after each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

    def test_full_workflow(self, tmp_path):
        """Test complete logging workflow."""
        log_dir = tmp_path / "logs"

        # Setup logging with file output
        _logger = setup_logging(save_log=True, log_level="INFO", log_dir=log_dir)

        # Get a named logger
        module_logger = get_logger("test_module")

        # Log messages at different levels
        module_logger.debug("This should not appear")  # Below INFO level
        module_logger.info("Info message")
        module_logger.warning("Warning message")
        module_logger.error("Error message")

        # Change log level
        set_log_level("DEBUG")

        module_logger.debug("This should appear now")

        # Check log file exists and has content
        log_files = list(log_dir.glob("app_*.log"))
        assert len(log_files) == 1

        log_content = log_files[0].read_text()
        assert "Info message" in log_content
        assert "Warning message" in log_content
        assert "Error message" in log_content
        assert "This should appear now" in log_content

    def test_environment_variable_integration(self, tmp_path):
        """Test logging with environment variable configuration."""
        os.environ["LOG_LEVEL"] = "WARNING"
        log_dir = tmp_path / "logs"

        _logger = setup_logging(save_log=True, log_dir=log_dir)

        module_logger = get_logger("test_module")
        module_logger.debug("Debug message")
        module_logger.info("Info message")
        module_logger.warning("Warning message")

        # Only warning and above should be in log
        log_files = list(log_dir.glob("app_*.log"))
        log_content = log_files[0].read_text()

        assert "Debug message" not in log_content
        assert "Info message" not in log_content
        assert "Warning message" in log_content

        # Clean up
        del os.environ["LOG_LEVEL"]


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
