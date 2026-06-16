import logging
from pathlib import Path
from typing import Optional

# Global variable: stores the full formatter instance for restoration
_full_formatter: Optional[logging.Formatter] = None


def setup_logging(
    log_file: Path,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure global logging: output to both console and a single log file (no rotation).
    Must be called once at the program entry point.

    Args:
        log_file: Full path to the log file. Parent directories are created automatically.
        level: Root logger level.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
    """
    global _full_formatter

    # Ensure the directory for the log file exists
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Full format with timestamp, logger name, level, filename, line number
    _full_formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler (using full format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_full_formatter)
    root_logger.addHandler(console_handler)

    # File handler (no rotation, using full format)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_full_formatter)
    root_logger.addHandler(file_handler)

    # Prevent propagation (root has no parent, but keep for clarity)
    root_logger.propagate = False

    # Log startup message
    # root_logger.info(f"Logging initialized. Log file: {log_file}")


def set_simple_logging() -> None:
    """
    Switch all log handlers to a simple format that shows only the message.
    Useful when printing large amounts of data to avoid visual clutter.
    """
    simple_formatter = logging.Formatter(fmt='%(message)s')
    _apply_formatter_to_all_handlers(simple_formatter)
    # logging.getLogger().info("Switched to simple logging format (message only)")


def set_normal_logging() -> None:
    """
    Restore the full logging format (timestamp, logger name, level, file name, line number).
    """
    if _full_formatter is None:
        raise RuntimeError("Full formatter not initialized. Call setup_logging() first.")
    _apply_formatter_to_all_handlers(_full_formatter)
    # logging.getLogger().info("Switched to normal logging format")


def _apply_formatter_to_all_handlers(formatter: logging.Formatter) -> None:
    """Helper function: replace the formatter of all handlers attached to the root logger."""
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)


def center_string(s: str = '=', width: int = 80, fillchar: str = '=') -> str:
    if len(fillchar) != 1:
        raise ValueError("fillchar must be single char")
    return s.center(width, fillchar)
