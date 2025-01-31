
import os
import logging
from rich.console import Console
from rich.logging import RichHandler

class MaxLevelFilter(logging.Filter):
    """
    Allows log records with level less than max_level.
    """
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno < self.max_level

def setup_logger(logger_name, log_file_name, log_folder=None, log_level=logging.DEBUG):
    """
    Sets up a logger with a file handler for logs and a Rich console handler for errors.

    Args:
        logger_name (str): The name of the logger.
        log_file_name (str): The name of the log file (e.g., "simulation.log").
        log_folder (str): The folder to store the log file. Defaults to current directory if None.
        log_level (int): The logging level for the file. Default is DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create log folder if it doesn't exist
    if log_folder:
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, log_file_name)
    else:
        log_file_path = log_file_name

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all messages

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for DEBUG and INFO
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.addFilter(MaxLevelFilter(logging.ERROR))  # Exclude ERROR and above

    # Console handler for ERROR and above
    console_handler = RichHandler(
        console=Console(),
        level=logging.ERROR, 
        show_time=False,
        show_level=False
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Verification: Print out handlers and their levels
    print(f"Logger '{logger_name}' handlers:")
    for handler in logger.handlers:
        handler_type = type(handler).__name__
        handler_level = logging.getLevelName(handler.level)
        print(f" - {handler_type} with level {handler_level}")

    logger.propagate = False  # Prevent messages from being propagated to the root logger

    return logger