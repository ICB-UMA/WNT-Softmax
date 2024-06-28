import logging
import sys

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga 
"""

def setup_custom_logger(name, log_file='application.log', enable_console=True):
    """
    Set up a custom logger that conditionally formats output with color codes based on the output type.

    Args:
        name (str): Name of the logger.
        log_file (str): Filename for the log file.
        enable_console (bool): If True, logs will also be printed to console.

    Returns:
        logging.Logger: Configured logger with optional color formatting and multiple outputs.
    """
    class CustomFormatter(logging.Formatter):
        # Define basic format with optional color
        base_format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        datefmt = "%Y-%m-%d %H:%M:%S"

        COLORS = {
            logging.DEBUG: "\x1b[38;21m",
            logging.INFO: "\x1b[37m",
            logging.WARNING: "\x1b[33m",
            logging.ERROR: "\x1b[31m",
            logging.CRITICAL: "\x1b[31;1m",
            "RESET": "\x1b[0m"
        }

        def format(self, record):
            if self._stream_supports_color(record):
                log_fmt = self.COLORS.get(record.levelno, "RESET") + self.base_format + self.COLORS["RESET"]
            else:
                log_fmt = self.base_format
            self._style._fmt = log_fmt
            return super().format(record)

        def _stream_supports_color(self, record):
            """ Checks if the output stream supports color coding. """
            if getattr(record, 'stream', None):
                return hasattr(record.stream, 'isatty') and record.stream.isatty()
            return False

    handlers = []
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomFormatter())
        handlers.append(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        handlers.append(console_handler)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  
    for handler in handlers:
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)

    return logger
