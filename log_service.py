import logging
import sys

# Used for unhandled exceptions only
_logger = logging.getLogger(__name__)
fHandler = logging.FileHandler("critical.log")
fHandler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
_logger.addHandler(fHandler)


def getLogger(name):
    logger = logging.getLogger(name)

    # Create handlers
    file_handler = logging.FileHandler("debug.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        #format="%(asctime)s - [%(name)s:%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            console_handler
        ]
    )
    return logger

# Taken from: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exception logging, must be bound to sys.excepthook.
    """
    # Avoid to log keyboard interrupts
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    _logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))