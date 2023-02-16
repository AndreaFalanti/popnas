import logging
import os.path

from flask.logging import default_handler


def get_logger(name, filename='server.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(name)s:%(levelname)s] %(message)s"))

    logger.addHandler(default_handler)
    logger.addHandler(file_handler)

    return logger
