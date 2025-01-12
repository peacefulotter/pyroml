import logging

from rich.logging import RichHandler


def initialize_logging():
    logging.basicConfig()
    logging.captureWarnings(True)
    logging.root.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.WARNING, handlers=[RichHandler(level=logging.WARNING)]
    )


def get_logger(name: str, level=logging.WARNING):
    logging.basicConfig(level=level, handlers=[RichHandler(level=level)])
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def set_level_all_loggers(level: int):
    """
    level: logging.INFO | logging.CRITICAL | logging.ERROR | logging.WARNING | logging.INFO | logging.DEBUG | logging.NOTSET
    """
    for name in logging.root.manager.loggerDict:
        if not name.startswith("pyro"):
            continue
        logger = logging.getLogger(name)
        logger.setLevel(level)
