import logging

logging.basicConfig()
logging.root.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


def set_level_all_loggers(level: int):
    """
    level: logging.INFO | logging.CRITICAL | logging.ERROR | logging.WARNING | logging.INFO | logging.DEBUG | logging.NOTSET
    """
    for name in logging.root.manager.loggerDict:
        if not name.startswith("pyro"):
            continue
        logger = logging.getLogger(name)
        logger.setLevel(level)
