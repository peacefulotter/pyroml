import logging


def get_logger(
    name: str, level: int | str | None = logging.WARNING, allow_warnings: bool = True
):
    logging.captureWarnings(allow_warnings)
    logging.basicConfig(
        level=level or logging.NOTSET,
        format="%(message)s",
        datefmt="[%X]",
        # handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
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
