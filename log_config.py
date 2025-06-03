import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = "casiquiare", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with a rotating file handler.

    Parameters
    ----------
    name:
        Name of the logger.
    log_dir:
        Directory where log files will be stored.
    level:
        Logging level to use. Defaults to ``logging.INFO``.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "casiquiare.log"
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(handler)
    logger.propagate = True
    return logger
