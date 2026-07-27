# utils/logger.py

import logging
import sys
import colorlog


def get_logger(name: str, level=logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    formatter = colorlog.ColoredFormatter(
        fmt="%(asctime)s | %(log_color)s%(levelname)s%(reset)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger