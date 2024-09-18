import logging
import os
import sys
from types import FrameType
from typing import cast

from loguru import logger


class Logger:
    filepath = "logs"


config = Logger()


class Logger:
    """输出日志到文件和控制台"""

    def __init__(self):
        self.logger = logger
        self.logger.remove()

        log_path = os.path.join(config.filepath, "{time:YYYY-MM-DD}.log")
        if not os.path.exists(config.filepath):
            os.makedirs(config.filepath)

        self.logger.add(
            sys.stdout,
            format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "
            "{process.name} | "
            "{thread.name} | "
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>"
            ":<cyan>{line}</cyan> | "
            "<level>{level}</level>: "
            "<level>{message}</level>",
            level="WARNING",
        )

        self.logger.add(
            log_path,
            format="{time:YYYYMMDD HH:mm:ss} - "
            "{process.name} | "
            "{thread.name} | "
            "{module}.{function}:{line} - {level} -{message}",
            encoding="utf-8",
            retention="7 days",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            rotation="00:00",
        )

    def init_config(self):
        LOGGER_NAMES = ("uvicorn.asgi", "uvicorn.access", "uvicorn")
        logging.getLogger().handlers = [InterceptHandler()]
        for logger_name in LOGGER_NAMES:
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = [InterceptHandler()]

    def get_logger(self):
        return self.logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


Loggers = Logger()
log = Loggers.get_logger()
