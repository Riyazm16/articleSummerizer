import logging
from logging.handlers import TimedRotatingFileHandler
import os

def get_logger():
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    log_file_path = os.path.join(log_folder, "appLogs.log")
    handler = TimedRotatingFileHandler(
        log_file_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=True,
    )

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger = logging.getLogger("my_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


# # Example usage
# logger.info("Script started.")
# logger.warning("This is a warning message.")
# logger.error("An error occurred.")
# logger.info("Script finished.")
