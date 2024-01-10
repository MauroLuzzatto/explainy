# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:13:34 2020

@author: mauro
"""
import logging
import os


class Logger(object):
    def __init__(self, name: str, path_log: str):
        self.name = name
        self.path_log = path_log

    def get_logger(self) -> logging.Logger:
        """
        Create a log file to record the experiment's logs

        Args:

        Return:
            logger (obj): logger that record logs
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        if logger.hasHandlers():
            for _logger in logger.handlers:
                _logger.close()
            logger.handlers.clear()

        if self.path_log:
            log_file = os.path.join(self.path_log, f"{self.name}.log")
            # file handler
            file_handler = logging.FileHandler(filename=log_file, mode="w")
            file_logging_format = (
                "%(asctime)s: %(levelname)s: %(filename)s: %(message)s"
            )
            formatter = logging.Formatter(file_logging_format)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        # console handler
        console_handler = logging.StreamHandler()
        console_logging_format = "%(message)s"
        formatter = logging.Formatter(console_logging_format)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level=logging.DEBUG)
        logger.addHandler(console_handler)
        return logger
