# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:13:34 2020

@author: mauro
"""
import logging
import os


class Logger(object):
    def __init__(self, name, path_log):
        self.name = name
        self.path_log = path_log

    def get_logger(self):
        """
        Create a log file to record the experiment's logs

        Args:

        Retunr:
            logger (obj): logger that record logs
        """
        # check if the file exist
        log_file = os.path.join(self.path_log, "{}.log".format(self.name))

        file_logging_format = "%(asctime)s: %(levelname)s: %(message)s"
        # configure logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        # Reset the logger.handlers if it already exists.
        if logger.handlers:
            logger.handlers = []

        # create a file handler for output file
        handler = logging.FileHandler(filename=log_file, mode="w")
        formatter = logging.Formatter(file_logging_format)
        handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # set logging format
        console_logging_format = "%(asctime)s: %(levelname)s: %(message)s"
        # console log
        consoleHandler = logging.StreamHandler()
        # set the logging level for log file
        consoleHandler.setLevel(level=logging.INFO)
        # set the logging format
        formatter = logging.Formatter(console_logging_format)
        consoleHandler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(consoleHandler)
        return logger
