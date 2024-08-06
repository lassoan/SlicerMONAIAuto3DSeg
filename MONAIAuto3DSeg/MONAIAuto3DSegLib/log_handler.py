import logging
from typing import Callable


class LogHandler(logging.Handler):
  """

  code:

    logger = logging.getLogger("XYZ")

    callback = ... # any callable
    # NB: only catching info level messages and forwarding it to callback
    handler = LogHandler(callback, logging.INFO)
    # can format log messages
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  """
  def __init__(self, callback: Callable, level=logging.NOTSET):
    self._callback = callback
    super().__init__(level)

  def emit(self, record):
    msg = self.format(record)
    self._callback(msg)