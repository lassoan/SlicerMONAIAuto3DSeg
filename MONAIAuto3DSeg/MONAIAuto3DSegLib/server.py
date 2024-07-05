import slicer
import psutil

import logging
import queue
import threading

import qt
from pathlib import Path
from typing import Callable

from MONAIAuto3DSegLib.constants import APPLICATION_NAME
logger = logging.getLogger(APPLICATION_NAME)


class WebServer:

  CHECK_TIMER_INTERVAL = 1000

  @staticmethod
  def getPSProcess(pid):
    try:
      return psutil.Process(pid)
    except psutil.NoSuchProcess:
      return None

  def __init__(self, completedCallback: Callable):
    self.completedCallback = completedCallback
    self.procThread = None
    self.serverProc = None
    self.queue = None

  def isRunning(self):
    if self.serverProc is not None:
      psProcess = self.getPSProcess(self.serverProc.pid)
      if psProcess:
        return psProcess.is_running()
    return False

  def __del__(self):
    self.killProcess()

  def killProcess(self):
    if not self.serverProc:
      return
    psProcess = self.getPSProcess(self.serverProc.pid)
    if not psProcess:
      return
    for psChildProcess in psProcess.children(recursive=True):
      psChildProcess.kill()
    if psProcess.is_running():
      psProcess.kill()

  def launchConsoleProcess(self, cmd):
    self.serverProc = \
      slicer.util.launchConsoleProcess(cmd, cwd=Path(__file__).parent.parent / "auto3dseg", useStartupEnvironment=False)

    self.queue = queue.Queue()
    self.procThread = threading.Thread(target=self._handleProcessOutputThreadProcess)
    self.procThread.start()
    self.checkProcessOutput()

  def cleanup(self):
    if self.procThread:
      self.procThread.join()
    self.completedCallback()
    self.serverProc = None
    self.procThread = None
    self.queue = None

  def _handleProcessOutputThreadProcess(self):
    while True:
      try:
        line = self.serverProc.stdout.readline()
        if not line:
          break
        self.queue.put(line.rstrip())
      except UnicodeDecodeError as e:
        pass
    self.serverProc.wait()

  def checkProcessOutput(self):
    outputQueue = self.queue
    while outputQueue:
      try:
        line = outputQueue.get_nowait()
        logger.info(line)
      except queue.Empty:
        break

    psProcess = self.getPSProcess(self.serverProc.pid)
    if psProcess and psProcess.is_running(): # No more outputs to process now, check again later
      qt.QTimer.singleShot(self.CHECK_TIMER_INTERVAL, self.checkProcessOutput)
    else:
      self.cleanup()