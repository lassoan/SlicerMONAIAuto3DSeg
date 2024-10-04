import slicer
import psutil

import logging
import queue
import threading

import qt
from pathlib import Path
from typing import Callable


class WebServer:
    """ Web server class to be used from 3D Slicer. Upon starting the server with the given cmd command, it will be
    checked every {CHECK_TIMER_INTERVAL} milliseconds for process status/outputs.

    code:

        cmd = [sys.executable, "main.py", "--host", hostName, "--port", port]

        from MONAIAuto3DSegLib.server import WebServer
        server = WebServer(completedCallback=...)
        server.launchConsoleProcess(cmd)

        ...

        server.killProcess()

    """

    CHECK_TIMER_INTERVAL = 1000

    @staticmethod
    def getPSProcess(pid):
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None

    def __init__(self, logCallback: Callable=None, completedCallback: Callable=None):
        self.logCallback = logCallback
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
        psProcess = self.getPSProcess(self.serverProc.pid) # proc.kill() does not work, that would only stop the launcher
        if not psProcess:
            return
        for psChildProcess in psProcess.children(recursive=True):
            psChildProcess.kill()
        if psProcess.is_running():
            psProcess.kill()

    def launchConsoleProcess(self, cmd):
        self.serverProc = \
            slicer.util.launchConsoleProcess(cmd, cwd=Path(__file__).parent.parent / "MONAIAuto3DSegServer", useStartupEnvironment=False)

        if self.logCallback and self.isRunning():
            self.logCallback("Server Started")

        self.queue = queue.Queue()
        self.procThread = threading.Thread(target=self._handleProcessOutputThreadProcess)
        self.procThread.start()
        self.checkProcessOutput()

    def cleanup(self):
        if self.procThread:
            self.procThread.join()
        if self.completedCallback:
            self.completedCallback()
        if self.logCallback:
            self.logCallback("Server Stopped")
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
                logging.info(line)
            except queue.Empty:
                break

        psProcess = self.getPSProcess(self.serverProc.pid)
        if psProcess and psProcess.is_running(): # No more outputs to process now, check again later
            qt.QTimer.singleShot(self.CHECK_TIMER_INTERVAL, self.checkProcessOutput)
        else:
            self.cleanup()