import subprocess

import slicer

import sys
import logging
import queue
import threading

import qt
from pathlib import Path
from typing import Callable, Any
import time
from dataclasses import dataclass, field

from enum import Enum


class ExitCode(Enum):
    USER_CANCELLED = 1001
    DID_NOT_RUN = 1002


@dataclass
class ProcessInfo:
    proc: subprocess.Popen = None
    startTime: float = field(default_factory=time.time)
    procReturnCode: ExitCode = ExitCode.DID_NOT_RUN
    procOutputQueue: queue.Queue = queue.Queue()
    procThread: threading.Thread = None


class SegmentationProcessInfo(ProcessInfo):
    tempDir: str = ""
    inputNodes: list = None
    outputSegmentation: slicer.vtkMRMLSegmentationNode = None
    outputSegmentationFile: str = ""
    model: str = ""
    customData: Any = None


class BackgroundProcess:
    """ Any kind of process with threads and continuous checking until stopped"""

    # Timer for checking the output of the process that is running in the background
    CHECK_TIMER_INTERVAL = 1000

    @property
    def proc(self):
        return self.processInfo.proc

    @proc.setter
    def proc(self, proc):
        self.processInfo.proc = proc

    @property
    def procThread(self):
        return self.processInfo.procThread

    @procThread.setter
    def procThread(self, procThread):
        self.processInfo.procThread = procThread

    @property
    def procOutputQueue(self):
        return self.processInfo.procOutputQueue

    @procOutputQueue.setter
    def procOutputQueue(self, procOutputQueue):
        self.processInfo.procOutputQueue = procOutputQueue

    @staticmethod
    def getPSProcess(pid):
        import psutil
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None

    def __init__(self, processInfo: ProcessInfo = None, logCallback: Callable = None, completedCallback: Callable = None):
        self.processInfo = processInfo if processInfo else ProcessInfo()
        self.logCallback = logCallback
        self.completedCallback = completedCallback

        # NB: making sure that the following values were not set previously
        self.proc = None
        self.procThread = None
        self.procOutputQueue = None

    def __del__(self):
        self._killProcess()

    def cleanup(self):
        if self.procThread:
            self.procThread.join()
        if self.completedCallback:
            if self.proc.returncode not in [-9, 0]: # killed or stopped cleanly
                self.addLog(self._err)

            self.completedCallback(self.processInfo)
        self.proc = None
        self.procThread = None
        self.procOutputQueue = None

    def isRunning(self):
        if self.proc is not None:
            psProcess = self.getPSProcess(self.proc.pid)
            if psProcess:
                return psProcess.is_running()
        return False

    def _killProcess(self):
        if not self.proc:
            return
        psProcess = self.getPSProcess(self.proc.pid) # proc.kill() does not work, that would only stop the launcher
        if not psProcess:
            return
        for psChildProcess in psProcess.children(recursive=True):
            psChildProcess.kill()
        if psProcess.is_running():
            psProcess.kill()

    def stop(self):
        logging.info("Cancel is requested.")
        self._killProcess()
        self.processInfo.procReturnCode = ExitCode.USER_CANCELLED

    def _startHandleProcessOutputThread(self):
        self.procOutputQueue = queue.Queue()
        self.procThread = threading.Thread(target=self._handleProcessOutputThreadProcess)
        self.procThread.start()
        self.checkProcessOutput()

    def _handleProcessOutputThreadProcess(self):
        while True:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                text = line.rstrip()
                self.procOutputQueue.put(text)
            except UnicodeDecodeError as e:
                pass
        self.proc.wait()
        self.processInfo.procReturnCode = self.proc.returncode # non-zero return code means error

    def checkProcessOutput(self):
        outputQueue = self.procOutputQueue
        while outputQueue:
            if self.processInfo.procReturnCode != ExitCode.DID_NOT_RUN:
                self.completedCallback(self.processInfo)
                return
            try:
                line = outputQueue.get_nowait()
                logging.info(line)
            except queue.Empty:
                break

        psProcess = self.getPSProcess(self.proc.pid)
        if psProcess and psProcess.is_running(): # No more outputs to process now, check again later
            qt.QTimer.singleShot(self.CHECK_TIMER_INTERVAL, self.checkProcessOutput)
        else:
            self.cleanup()

    def addLog(self, text):
        if self.logCallback:
            self.logCallback(text)


class InferenceServer(BackgroundProcess):
    """ Web server class to be used from 3D Slicer. Upon starting the server with the given cmd command, it will be
        checked every {CHECK_TIMER_INTERVAL} milliseconds for process status/outputs.

        code:

            cmd = [sys.executable, "main.py", "--host", hostName, "--port", port]

            from MONAIAuto3DSegLib.process import InferenceServer
            server = InferenceServer(completedCallback=...)
            server.start()

            ...

            server.stop()

    """

    def __init__(self, processInfo: ProcessInfo = None, logCallback: Callable = None,
                 completedCallback: Callable = None):
        super().__init__(processInfo, logCallback, completedCallback)
        self.hostName = "127.0.0.1"
        self.port = 8891

    def getAddressUrl(self):
        return f"http://{self.hostName}:{self.port}"

    def start(self):
        cmd = [
            sys.executable,
            Path(__file__).parent.parent / "MONAIAuto3DSegServer" / "main.py", "--host",
            self.hostName, "--port", self.port
        ]

        logging.debug(f"Launching process: {cmd}")
        self.proc = slicer.util.launchConsoleProcess(cmd, useStartupEnvironment=False)

        if self.isRunning():
            self.addLog("Server Started")

        self._startHandleProcessOutputThread()


class LocalInference(BackgroundProcess):
    """ Running local inference until finished or cancelled. """

    def __init__(self, processInfo: SegmentationProcessInfo, logCallback: Callable = None,
                 completedCallback: Callable = None, waitForCompletion: bool = True):
        super().__init__(processInfo, logCallback, completedCallback)
        self.waitForCompletion = waitForCompletion

    def run(self, cmd, additionalEnvironmentVariables=None, waitForCompletion=True):
        logging.debug(f"Launching process: {cmd}")
        self.proc = slicer.util.launchConsoleProcess(cmd, updateEnvironment=additionalEnvironmentVariables)

        if self.isRunning():
            self.addLog("Process Started")

        if waitForCompletion:
            self.logProcessOutputUntilCompleted()
            self.completedCallback(self.processInfo)
        else:
            self._startHandleProcessOutputThread()

    def logProcessOutputUntilCompleted(self):
        # Wait for the process to end and forward output to the log
        proc = self.proc
        while True:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                logging.info(line.rstrip())
            except UnicodeDecodeError as e:
                # Code page conversion happens because `universal_newlines=True` sets process output to text mode,
                # and it fails because probably system locale is not UTF8. We just ignore the error and discard the string,
                # as we only guarantee correct behavior if an UTF8 locale is used.
                pass
        proc.wait()
        retcode = proc.returncode
        self.processInfo.procReturnCode = retcode

        if retcode != 0:
            from subprocess import CalledProcessError
            raise CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
