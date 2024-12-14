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

@dataclass
class SegmentationTaskInfo:
    tempDir: str = ""
    outputSegmentationFile: str = ""
    backgroundProcess = None
    segmentationTaskListInfo = None
    sequenceItemIndex: int = 0
    resultsImported: bool = False

class EventCode(Enum):
    TASKLIST_PROCESSING_STARTED = 1
    TASK_PROCESSING_STARTED = 2
    TASK_PROCESSING_ENDED = 3
    TASK_IMPORTING_RESULTS_STARTED = 4
    TASK_IMPORTING_RESULTS_ENDED = 5
    TASKLIST_PROCESSING_ENDED = 6

class ExitCode(Enum):
    USER_CANCELLED = 1001
    DID_NOT_RUN = 1002

@dataclass
class SegmentationTaskListInfo:
    inputNodes: list = None
    outputSegmentation: slicer.vtkMRMLSegmentationNode = None
    model: str = ""
    cpu: bool = False
    waitForCompletion: bool = False
    sequenceBrowserNode: slicer.vtkMRMLSequenceBrowserNode = None
    segmentationTasks: list = field(default_factory=list) # list of SegmentationTaskInfo objects, one for each sequence item
    eventCallback: Callable = None
    customEventCallbackData: Any = None

class BackgroundProcess:
    """ Any kind of process with threads and continuous checking until stopped"""

    # Timer for checking the output of the process that is running in the background
    CHECK_TIMER_INTERVAL = 1000

    @staticmethod
    def getPSProcess(pid):
        import psutil
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None

    def __init__(self,
                 taskInfo: SegmentationTaskInfo = None,
                 logCallback: Callable = None,
                 completedCallback: Callable = None):
        self.startTime = time.time()
        self.proc = None  # subprocess.Popen object
        self.procReturnCode: ExitCode = ExitCode.DID_NOT_RUN
        self.procOutputQueue = queue.Queue()
        self.procThread = None # threading.Thread object

        self.logCallback = logCallback
        self.completedCallback = completedCallback
        self.taskInfo = taskInfo

    def __del__(self):
        self._killProcess()

    def handleSubProcessLogging(self, text):
        logging.info(text)

    def cleanup(self):
        if self.procThread:
            self.procThread.join()
        if self.completedCallback:
            self.completedCallback(self.taskInfo)
        self.proc = None
        self.procThread = None
        self.procOutputQueue = None

    def isRunning(self):
        if self.proc is None:
            return False
        psProcess = self.getPSProcess(self.proc.pid)
        if psProcess:
            return psProcess.is_running()
        else:
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
        self._killProcess()
        self._setProcReturnCode(ExitCode.USER_CANCELLED)

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
        self._setProcReturnCode(self.proc.returncode) # non-zero return code means error

    def checkProcessOutput(self):
        outputQueue = self.procOutputQueue
        while outputQueue:
            try:
                line = outputQueue.get_nowait()
                self.handleSubProcessLogging(line)
            except queue.Empty:
                break
            if self.procReturnCode != ExitCode.DID_NOT_RUN:
                self.completedCallback(self.taskInfo)
                return

        psProcess = self.getPSProcess(self.proc.pid)
        if psProcess and psProcess.is_running(): # No more outputs to process now, check again later
            qt.QTimer.singleShot(self.CHECK_TIMER_INTERVAL, self.checkProcessOutput)
        else:
            self.cleanup()

    def addLog(self, text):
        if self.logCallback:
            self.logCallback(text)

    def _setProcReturnCode(self, rcode):
        # if user cancelled, leave it at that and don't change it
        if self.procReturnCode == ExitCode.USER_CANCELLED:
            return
        self.procReturnCode = rcode



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

    def __init__(self,
                 taskInfo: SegmentationTaskInfo = None,
                 logCallback: Callable = None,
                 completedCallback: Callable = None):
        super().__init__(taskInfo, logCallback, completedCallback)
        self.hostName = "127.0.0.1"
        self.port = 8891

    def getAddressUrl(self):
        return f"http://{self.hostName}:{self.port}"

    def start(self):
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "MONAIAuto3DSegServer" / "main.py"),
            "--host", self.hostName,
            "--port", self.port
        ]

        logging.debug(f"Launching process: {cmd}")
        self.proc = slicer.util.launchConsoleProcess(cmd, useStartupEnvironment=False)
        self._startHandleProcessOutputThread()

    def handleSubProcessLogging(self, text):
        # NB: let upper level handle if it should be logged to console or UI
        self.addLog(text)


class LocalInference(BackgroundProcess):
    """ Running local inference until finished or cancelled. """

    def __init__(self,
                 taskInfo: SegmentationTaskInfo = None,
                 logCallback: Callable = None,
                 completedCallback: Callable = None,
                 waitForCompletion: bool = True):
        super().__init__(taskInfo, logCallback, completedCallback)
        self.waitForCompletion = waitForCompletion

    def run(self, cmd, additionalEnvironmentVariables=None, waitForCompletion=True):
        logging.debug(f"Launching process: {cmd}")
        self.proc = slicer.util.launchConsoleProcess(cmd, updateEnvironment=additionalEnvironmentVariables)

        if self.isRunning():
            self.addLog("Process Started")

        if waitForCompletion:
            self.logProcessOutputUntilCompleted()
            self.completedCallback(self.taskInfo)
        else:
            self._startHandleProcessOutputThread()

    def handleSubProcessLogging(self, text):
        self.addLog(text)
        logging.info(text)

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
        self._setProcReturnCode(retcode)

        if retcode != 0:
            from subprocess import CalledProcessError
            raise CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
