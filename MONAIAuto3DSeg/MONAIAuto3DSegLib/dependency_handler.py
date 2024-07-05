import sys

import shutil
import subprocess
import logging

from MONAIAuto3DSegLib.constants import APPLICATION_NAME
logger = logging.getLogger(APPLICATION_NAME)


from abc import ABC, abstractmethod


class DependenciesBase(ABC):

    minimumTorchVersion = "1.12"

    def __init__(self):
        self.dependenciesInstalled = False  # we don't know yet if dependencies have been installed

    @abstractmethod
    def installedMONAIPythonPackageInfo(self):
        pass

    @abstractmethod
    def setupPythonRequirements(self, upgrade=False):
        pass


class LocalPythonDependencies(DependenciesBase):

    def installedMONAIPythonPackageInfo(self):
        versionInfo = subprocess.check_output([sys.executable, "-m", "pip", "show", "MONAI"]).decode()
        return versionInfo

    def _checkModuleInstalled(self, moduleName):
      try:
        import importlib
        importlib.import_module(moduleName)
        return True
      except ModuleNotFoundError:
        return False

    def setupPythonRequirements(self, upgrade=False):
        def install(package):
          subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        logger.info("Initializing PyTorch...")

        packageName = "torch"
        if not self._checkModuleInstalled(packageName):
          logger.info("PyTorch Python package is required. Installing... (it may take several minutes)")
          install(packageName)
          if not self._checkModuleInstalled(packageName):
            raise ValueError("pytorch needs to be installed to use this module.")
        else:  # torch is installed, check version
            from packaging import version
            import torch
            if version.parse(torch.__version__) < version.parse(self.minimumTorchVersion):
                raise ValueError(f"PyTorch version {torch.__version__} is not compatible with this module."
                                 + f" Minimum required version is {self.minimumTorchVersion}. You can use 'PyTorch Util' module to install PyTorch"
                                 + f" with version requirement set to: >={self.minimumTorchVersion}")

        logger.info("Initializing MONAI...")
        monaiInstallString = "monai[fire,pyyaml,nibabel,pynrrd,psutil,tensorboard,skimage,itk,tqdm]>=1.3"
        if upgrade:
            monaiInstallString += " --upgrade"
        install(monaiInstallString)

        self.dependenciesInstalled = True
        logger.info("Dependencies are set up successfully.")


class RemotePythonDependencies(DependenciesBase):

    def installedMONAIPythonPackageInfo(self, server_address):
        if not server_address:
            return []
        else:
            import json
            import requests
            response = requests.get(server_address + "/monaiinfo")
            json_data = json.loads(response.text)
            return json_data

    def setupPythonRequirements(self, upgrade=False):
        logger.error("No permission to update remote python packages. Please contact developer.")


class SlicerPythonDependencies(DependenciesBase):

    def installedMONAIPythonPackageInfo(self):
        versionInfo = subprocess.check_output([shutil.which("PythonSlicer"), "-m", "pip", "show", "MONAI"]).decode()
        return versionInfo

    def setupPythonRequirements(self, upgrade=False):
        # Install PyTorch
        try:
            import PyTorchUtils
        except ModuleNotFoundError as e:
            raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")

        logger.info("Initializing PyTorch...")

        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            logger.info("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement=f">={self.minimumTorchVersion}")
            if torch is None:
                raise ValueError("PyTorch extension needs to be installed to use this module.")
        else:  # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(self.minimumTorchVersion):
                raise ValueError(f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module."
                                 + f" Minimum required version is {self.minimumTorchVersion}. You can use 'PyTorch Util' module to install PyTorch"
                                 + f" with version requirement set to: >={self.minimumTorchVersion}")

        # Install MONAI with required components
        logger.info("Initializing MONAI...")
        # Specify minimum version 1.3, as this is a known working version (it is possible that an earlier version works, too).
        # Without this, for some users monai-0.9.0 got installed, which failed with this error:
        # "ImportError: cannot import name ‘MetaKeys’ from 'monai.utils'"
        monaiInstallString = "monai[fire,pyyaml,nibabel,pynrrd,psutil,tensorboard,skimage,itk,tqdm]>=1.3"
        if upgrade:
            monaiInstallString += " --upgrade"
        import slicer
        slicer.util.pip_install(monaiInstallString)

        self.dependenciesInstalled = True
        logger.info("Dependencies are set up successfully.")