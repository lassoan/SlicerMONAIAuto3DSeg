import logging
import re
import subprocess
import sys
from pathlib import Path

import qt
import requests
import slicer
from MRMLCorePython import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

from .Signal import Signal


class MonaiBundleLogic:

    def __init__(self):
        self._bundleRootDir = Path('')
        self._tmpDir = qt.QTemporaryDir()
        self.metadataParser = None
        self.bundleLoaded = Signal()

    @property
    def bundleRootDir(self):
        return self._bundleRootDir

    @bundleRootDir.setter
    def bundleRootDir(self, newRootDir):
        self._bundleRootDir = Path(newRootDir)

        from monai.bundle import ConfigParser
        self.metadataParser = ConfigParser()
        try:
            self.metadataParser.read_config(self.configDir.joinpath("metadata.json"))
        except FileNotFoundError:
            self.metadataParser = None

    @property
    def configDir(self):
        return self.bundleRootDir.joinpath("configs")

    def downloadBundleFromModelZoo(self, modelName, version, bundleDir: Path):
        command = [
            sys.executable, "-m", "monai.bundle", "download",
            "--name", modelName,
            "--version", version,
            "--source", "github",
            "--repo", "Project-MONAI/model-zoo/hosting_storage_v1",
            "--bundle_dir", bundleDir
        ]
        subprocess.run(command)

    def getAvailableModelsFromZoo(self):
        """Get all the bundles available on the MONAI model zoo

        :return: dict mapping each model to the available versions in semantic format:
         {modelName: ['x1.y1.z1', 'x2.y2.z2', ...]}"""
        # URL for the model zoo repo where the bundles are stored as assets
        url = "https://api.github.com/repos/Project-MONAI/model-zoo/releases/tags/hosting_storage_v1"
        response = requests.get(url)
        print("Retrieving available models from model zoo...")

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")

        content = response.json()
        if not content["assets"]:
            print("No model found in model zoo")
            return []

        model_versions = {}
        for bundle in content["assets"]:
            bundleName = bundle['name']
            match = re.match(r"(.+)_v(\d+\.\d+\.\d+)\.zip", bundleName)
            if not match:
                print(f"- {bundleName} (Unable to parse name and version)")
                continue
            model_name = match.group(1)
            version = match.group(2)
            if model_name not in model_versions:
                model_versions[model_name] = [version]
            else:
                model_versions[model_name].append(version)

        return model_versions

    def setDevice(self, deviceName: str):
        import torch
        if deviceName == 'auto':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(deviceName)
        self.configParser["device"] = device

    def loadBundleConfig(self, configFilename: Path):
        """Load the bundle from a .json config file"""
        from monai.bundle import ConfigParser

        configFilePath = self.configDir.joinpath(configFilename)

        self.configParser = ConfigParser()
        self.configParser.read_config(configFilePath.as_posix())
        self.configParser["bundle_root"] = self.bundleRootDir.as_posix()
        self.bundleLoaded.emit()

    def runInference(self, volumeNode: vtkMRMLScalarVolumeNode) -> vtkMRMLSegmentationNode:
        """Run the segmentation on a slicer volumeNode, get the result as a segmentationNode"""
        self._prepareInferenceDir(volumeNode)
        self._runMonaiBundleInference()
        segmentationNode = slicer.util.loadSegmentation(self._outFile)
        self.renameSegments(segmentationNode)
        return segmentationNode

    def renameSegments(self, segmentationNode: vtkMRMLSegmentationNode):
        """Rename the segments using the labels from the metadata"""
        segmentNames = self.metadataParser["network_data_format"]["outputs"]["pred"]["channel_def"]

        for index, segmentID in enumerate(segmentationNode.GetSegmentation().GetSegmentIDs()):
            # needs to be offset by one to avoid background segment (not loaded in slicer)
            segmentationNode.GetSegmentation().GetSegment(segmentID).SetName(segmentNames[str(index + 1)])

    def _prepareInferenceDir(self, volumeNode):
        self._tmpDir.remove()
        self._outDir.mkdir(parents=True)
        self._inDir.mkdir(parents=True)

        volumePath = self._inDir.joinpath("volume.nii.gz")
        assert slicer.util.exportNode(volumeNode, volumePath)
        assert volumePath.exists(), "Failed to export volume for segmentation."

    @property
    def _outFile(self) -> str:
        return next(file for file in self._outDir.rglob("*.nii*")).as_posix()

    @property
    def _outDir(self):
        return Path(self._tmpDir.path()).joinpath("output")

    @property
    def _inDir(self):
        return Path(self._tmpDir.path()).joinpath("input")

    def _runMonaiBundleInference(self):
        import torch

        try:
            with torch.no_grad():
                dataset_dir = self._inDir
                output_dir = self._outDir

                # Update config vars
                self.configParser["dataset_dir"] = dataset_dir.as_posix()
                self.configParser["output_dir"] = output_dir.as_posix()
                self.configParser["datalist"] = [p.as_posix() for p in dataset_dir.glob("*.nii*")]
                self.configParser["window_device"] = "cpu"
                # avoid hanging in windows
                self.configParser["dataloader"]["num_workers"] = 0
                assert len(self.configParser["datalist"]) > 0, "Empty data list as input"
                dataloader = self.configParser.get_parsed_content("dataloader")
                assert len(dataloader) > 0
                evaluator = self.configParser.get_parsed_content("evaluator")
                evaluator.run()
        except RuntimeError as e:
            logging.error(e)
        finally:
            torch.cuda.empty_cache()
