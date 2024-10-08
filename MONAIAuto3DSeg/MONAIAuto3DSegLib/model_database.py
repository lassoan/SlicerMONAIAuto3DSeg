import json
import logging
import os
from pathlib import Path

from MONAIAuto3DSegLib.utils import humanReadableTimeFromSec


class ModelDatabase:
    """ Retrieve model information, download and store models in model cache directory. Model information is stored
    in local Models.json.
    """

    DEFAULT_CACHE_DIR_NAME = ".MONAIAuto3DSeg"

    @property
    def defaultModel(self):
        return self.models[0]["id"]

    @property
    def models(self):
        if not self._models:
            self._models = self.loadModelsDescription()
        return self._models

    @property
    def modelsPath(self):
        modelsPath = self.fileCachePath.joinpath("models")
        modelsPath.mkdir(exist_ok=True, parents=True)
        return modelsPath

    @property
    def modelsDescriptionJsonFilePath(self):
        return os.path.join(self.moduleDir, "Resources", "Models.json")

    def __init__(self):
        self.fileCachePath = Path.home().joinpath(f"{self.DEFAULT_CACHE_DIR_NAME}")
        self.moduleDir = Path(__file__).parent.parent

        # Disabling this flag preserves input and output data after execution is completed,
        # which can be useful for troubleshooting.
        self.clearOutputFolder = True
        self._models = []

    def model(self, modelId):
        for model in self.models:
            if model["id"] == modelId:
                return model
        raise RuntimeError(f"Model {modelId} not found")

    def loadModelsDescription(self):
        modelsJsonFilePath = self.modelsDescriptionJsonFilePath
        try:
            models = []
            import json
            import re
            with open(modelsJsonFilePath) as f:
                modelsTree = json.load(f)["models"]
            for model in modelsTree:
                deprecated = False
                for version in model["versions"]:
                    url = version["url"]
                    # URL format: <path>/<filename>-v<version>.zip
                    # Example URL: https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/download/Models/17-segments-TotalSegmentator-v1.0.3.zip
                    match = re.search(r"(?P<filename>[^/]+)-v(?P<version>\d+\.\d+\.\d+)", url)
                    if match:
                        filename = match.group("filename")
                        version = match.group("version")
                    else:
                        logging.error(f"Failed to extract model id and version from url: {url}")
                    if "inputs" in model:
                        # Contains a list of dict. One dict for each input.
                        # Currently, only "title" (user-displayable name) and "namePattern" of the input are specified.
                        # In the future, inputs could have additional properties, such as name, type, optional, ...
                        inputs = model["inputs"]
                    else:
                        # Inputs are not defined, use default (single input volume)
                        inputs = [{"title": "Input volume"}]
                    segmentNames = model.get('segmentNames')
                    if not segmentNames:
                        segmentNames = "N/A"
                    models.append({
                        "id": f"{filename}-v{version}",
                        "title": model['title'],
                        "version": version,
                        "inputs": inputs,
                        "imagingModality": model["imagingModality"],
                        "description": model["description"],
                        "sampleData": model.get("sampleData"),
                        "segmentNames": model.get("segmentNames"),
                        "details":
                            f"<p><b>Model:</b> {model['title']} (v{version})"
                            f"<p><b>Description:</b> {model['description']}\n"
                            f"<p><b>Computation time on GPU:</b> {humanReadableTimeFromSec(model.get('segmentationTimeSecGPU'))}\n"
                            f"<br><b>Computation time on CPU:</b> {humanReadableTimeFromSec(model.get('segmentationTimeSecCPU'))}\n"
                            f"<p><b>Imaging modality:</b> {model['imagingModality']}\n"
                            f"<p><b>Subject:</b> {model['subject']}\n"
                            f"<p><b>Segments:</b> {', '.join(segmentNames)}",
                        "url": url,
                        "deprecated": deprecated
                        })
                    # First version is not deprecated, all subsequent versions are deprecated
                    deprecated = True
            return models
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load models description from {modelsJsonFilePath}")

    def updateModelsDescriptionJsonFilePathFromTestResults(self, modelsTestResultsJsonFilePath):
        modelsDescriptionJsonFilePath = self.modelsDescriptionJsonFilePath

        with open(modelsTestResultsJsonFilePath) as f:
            modelsTestResults = json.load(f)

        with open(modelsDescriptionJsonFilePath) as f:
            modelsDescription = json.load(f)

        for model in modelsDescription["models"]:
            title = model["title"]
            for modelTestResult in modelsTestResults:
                if modelTestResult["title"] == title:
                    for fieldName in ["segmentationTimeSecGPU", "segmentationTimeSecCPU", "segmentNames"]:
                        fieldValue = modelTestResult.get(fieldName)
                        if fieldValue:
                            model[fieldName] = fieldValue
                    break

        with open(modelsDescriptionJsonFilePath, 'w', newline="\n") as f:
            json.dump(modelsDescription, f, indent=2)

    def createModelsDir(self):
        modelsDir = self.modelsPath
        if not os.path.exists(modelsDir):
            os.makedirs(modelsDir)

    def modelPath(self, modelName):
        try:
            return self._modelPath(modelName)
        except RuntimeError:
            self.downloadModel(modelName)
            return self._modelPath(modelName)

    def _modelPath(self, modelName):
        modelRoot = self.modelsPath.joinpath(modelName)
        # find labels.csv file within the modelRoot folder and subfolders
        for path in Path(modelRoot).rglob("labels.csv"):
            return path.parent
        raise RuntimeError(f"Model {modelName} path not found")

    def deleteAllModels(self):
        if self.modelsPath.exists():
            import shutil
            shutil.rmtree(self.modelsPath)

    def downloadAllModels(self):
        for model in self.models:
            slicer.app.processEvents()
            self.downloadModel(model["id"])

    def downloadModel(self, modelName):
        url = self.model(modelName)["url"]
        import zipfile
        import requests
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            tempDir = Path(td)
            modelDir = self.modelsPath.joinpath(modelName)
            Path(modelDir).mkdir(exist_ok=True)
            modelZipFile = tempDir.joinpath("autoseg3d_model.zip")
            logging.info(f"Downloading model '{modelName}' from {url}...")
            logging.debug(f"Downloading from {url} to {modelZipFile}...")
            try:
                with open(modelZipFile, 'wb') as f:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        reporting_increment_percent = 1.0
                        last_reported_download_percent = -reporting_increment_percent
                        downloaded_size = 0
                        for chunk in r.iter_content(chunk_size=8192 * 16):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            downloaded_percent = 100.0 * downloaded_size / total_size
                            if downloaded_percent - last_reported_download_percent > reporting_increment_percent:
                                logging.info(
                                    f"Downloading model: {downloaded_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({downloaded_percent:.1f}%)")
                                last_reported_download_percent = downloaded_percent

                logging.info(f"Download finished. Extracting to {modelDir}...")
                with zipfile.ZipFile(modelZipFile, 'r') as zip_f:
                    zip_f.extractall(modelDir)
            except Exception as e:
                raise e
            finally:
                if self.clearOutputFolder:
                    logging.info("Cleaning up temporary model download folder...")
                    if os.path.isdir(tempDir):
                        import shutil
                        shutil.rmtree(tempDir)
                else:
                    logging.info(f"Not cleaning up temporary model download folder: {tempDir}")
