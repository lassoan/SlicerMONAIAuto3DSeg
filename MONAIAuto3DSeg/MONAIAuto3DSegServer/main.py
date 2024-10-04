# pip install python-multipart
# pip install fastapi
# pip install "uvicorn[standard]"

# usage: uvicorn main:app --reload --host example.com --port 8891
# usage: uvicorn main:app --reload --host localhost --port 8891


import os
import logging
import sys
from pathlib import Path

paths = [str(Path(__file__).parent.parent)]
for path in paths:
    if not path in sys.path:
        sys.path.insert(0, path)

from MONAIAuto3DSegLib.model_database import ModelDatabase
from MONAIAuto3DSegLib.dependency_handler import DependenciesBase

import shutil
import asyncio
import subprocess
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.background import BackgroundTasks


class LocalPythonDependencies(DependenciesBase):
    """ Dependency handler when running locally (not within 3D Slicer)

    code:

        from MONAIAuto3DSegLib.dependency_handler import LocalPythonDependencies
        dependencies = LocalPythonDependencies()
        dependencies.installedMONAIPythonPackageInfo()

        # dependencies.setupPythonRequirements(upgrade=True)
    """

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

        logging.debug("Initializing PyTorch...")

        packageName = "torch"
        if not self._checkModuleInstalled(packageName):
          logging.debug("PyTorch Python package is required. Installing... (it may take several minutes)")
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

        logging.debug("Initializing MONAI...")
        monaiInstallString = "monai[fire,pyyaml,nibabel,pynrrd,psutil,tensorboard,skimage,itk,tqdm]>=1.3"
        if upgrade:
            monaiInstallString += " --upgrade"
        install(monaiInstallString)

        self.dependenciesInstalled = True
        logging.debug("Dependencies are set up successfully.")


app = FastAPI()
modelDB = ModelDatabase()
dependencyHandler = LocalPythonDependencies()


def upload(file, session_dir, identifier):
    extension = "".join(Path(file.filename).suffixes)
    file_location = f"{session_dir}/{identifier}{extension}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return file_location


@app.get("/monaiinfo")
def monaiInfo():
    return dependencyHandler.installedMONAIPythonPackageInfo()


@app.get("/models")
async def models():
    return modelDB.models


@app.get("/modelinfo")
async def getModelInfo(id: str):
    return modelDB.model(id)


@app.get("/labelDescriptions")
def getLabelsFile(id: str):
    return FileResponse(modelDB.modelPath(id).joinpath("labels.csv"), media_type = 'application/octet-stream', filename="labels.csv")


@app.post("/infer")
async def infer(
    background_tasks: BackgroundTasks,
    image_file: UploadFile,
    model_name: str,
    image_file_2: UploadFile = None,
    image_file_3: UploadFile = None,
    image_file_4: UploadFile = None
):
    import tempfile
    session_dir = tempfile.mkdtemp(dir=tempfile.gettempdir())
    background_tasks.add_task(shutil.rmtree, session_dir)

    logging.debug(session_dir)
    inputFiles = list()
    inputFiles.append(upload(image_file, session_dir, "image_file"))
    if image_file_2:
        inputFiles.append(upload(image_file_2, session_dir, "image_file_2"))
    if image_file_3:
        inputFiles.append(upload(image_file_3, session_dir, "image_file_3"))
    if image_file_4:
        inputFiles.append(upload(image_file_4, session_dir, "image_file_4"))

    # logging.info("Input Files: ", inputFiles)

    outputSegmentationFile = f"{session_dir}/output-segmentation.nrrd"

    modelPath = modelDB.modelPath(model_name)
    modelPtFile = modelPath.joinpath("model.pt")

    assert os.path.exists(modelPtFile)

    dependencyHandler.setupPythonRequirements()

    moduleDir = Path(__file__).parent.parent
    inferenceScriptPyFile = os.path.join(moduleDir, "Scripts", "auto3dseg_segresnet_inference.py")
    auto3DSegCommand = [sys.executable, str(inferenceScriptPyFile),
                        "--model-file", str(modelPtFile),
                        "--image-file", inputFiles[0],
                        "--result-file", str(outputSegmentationFile)]
    for inputIndex in range(1, len(inputFiles)):
        auto3DSegCommand.append(f"--image-file-{inputIndex + 1}")
        auto3DSegCommand.append(inputFiles[inputIndex])

    try:
        # logging.debug(auto3DSegCommand)
        proc = await asyncio.create_subprocess_shell(" ".join(auto3DSegCommand))
        await proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, " ".join(auto3DSegCommand))
        return FileResponse(outputSegmentationFile, media_type='application/octet-stream', background=background_tasks)
    except Exception as e:
        shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Failed to run CMD command: {str(e)}")


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="MONAIAuto3DSeg server")
    parser.add_argument("-ip", "--host", type=str, metavar="PATH", required=False, default="localhost", help="host name")
    parser.add_argument("-p", "--port", type=int, metavar="PATH", required=True, help="port")

    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("main:app", host=args.host, port=args.port, reload=True, log_level="debug")


if __name__ == "__main__":
    main(sys.argv[1:])
