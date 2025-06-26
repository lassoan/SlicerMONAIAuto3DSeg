from pathlib import Path

import ctk
import qt
import slicer

from .MonaiBundleLogic import MonaiBundleLogic
from .Utils import warningMessageBox


class MonaiBundleWidget(qt.QWidget):
    """This Widget loads and run a monai bundle"""

    def __init__(self, parent=None):
        super(MonaiBundleWidget, self).__init__(parent)
        self.logic = MonaiBundleLogic()
        self._resourceDir = Path(__file__).parent.parent.joinpath("Resources")
        self._defaultBundleStorageDir = Path(
            qt.QSettings().value("MonaiBundleRunner/bundleStorageDir") or self._resourceDir.joinpath("bundles"))

        uiWidget = slicer.util.loadUI(self._resourceDir.joinpath("UI/MONAIBundle.ui"))
        layout = qt.QVBoxLayout(self)
        layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.setupLayout()
        self.modelVersionsDict = {}

    def setupLayout(self):
        self.ui.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ui.inputSelector.selectNodeUponCreation = False
        self.ui.inputSelector.addEnabled = False
        self.ui.inputSelector.showHidden = False
        self.ui.inputSelector.removeEnabled = False
        self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateRunEnabled)

        self.ui.updateModelsFromZooButton.connect("clicked(bool)", self.onUpdateModelsFromZooClicked)
        self.ui.modelZooSelectionBox.currentIndexChanged.connect(self.onSelectedModelChanged)
        self.ui.downloadModelFromZooButton.connect("clicked(bool)", self.downloadBundleFromModelZoo)

        self.ui.bundleStorageDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.bundleStorageDirSelector.settingKey = "MonaiBundleRunner/bundleStorageDir"
        self.ui.bundleStorageDirSelector.currentPathChanged.connect(self.onBundleStorageDirChanged)
        self.ui.bundleStorageDirSelector.setCurrentPath(self._defaultBundleStorageDir.as_posix())

        self.ui.bundleRootDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.bundleRootDirSelector.settingKey = "MonaiBundleRunner/bundleRootDir"
        self.ui.bundleRootDirSelector.currentPathChanged.connect(self.onBundleRootDirChanged)

        self.ui.deviceSelectionBox.currentIndexChanged.connect(self.onDeviceSelectionChanged)

        self.ui.runButton.clicked.connect(self.run)

        # Signals
        self.logic.bundleLoaded.connect(self.onBundleLoaded)

    def onUpdateModelsFromZooClicked(self):
        """Retrieve models names and versions from Model Zoo."""
        self.ui.modelZooSelectionBox.clear()
        self.modelVersionsDict = self.logic.getAvailableModelsFromZoo()
        # for now, only segmentation models are supported
        segmentationModels = [name for name in list(self.modelVersionsDict.keys()) if "segmentation" in name]
        self.ui.modelZooSelectionBox.addItems(segmentationModels)

    def onSelectedModelChanged(self):
        """Update the versions available for the selected model, select the latest by default"""
        self.ui.modelVersionSelectionBox.clear()
        versions = self.modelVersionsDict[self.ui.modelZooSelectionBox.currentText]
        self.ui.modelVersionSelectionBox.addItems(versions)
        self.ui.modelVersionSelectionBox.setCurrentText(max(versions))

    def updateRunEnabled(self):
        self.ui.runButton.setEnabled(
            self.ui.inputSelector.currentNode() is not None and self.configFilename.is_file())

    @property
    def bundleRootDir(self) -> Path:
        return Path(self.ui.bundleRootDirSelector.currentPath)

    @property
    def configFilename(self) -> Path:
        return self.bundleRootDir.joinpath("configs", "inference.json")

    def updateBundleDescription(self):
        metadata = self.logic.metadataParser
        if metadata is None:
            slicer.util.warningDisplay("Could not load the bundle from the provided folder")
            return
        self.ui.bundleDescriptionTextEdit.setPlainText(
            f"{metadata['name']} v{metadata['version']}\n"
            f"{metadata['authors']} - {metadata['copyright']}\n\n"
            f"{metadata['description']}\n\n"
            f"Requirements :\n"
            f"monai : {metadata['monai_version']}\n"
            f"torch : {metadata['pytorch_version']}\n"
            f"numpy : {metadata['numpy_version']}"
        )

    def onBundleRootDirChanged(self):
        self.logic.bundleRootDir = Path(self.bundleRootDir)
        try:
            self.logic.loadBundleConfig(self.configFilename)
        except FileNotFoundError:
            warningMessageBox("File not found",
                              f"Could not locate {self.configFilename.name} in the configs folder "
                              f"of the specified bundle {self.bundleRootDir}")

    def onBundleStorageDirChanged(self):
        self.bundleStorageDir = Path(self.ui.bundleStorageDirSelector.currentPath)
        qt.QSettings().setValue("MonaiBundleRunner/bundleStorageDir", self.bundleStorageDir.as_posix())

    def onDeviceSelectionChanged(self):
        self.logic.setDevice(self.ui.deviceSelectionBox.currentText)

    def onBundleLoaded(self):
        self.updateBundleDescription()
        self.ui.deviceSelectionBox.setEnabled(True)
        self.updateRunEnabled()

    def downloadBundleFromModelZoo(self):
        modelName = self.ui.modelZooSelectionBox.currentText
        modelVersion = self.ui.modelVersionSelectionBox.currentText
        progressText = f"Downloading {modelName}..."
        progressDialog = slicer.util.createProgressDialog(labelText=progressText, minimum=0, maximum=0, value=0)
        progressDialog.setCancelButton(None)
        slicer.app.processEvents()
        self.logic.downloadBundleFromModelZoo(modelName, modelVersion, self.bundleStorageDir)
        self.ui.bundleRootDirSelector.setCurrentPath(self.bundleStorageDir.joinpath(modelName).as_posix())
        progressDialog.close()

    def run(self):
        self.ui.runButton.setEnabled(False)
        self.ui.runButton.setText("Running...")
        progressText = f"Running {self.configFilename.name}..."
        progressDialog = slicer.util.createProgressDialog(labelText=progressText, minimum=0, maximum=0, value=0)
        progressDialog.setCancelButton(None)
        slicer.app.processEvents()
        self.logic.runInference(self.ui.inputSelector.currentNode())
        progressDialog.close()
        self.ui.runButton.setEnabled(True)
        self.ui.runButton.setText("Run")
