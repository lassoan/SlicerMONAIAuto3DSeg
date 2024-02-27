import logging
import os
import re

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# MONAIAuto3DSeg
#
#

class MONAIAuto3DSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MONAI Auto3DSeg"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Andras Lasso (PerkLab, Queen's University)", "Andres Diaz-Pinto (NVIDIA & KCL)", "Rudolf Bumm (KSGR Switzerland)"]
        self.parent.helpText = """
3D Slicer extension for segmentation using MONAI Auto3DSeg AI model.
See more information in the <a href="https://github.com/lassoan/SlicerMONAIAuto3DSeg">extension documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Andras Lasso (PerkLab, Queen's University).
The module uses <a href="https://github.com/Project-MONAI/tutorials/blob/main/MONAIAuto3DSeg/README.md">MONAI Auto3DSeg model</a>.
"""
        slicer.app.connect("startupCompleted()", self.configureDefaultTerminology)

    def configureDefaultTerminology(self):
        moduleDir = os.path.dirname(self.parent.path)
        terminologyFilePath = os.path.join(moduleDir, "Resources", "SegmentationCategoryTypeModifier-MONAIAuto3DSeg.term.json")
        tlogic = slicer.modules.terminologies.logic()
        self.terminologyName = tlogic.LoadTerminologyFromFile(terminologyFilePath)

#
# MONAIAuto3DSegWidget
#

class MONAIAuto3DSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    PROCESSING_IDLE = 0
    PROCESSING_STARTING = 1
    PROCESSING_IN_PROGRESS = 2
    PROCESSING_IMPORT_RESULTS = 3
    PROCESSING_CANCEL_REQUESTED = 4

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self._processingState = MONAIAuto3DSegWidget.PROCESSING_IDLE
        self._segmentationProcessInfo = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MONAIAuto3DSeg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MONAIAuto3DSegLogic()
        self.logic.logCallback = self.addLog
        self.logic.processingCompletedCallback = self.onProcessingCompleted
        self.logic.startResultImportCallback = self.onProcessImportStarted
        self.logic.endResultImportCallback = self.onProcessImportEnded

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.cpuCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.showAllModelsCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.useStandardSegmentNamesCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.ui.modelComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.ui.segmentationShow3DButton.setSegmentationNode)

        # Buttons
        self.ui.packageInfoUpdateButton.connect("clicked(bool)", self.onPackageInfoUpdate)
        self.ui.packageUpgradeButton.connect("clicked(bool)", self.onPackageUpgrade)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.browseToModelsFolderButton.connect("clicked(bool)", self.onBrowseModelsFolder)
        self.ui.deleteAllModelsButton.connect("clicked(bool)", self.onClearModelsFolder)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        self.updateGUIFromParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
          self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        import qt

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        showAllModels = self._parameterNode.GetParameter("showAllModels") == "true"
        self.ui.modelComboBox.clear()
        for model in self.logic.models:
            deprecated = model.get("deprecated")
            modelTitle = model["title"]
            if deprecated:
                if showAllModels:
                    modelTitle += " -- deprecated"
                else:
                    # Do not show deprecated models
                    continue
            itemIndex = self.ui.modelComboBox.count
            self.ui.modelComboBox.addItem(modelTitle, model["id"])
            self.ui.modelComboBox.setItemData(itemIndex, model.get("description"), qt.Qt.ToolTipRole)

        # Update node selectors and sliders
        self.ui.inputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        model = self._parameterNode.GetParameter("Model")
        self.ui.modelComboBox.setCurrentIndex(self.ui.modelComboBox.findData(model))
        self.ui.cpuCheckBox.checked = self._parameterNode.GetParameter("CPU") == "true"
        self.ui.showAllModelsCheckBox.checked = showAllModels
        self.ui.useStandardSegmentNamesCheckBox.checked = self._parameterNode.GetParameter("UseStandardSegmentNames") == "true"
        self.ui.outputSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSegmentation"))

        inputVolume = self._parameterNode.GetNodeReference("InputVolume")

        state = self._processingState
        if state == MONAIAuto3DSegWidget.PROCESSING_IDLE:
            self.ui.applyButton.text = "Apply"
            if inputVolume:
                self.ui.applyButton.toolTip = "Start segmentation"
                self.ui.applyButton.enabled = True
            else:
                self.ui.applyButton.toolTip = "Select input volume"
                self.ui.applyButton.enabled = False
        elif state == MONAIAuto3DSegWidget.PROCESSING_STARTING:
            self.ui.applyButton.text = "Starting..."
            self.ui.applyButton.toolTip = "Please wait while the segmentation is being initialized"
            self.ui.applyButton.enabled = False
        elif state == MONAIAuto3DSegWidget.PROCESSING_IN_PROGRESS:
            self.ui.applyButton.text = "Cancel"
            self.ui.applyButton.toolTip = "Cancel in-progress segmentation"
            self.ui.applyButton.enabled = True
        elif state == MONAIAuto3DSegWidget.PROCESSING_IMPORT_RESULTS:
            self.ui.applyButton.text = "Importing results..."
            self.ui.applyButton.toolTip = "Please wait while the segmentation result is being imported"
            self.ui.applyButton.enabled = False
        elif state == MONAIAuto3DSegWidget.PROCESSING_CANCEL_REQUESTED:
            self.ui.applyButton.text = "Cancelling..."
            self.ui.applyButton.toolTip = "Please wait for the segmentation to be cancelled"
            self.ui.applyButton.enabled = False

        if inputVolume:
            self.ui.outputSegmentationSelector.baseName = inputVolume.GetName() + " segmentation"

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        self._parameterNode.SetParameter("Model", self.ui.modelComboBox.currentData)
        self._parameterNode.SetParameter("CPU", "true" if self.ui.cpuCheckBox.checked else "false")
        self._parameterNode.SetParameter("showAllModels", "true" if self.ui.showAllModelsCheckBox.checked else "false")
        self._parameterNode.SetParameter("UseStandardSegmentNames", "true" if self.ui.useStandardSegmentNamesCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputSegmentation", self.ui.outputSegmentationSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def addLog(self, text):
        """Append text to log window
        """
        self.ui.statusLabel.appendPlainText(text)
        slicer.app.processEvents()  # force update

    def setProcessingState(self, state):
        self._processingState = state
        self.updateGUIFromParameterNode()
        slicer.app.processEvents()

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """

        if self._processingState == MONAIAuto3DSegWidget.PROCESSING_IDLE:
            self.onApply()
        else:
            self.onCancel()

    def onApply(self):
        self.ui.statusLabel.plainText = ""

        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_STARTING)

        if not self.logic.dependenciesInstalled:
            with slicer.util.tryWithErrorDisplay("Failed to install required dependencies.", waitCursor=True):
                self.logic.setupPythonRequirements()

        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Create new segmentation node, if not selected yet
            if not self.ui.outputSegmentationSelector.currentNode():
                self.ui.outputSegmentationSelector.addNode()

            self.logic.useStandardSegmentNames = self.ui.useStandardSegmentNamesCheckBox.checked

            # Compute output
            self._segmentationProcessInfo = self.logic.process(self.ui.inputVolumeSelector.currentNode(), self.ui.outputSegmentationSelector.currentNode(),
                self.ui.modelComboBox.currentData, self.ui.cpuCheckBox.checked, waitForCompletion=False)

            self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IN_PROGRESS)

    def onCancel(self):
        with slicer.util.tryWithErrorDisplay("Failed to cancel processing.", waitCursor=True):
            self.logic.cancelProcessing(self._segmentationProcessInfo)
            self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_CANCEL_REQUESTED)

    def onProcessImportStarted(self, customProcessData):
        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IMPORT_RESULTS)
        import qt
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

    def onProcessImportEnded(self, customProcessData):
        import qt
        qt.QApplication.restoreOverrideCursor()
        slicer.app.processEvents()

    def onProcessingCompleted(self, returnCode, customProcessData):
        self.ui.statusLabel.appendPlainText("\nProcessing finished.")
        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IDLE)
        self._segmentationProcessInfo = None

    def onPackageInfoUpdate(self):
        self.ui.packageInfoTextBrowser.plainText = ""
        with slicer.util.tryWithErrorDisplay("Failed to get MONAI package version information", waitCursor=True):
            self.ui.packageInfoTextBrowser.plainText = self.logic.installedMONAIPythonPackageInfo().rstrip()

    def onPackageUpgrade(self):
        with slicer.util.tryWithErrorDisplay("Failed to upgrade MONAI", waitCursor=True):
            self.logic.setupPythonRequirements(upgrade=True)
        self.onPackageInfoUpdate()
        if not slicer.util.confirmOkCancelDisplay(f"This MONAI update requires a 3D Slicer restart.","Press OK to restart."):
            raise ValueError("Restart was cancelled.")
        else:
            slicer.util.restart()

    def onBrowseModelsFolder(self):
        import qt
        self.logic.createModelsDir()
        qt.QDesktopServices().openUrl(qt.QUrl.fromLocalFile(self.logic.modelsPath()))

    def onClearModelsFolder(self):
        if not os.path.exists(self.logic.modelsPath()):
            slicer.util.messageBox("There are no downloaded models.")
            return
        if not slicer.util.confirmOkCancelDisplay("All downloaded model files will be deleted. The files will be automatically downloaded again as needed."):
            return
        self.logic.deleteAllModels()
        slicer.util.messageBox("Downloaded models are deleted.")

#
# MONAIAuto3DSegLogic
#

class MONAIAuto3DSegLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    EXIT_CODE_USER_CANCELLED = 1001
    EXIT_CODE_DID_NOT_RUN = 1002

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        from collections import OrderedDict

        ScriptedLoadableModuleLogic.__init__(self)

        import pathlib
        self.fileCachePath = pathlib.Path.home().joinpath(".MONAIAuto3DSeg")

        self.dependenciesInstalled = False  # we don't know yet if dependencies have been installed

        self.moduleDir = os.path.dirname(slicer.util.getModule('MONAIAuto3DSeg').path)

        self.logCallback = None
        self.processingCompletedCallback = None
        self.startResultImportCallback = None
        self.endResultImportCallback = None
        self.useStandardSegmentNames = True

        # List of property type codes that are specified by in the MONAIAuto3DSeg terminology.
        #
        # # Codes are stored as a list of strings containing coding scheme designator and code value of the property type,
        # separated by "^" character. For example "SCT^123456".
        #
        # If property the code is found in this list then the MONAIAuto3DSeg terminology will be used,
        # otherwise the DICOM terminology will be used. This is necessary because the DICOM terminology
        # does not contain all the necessary items and some items are incomplete (e.g., don't have color or 3D Slicer label).
        #
        self.MONAIAuto3DSegTerminologyPropertyTypes = self._MONAIAuto3DSegTerminologyPropertyTypes()

        # Segmentation models specified by MONAIAuto3DSeg
        # Ideally, this information should be provided by MONAIAuto3DSeg itself.
        self.models = OrderedDict()

        # Main
        self.models = self.loadModelsDescription()
        self.defaultModel = self.models[0]["id"]

        # Timer for checking the output of the segmentation process that is running in the background
        self.processOutputCheckTimerIntervalMsec = 1000

        # Disabling this flag preserves input and output data after execution is completed,
        # which can be useful for troubleshooting.
        self.clearOutputFolder = True

        # For testing the logic without actually running inference, set self.debugSkipInferenceTempDir to the location
        # where inference result is stored and set self.debugSkipInference to True.
        self.debugSkipInference = False
        self.debugSkipInferenceTempDir = r"c:\Users\andra\AppData\Local\Temp\Slicer\__SlicerTemp__2024-01-16_15+26+25.624"


    def model(self, modelId):
        for model in self.models:
            if model["id"] == modelId:
                return model
        raise RuntimeError(f"Model {modelId} not found")

    def loadModelsDescription(self):
        modelsJsonFilePath = os.path.join(self.moduleDir, "Resources", "Models.json")
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
                        models.append({
                            "id": filename,
                            "title": f"{model['title']} (v{version})",
                            "description":
                                f"{model['description']}\n"
                                f"Subject: {model['subject']}\n"
                                f"Imaging modality: {model['imagingModality']}",
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

    def modelsPath(self):
        import pathlib
        return self.fileCachePath.joinpath("models")

    def createModelsDir(self):
        modelsDir = self.modelsPath()
        if not os.path.exists(modelsDir):
            os.makedirs(modelsDir)

    def modelPath(self, modelName):
        import pathlib
        modelRoot = self.modelsPath().joinpath(modelName)
        # find labels.csv file within the modelRoot folder and subfolders
        for path in pathlib.Path(modelRoot).rglob("labels.csv"):
            return path.parent
        raise RuntimeError(f"Model {modelName} path not found")

    def deleteAllModels(self):
        if self.modelsPath().exists():
            import shutil
            shutil.rmtree(self.modelsPath())

    def downloadModel(self, modelName):

        url = self.model(modelName)["url"]

        import zipfile
        import requests
        import pathlib

        tempDir = pathlib.Path(slicer.util.tempDirectory())
        modelDir = self.modelsPath().joinpath(modelName)
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)

        modelZipFile = tempDir.joinpath("autoseg3d_model.zip")
        self.log(f"Downloading model '{modelName}' from {url}...")
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
                            self.log(f"Downloading model: {downloaded_size/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({downloaded_percent:.1f}%)")
                            last_reported_download_percent = downloaded_percent

            self.log(f"Download finished. Extracting to {modelDir}...")
            with zipfile.ZipFile(modelZipFile, 'r') as zip_f:
                zip_f.extractall(modelDir)
        except Exception as e:
            raise e
        finally:
            if self.clearOutputFolder:
                self.log("Cleaning up temporary model download folder...")
                if os.path.isdir(tempDir):
                    import shutil
                    shutil.rmtree(tempDir)
            else:
                self.log(f"Not cleaning up temporary model download folder: {tempDir}")


    def _MONAIAuto3DSegTerminologyPropertyTypes(self):
        """Get label terminology property types defined in from MONAI Auto3DSeg terminology.
        Terminology entries are either in DICOM or MONAI Auto3DSeg "Segmentation category and type".
        """

        terminologiesLogic = slicer.util.getModuleLogic("Terminologies")
        MONAIAuto3DSegTerminologyName = "Segmentation category and type - MONAI Auto3DSeg"

        # Get anatomicalStructureCategory from the MONAI Auto3DSeg terminology
        anatomicalStructureCategory = slicer.vtkSlicerTerminologyCategory()
        numberOfCategories = terminologiesLogic.GetNumberOfCategoriesInTerminology(MONAIAuto3DSegTerminologyName)
        for i in range(numberOfCategories):
            terminologiesLogic.GetNthCategoryInTerminology(MONAIAuto3DSegTerminologyName, i, anatomicalStructureCategory)
            if anatomicalStructureCategory.GetCodingSchemeDesignator() == "SCT" and anatomicalStructureCategory.GetCodeValue() == "123037004":
                # Found the (123037004, SCT, "Anatomical Structure") category within DICOM master list
                break

        # Retrieve all anatomicalStructureCategory property type codes
        terminologyPropertyTypes = []
        terminologyType = slicer.vtkSlicerTerminologyType()
        numberOfTypes = terminologiesLogic.GetNumberOfTypesInTerminologyCategory(MONAIAuto3DSegTerminologyName, anatomicalStructureCategory)
        for i in range(numberOfTypes):
            if terminologiesLogic.GetNthTypeInTerminologyCategory(MONAIAuto3DSegTerminologyName, anatomicalStructureCategory, i, terminologyType):
                terminologyPropertyTypes.append(terminologyType.GetCodingSchemeDesignator() + "^" + terminologyType.GetCodeValue())

        return terminologyPropertyTypes

    def labelDescriptions(self, modelName):
        """Return mapping from label value to label description.
        Label description is a dict containing "name" and "terminology".
        Terminology string uses Slicer terminology entry format - see specification at
        https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmentations.html#terminologyentry-tag
        """

        # Helper function to get code string from CSV file row
        def getCodeString(field, columnNames, row):
            columnValues = []
            for fieldName in ["CodingSchemeDesignator", "CodeValue", "CodeMeaning"]:
                columnIndex = columnNames.index(f"{field}.{fieldName}")
                try:
                    columnValue = row[columnIndex]
                except IndexError:
                    # Probably the line in the CSV file was not terminated by multiple commas (,)
                    columnValue = ""
                columnValues.append(columnValue)
            return columnValues

        labelDescriptions = {}
        labelsFilePath = self.modelPath(modelName).joinpath("labels.csv")
        import csv
        with open(labelsFilePath, "r") as f:
            reader = csv.reader(f)
            columnNames = next(reader)
            data = {}
            # Loop through the rows of the csv file
            for row in reader:

                # Determine segmentation category (DICOM or MONAIAuto3DSeg)
                terminologyEntryStrWithoutCategoryName = (
                    "~"
                    # Property category: "SCT^123037004^Anatomical Structure" or "SCT^49755003^Morphologically Altered Structure"
                    + "^".join(getCodeString("SegmentedPropertyCategoryCodeSequence", columnNames, row))
                    + "~"
                    # Property type: "SCT^23451007^Adrenal gland", "SCT^367643001^Cyst", ...
                    + "^".join(getCodeString("SegmentedPropertyTypeCodeSequence", columnNames, row))
                    + "~"
                    # Property type modifier: "SCT^7771000^Left", ...
                    + "^".join(getCodeString("SegmentedPropertyTypeModifierCodeSequence", columnNames, row))
                    + "~Anatomic codes - DICOM master list"
                    + "~"
                    # Anatomic region (set if category is not anatomical structure): "SCT^64033007^Kidney", ...
                    + "^".join(getCodeString("AnatomicRegionSequence", columnNames, row))
                    + "~"
                    # Anatomic region modifier: "SCT^7771000^Left", ...
                    + "^".join(getCodeString("AnatomicRegionModifierSequence", columnNames, row))
                    + "|")
                terminologyEntry = slicer.vtkSlicerTerminologyEntry()
                terminologyPropertyTypeStr = (  # Example: SCT^23451007
                    row[columnNames.index("SegmentedPropertyTypeCodeSequence.CodingSchemeDesignator")]
                    + "^" + row[columnNames.index("SegmentedPropertyTypeCodeSequence.CodeValue")])
                if terminologyPropertyTypeStr in self.MONAIAuto3DSegTerminologyPropertyTypes:
                    terminologyEntryStr = "Segmentation category and type - MONAI Auto3DSeg" + terminologyEntryStrWithoutCategoryName
                else:
                    terminologyEntryStr = "Segmentation category and type - DICOM master list" + terminologyEntryStrWithoutCategoryName

                # Store the terminology string for this structure
                labelValue = int(row[columnNames.index("LabelValue")])
                name = row[columnNames.index("Name")]
                labelDescriptions[labelValue] = { "name": name, "terminology": terminologyEntryStr }

        return labelDescriptions


    def getSegmentLabelColor(self, terminologyEntryStr):
        """Get segment label and color from terminology"""

        def labelColorFromTypeObject(typeObject):
            """typeObject is a terminology type or type modifier"""
            label = typeObject.GetSlicerLabel() if typeObject.GetSlicerLabel() else typeObject.GetCodeMeaning()
            rgb = typeObject.GetRecommendedDisplayRGBValue()
            return label, (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)

        tlogic = slicer.modules.terminologies.logic()

        terminologyEntry = slicer.vtkSlicerTerminologyEntry()
        if not tlogic.DeserializeTerminologyEntry(terminologyEntryStr, terminologyEntry):
            raise RuntimeError(f"Failed to deserialize terminology string: {terminologyEntryStr}")

        numberOfTypes = tlogic.GetNumberOfTypesInTerminologyCategory(terminologyEntry.GetTerminologyContextName(), terminologyEntry.GetCategoryObject())
        foundTerminologyEntry = slicer.vtkSlicerTerminologyEntry()
        for typeIndex in range(numberOfTypes):
            tlogic.GetNthTypeInTerminologyCategory(terminologyEntry.GetTerminologyContextName(), terminologyEntry.GetCategoryObject(), typeIndex, foundTerminologyEntry.GetTypeObject())
            if terminologyEntry.GetTypeObject().GetCodingSchemeDesignator() != foundTerminologyEntry.GetTypeObject().GetCodingSchemeDesignator():
                continue
            if terminologyEntry.GetTypeObject().GetCodeValue() != foundTerminologyEntry.GetTypeObject().GetCodeValue():
                continue
            if terminologyEntry.GetTypeModifierObject() and terminologyEntry.GetTypeModifierObject().GetCodeValue():
                # Type has a modifier, get the color from there
                numberOfModifiers = tlogic.GetNumberOfTypeModifiersInTerminologyType(terminologyEntry.GetTerminologyContextName(), terminologyEntry.GetCategoryObject(), terminologyEntry.GetTypeObject())
                foundMatchingModifier = False
                for modifierIndex in range(numberOfModifiers):
                    tlogic.GetNthTypeModifierInTerminologyType(terminologyEntry.GetTerminologyContextName(), terminologyEntry.GetCategoryObject(), terminologyEntry.GetTypeObject(),
                        modifierIndex, foundTerminologyEntry.GetTypeModifierObject())
                    if terminologyEntry.GetTypeModifierObject().GetCodingSchemeDesignator() != foundTerminologyEntry.GetTypeModifierObject().GetCodingSchemeDesignator():
                        continue
                    if terminologyEntry.GetTypeModifierObject().GetCodeValue() != foundTerminologyEntry.GetTypeModifierObject().GetCodeValue():
                        continue
                    return labelColorFromTypeObject(foundTerminologyEntry.GetTypeModifierObject())
                continue
            return labelColorFromTypeObject(foundTerminologyEntry.GetTypeObject())

        raise RuntimeError(f"Color was not found for terminology {terminologyEntryStr}")

    def log(self, text):
        logging.info(text)
        if self.logCallback:
            self.logCallback(text)

    def installedMONAIPythonPackageInfo(self):
        import shutil
        import subprocess
        versionInfo = subprocess.check_output([shutil.which("PythonSlicer"), "-m", "pip", "show", "MONAI"]).decode()
        return versionInfo

    def setupPythonRequirements(self, upgrade=False):
        import importlib.metadata
        import importlib.util
        import packaging

        # Install PyTorch
        try:
          import PyTorchUtils
        except ModuleNotFoundError as e:
          raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")

        self.log("Initializing PyTorch...")
        minimumTorchVersion = "1.12"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            self.log("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement = f">={minimumTorchVersion}")
            if torch is None:
                raise ValueError("PyTorch extension needs to be installed to use this module.")
        else:
            # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module."
                                 + f" Minimum required version is {minimumTorchVersion}. You can use 'PyTorch Util' module to install PyTorch"
                                 + f" with version requirement set to: >={minimumTorchVersion}")

        # Install MONAI with required components
        self.log("Initializing MONAI...")
        monaiInstallString = "monai[fire,pyyaml,nibabel,pynrrd,psutil,tensorboard,skimage,itk,tqdm]"
        if upgrade:
            monaiInstallString += " --upgrade"
        slicer.util.pip_install(monaiInstallString)

        self.dependenciesInstalled = True
        self.log("Dependencies are set up successfully.")


    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Model"):
            parameterNode.SetParameter("Model", self.defaultModel)
        if not parameterNode.GetParameter("UseStandardSegmentNames"):
            parameterNode.SetParameter("UseStandardSegmentNames", "true")

    def logProcessOutput(self, proc, returnOutput=False):
        # Wait for the process to end and forward output to the log
        output = ""
        from subprocess import CalledProcessError
        while True:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                if returnOutput:
                    output += line
                self.log(line.rstrip())
            except UnicodeDecodeError as e:
                # Code page conversion happens because `universal_newlines=True` sets process output to text mode,
                # and it fails because probably system locale is not UTF8. We just ignore the error and discard the string,
                # as we only guarantee correct behavior if an UTF8 locale is used.
                pass
        proc.wait()
        retcode = proc.returncode
        if retcode != 0:
            raise CalledProcessError(retcode, proc.args, output=proc.stdout, stderr=proc.stderr)
        return output if returnOutput else None


    @staticmethod
    def executableName(name):
        return name + ".exe" if os.name == "nt" else name


    def process(self, inputVolume, outputSegmentation, model=None, cpu=False, waitForCompletion=True, customProcessData=None):

        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param model: one of self.models
        :param cpu: use CPU instead of GPU
        :param waitForCompletion: if True then the method waits for the processing to finish
        :param customProcessData: any custom data to identify or describe this processing request, it will be returned in the process completed callback when waitForCompletion is False
        """

        if not inputVolume:
            raise ValueError("Input volume is invalid")

        if not outputSegmentation:
            raise ValueError("Output segmentation is invalid")

        if model == None:
            model = self.defaultModel

        try:
            modelPath = self.modelPath(model)
        except:
            self.downloadModel(model)
            modelPath = self.modelPath(model)

        segmentationProcessInfo = {}

        import time
        startTime = time.time()
        self.log("Processing started")

        if self.debugSkipInference:
            # For debugging, use a fixed temporary folder
            tempDir = self.debugSkipInferenceTempDir
        else:
            # Create new empty folder
            tempDir = slicer.util.tempDirectory()

        inputImageFile = tempDir + "/input-volume.nrrd"

        import pathlib
        tempDirPath = pathlib.Path(tempDir)

        # Get Python executable path
        import shutil
        pythonSlicerExecutablePath = shutil.which("PythonSlicer")
        if not pythonSlicerExecutablePath:
            raise RuntimeError("Python was not found")

        modelMainPyFile = modelPath.joinpath("main.py")
        # check if modelMainPyFile exists
        prepareInputNeeded = os.path.isfile(modelMainPyFile)

        if prepareInputNeeded:
            # Legacy
            prepareInputFilesCommand = [ pythonSlicerExecutablePath, modelMainPyFile, inputImageFile ]
            proc = slicer.util.launchConsoleProcess(prepareInputFilesCommand)
            self.logProcessOutput(proc)

        # Write input volume to file
        self.log(f"Writing input file to {inputImageFile}")
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(inputImageFile)
        volumeStorageNode.UseCompressionOff()
        volumeStorageNode.WriteData(inputVolume)
        volumeStorageNode.UnRegister(None)

        if prepareInputNeeded:
            # Legacy
            inputConfigFile = modelPath.joinpath("input.yaml")
            outputSegmentationFile = modelPath.joinpath("ensemble_output/input-volume_ensemble.nrrd")
            workDir = modelPath  # tempDirPath?
            auto3DSegCommand = [ pythonSlicerExecutablePath, "-m", "monai.apps.auto3dseg", "AutoRunner", "run",
                "--input", str(inputConfigFile), "--work_dir", str(workDir),
                "--algos", "segresnet", "--train=False", "--analyze=False", "--ensemble=True" ]
        else:
            outputSegmentationFile = tempDir + "/output-segmentation.nrrd"
            modelPtFile = modelPath.joinpath("model.pt")
            inferenceScriptPyFile = os.path.join(self.moduleDir, "Scripts", "auto3dseg_segresnet_inference.py")
            auto3DSegCommand = [ pythonSlicerExecutablePath, str(inferenceScriptPyFile),
                "--model-file", str(modelPtFile),
                "--image-file", str(inputImageFile),
                "--result-file", str(outputSegmentationFile) ]

        self.log("Creating segmentations with MONAIAuto3DSeg AI...")
        self.log(f"Auto3DSeg command: {auto3DSegCommand}")

        additionalEnvironmentVariables = None
        if cpu:
            additionalEnvironmentVariables = {"CUDA_VISIBLE_DEVICES": "-1"}
            self.log(f"Additional environment variables: {additionalEnvironmentVariables}")

        if self.debugSkipInference:
            proc = None
        else:
            proc = slicer.util.launchConsoleProcess(auto3DSegCommand, updateEnvironment=additionalEnvironmentVariables)

        segmentationProcessInfo["proc"] = proc
        segmentationProcessInfo["procReturnCode"] = MONAIAuto3DSegLogic.EXIT_CODE_DID_NOT_RUN
        segmentationProcessInfo["cancelRequested"] = False
        segmentationProcessInfo["startTime"] = startTime
        segmentationProcessInfo["tempDir"] = tempDir
        segmentationProcessInfo["segmentationProcess"] = proc
        segmentationProcessInfo["inputVolume"] = inputVolume
        segmentationProcessInfo["outputSegmentation"] = outputSegmentation
        segmentationProcessInfo["outputSegmentationFile"] = outputSegmentationFile
        segmentationProcessInfo["model"] = model
        segmentationProcessInfo["customProcessData"] = customProcessData

        if proc:
            if waitForCompletion:
                # Wait for the process to end before returning
                self.logProcessOutput(proc)
                self.onSegmentationProcessCompleted(segmentationProcessInfo)
            else:
                # Run the process in the background
                self.startSegmentationProcessMonitoring(segmentationProcessInfo)
        else:
            # Debugging
            self.onSegmentationProcessCompleted(segmentationProcessInfo)

        return segmentationProcessInfo

    def cancelProcessing(self, segmentationProcessInfo):
        self.log("Cancel is requested.")
        segmentationProcessInfo["cancelRequested"] = True
        proc = segmentationProcessInfo.get("proc")
        if proc:
            # Simple proc.kill() would not work, that would only stop the launcher
            import psutil
            psProcess = psutil.Process(proc.pid)
            for psChildProcess in psProcess.children(recursive=True):
                psChildProcess.kill()
            if psProcess.is_running():
                psProcess.kill()
        else:
            self.onSegmentationProcessCompleted(segmentationProcessInfo)

    @staticmethod
    def _handleProcessOutputThreadProcess(segmentationProcessInfo):
        # Wait for the process to end and forward output to the log
        proc = segmentationProcessInfo["proc"]
        from subprocess import CalledProcessError
        while True:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                segmentationProcessInfo["procOutputQueue"].put(line.rstrip())
            except UnicodeDecodeError as e:
                # Code page conversion happens because `universal_newlines=True` sets process output to text mode,
                # and it fails because probably system locale is not UTF8. We just ignore the error and discard the string,
                # as we only guarantee correct behavior if an UTF8 locale is used.
                pass
        proc.wait()
        retcode = proc.returncode  # non-zero return code means error
        segmentationProcessInfo["procReturnCode"] = retcode


    def startSegmentationProcessMonitoring(self, segmentationProcessInfo):
        import queue
        import sys
        import threading

        segmentationProcessInfo["procOutputQueue"] = queue.Queue()
        segmentationProcessInfo["procThread"] = threading.Thread(target=MONAIAuto3DSegLogic._handleProcessOutputThreadProcess, args=[segmentationProcessInfo])
        segmentationProcessInfo["procThread"].start()

        self.checkSegmentationProcessOutput(segmentationProcessInfo)


    def checkSegmentationProcessOutput(self, segmentationProcessInfo):

        import queue
        outputQueue = segmentationProcessInfo["procOutputQueue"]
        while outputQueue:
            if segmentationProcessInfo.get("procReturnCode") != MONAIAuto3DSegLogic.EXIT_CODE_DID_NOT_RUN:
                self.onSegmentationProcessCompleted(segmentationProcessInfo)
                return
            try:
                line = outputQueue.get_nowait()
                self.log(line)
            except queue.Empty:
                break

        # No more outputs to process now, check again later
        import qt
        qt.QTimer.singleShot(self.processOutputCheckTimerIntervalMsec, lambda segmentationProcessInfo=segmentationProcessInfo: self.checkSegmentationProcessOutput(segmentationProcessInfo))


    def onSegmentationProcessCompleted(self, segmentationProcessInfo):

        startTime = segmentationProcessInfo["startTime"]
        tempDir = segmentationProcessInfo["tempDir"]
        inputVolume = segmentationProcessInfo["inputVolume"]
        outputSegmentation = segmentationProcessInfo["outputSegmentation"]
        outputSegmentationFile = segmentationProcessInfo["outputSegmentationFile"]
        model = segmentationProcessInfo["model"]
        customProcessData = segmentationProcessInfo["customProcessData"]
        procReturnCode = segmentationProcessInfo["procReturnCode"]
        cancelRequested = segmentationProcessInfo["cancelRequested"]

        if cancelRequested:
            procReturnCode = MONAIAuto3DSegLogic.EXIT_CODE_USER_CANCELLED
            self.log(f"Processing was cancelled.")
        else:
            if procReturnCode == 0:

                if self.startResultImportCallback:
                    self.startResultImportCallback(customProcessData)

                try:

                    # Load result
                    self.log("Importing segmentation results...")
                    self.readSegmentation(outputSegmentation, outputSegmentationFile, model)

                    # Set source volume - required for DICOM Segmentation export
                    outputSegmentation.SetNodeReferenceID(outputSegmentation.GetReferenceImageGeometryReferenceRole(), inputVolume.GetID())
                    outputSegmentation.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

                    # Place segmentation node in the same place as the input volume
                    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                    inputVolumeShItem = shNode.GetItemByDataNode(inputVolume)
                    studyShItem = shNode.GetItemParent(inputVolumeShItem)
                    segmentationShItem = shNode.GetItemByDataNode(outputSegmentation)
                    shNode.SetItemParent(segmentationShItem, studyShItem)

                finally:

                    if self.endResultImportCallback:
                        self.endResultImportCallback(customProcessData)

            else:
                self.log(f"Processing failed with return code {procReturnCode}")

        if self.clearOutputFolder:
            self.log("Cleaning up temporary folder.")
            if os.path.isdir(tempDir):
                import shutil
                shutil.rmtree(tempDir)
        else:
            self.log(f"Not cleaning up temporary folder: {tempDir}")

        # Report total elapsed time
        import time
        stopTime = time.time()
        segmentationProcessInfo["stopTime"] = stopTime
        elapsedTime = stopTime - startTime
        if cancelRequested:
            self.log(f"Processing was cancelled after {elapsedTime:.2f} seconds.")
        else:
            if procReturnCode == 0:
                self.log(f"Processing was completed in {elapsedTime:.2f} seconds.")
            else:
                self.log(f"Processing failed after {elapsedTime:.2f} seconds.")

        if self.processingCompletedCallback:
            self.processingCompletedCallback(procReturnCode, customProcessData)


    def readSegmentation(self, outputSegmentation, outputSegmentationFile, model):

        labelValueToDescription = self.labelDescriptions(model)

        # Get label descriptions
        maxLabelValue = max(labelValueToDescription.keys())
        if min(labelValueToDescription.keys()) < 0:
            raise RuntimeError("Label values in class_map must be positive")

        # Get color node with random colors
        randomColorsNode = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeRandom")
        rgba = [0, 0, 0, 0]

        # Create color table for this segmentation model
        colorTableNode = slicer.vtkMRMLColorTableNode()
        colorTableNode.SetTypeToUser()
        colorTableNode.SetNumberOfColors(maxLabelValue+1)
        colorTableNode.SetName(model)
        for labelValue in labelValueToDescription:
            randomColorsNode.GetColor(labelValue,rgba)
            colorTableNode.SetColor(labelValue, rgba[0], rgba[1], rgba[2], rgba[3])
            colorTableNode.SetColorName(labelValue, labelValueToDescription[labelValue]["name"])
        slicer.mrmlScene.AddNode(colorTableNode)

        # Load the segmentation
        outputSegmentation.SetLabelmapConversionColorTableNodeID(colorTableNode.GetID())
        outputSegmentation.AddDefaultStorageNode()
        storageNode = outputSegmentation.GetStorageNode()
        storageNode.SetFileName(outputSegmentationFile)
        storageNode.ReadData(outputSegmentation)

        slicer.mrmlScene.RemoveNode(colorTableNode)

        # Set terminology and color
        for labelValue in labelValueToDescription:
            segmentName = labelValueToDescription[labelValue]["name"]
            terminologyEntryStr = labelValueToDescription[labelValue]["terminology"]
            segmentId = segmentName
            self.setTerminology(outputSegmentation, segmentName, segmentId, terminologyEntryStr)

    def setTerminology(self, segmentation, segmentName, segmentId, terminologyEntryStr):
        segment = segmentation.GetSegmentation().GetSegment(segmentId)
        if not segment:
            # Segment is not present in this segmentation
            return
        if terminologyEntryStr:
            segment.SetTag(segment.GetTerminologyEntryTagName(), terminologyEntryStr)
            try:
                label, color = self.getSegmentLabelColor(terminologyEntryStr)
                if self.useStandardSegmentNames:
                    segment.SetName(label)
                segment.SetColor(color)
            except RuntimeError as e:
                self.log(str(e))

#
# MONAIAuto3DSegTest
#

class MONAIAuto3DSegTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_MONAIAuto3DSeg1()

    def test_MONAIAuto3DSeg1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        inputVolume = SampleData.downloadSample("CTACardio")
        self.delayDisplay("Loaded test data set")

        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        # Test the module logic

        # Logic testing is disabled by default to not overload automatic build machines (pytorch is a huge package and computation
        # on CPU takes 5-10 minutes). Set testLogic to True to enable testing.
        testLogic = False

        if testLogic:
            logic = MONAIAuto3DSegLogic()
            logic.logCallback = self._mylog

            self.delayDisplay("Set up required Python packages")
            logic.setupPythonRequirements()

            self.delayDisplay("Compute output")
            logic.process(inputVolume, outputSegmentation)

        else:
            logging.warning("test_MONAIAuto3DSeg1 logic testing was skipped")

        self.delayDisplay("Test passed")

    def _mylog(self,text):
        print(text)
