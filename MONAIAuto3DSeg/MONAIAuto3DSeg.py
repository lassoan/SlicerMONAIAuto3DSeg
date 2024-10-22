import logging
import os
import json
import sys
import time
from urllib.error import HTTPError

import vtk

import qt
import slicer
import requests

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from MONAIAuto3DSegLib.model_database import ModelDatabase
from MONAIAuto3DSegLib.utils import humanReadableTimeFromSec
from MONAIAuto3DSegLib.dependency_handler import SlicerPythonDependencies, RemotePythonDependencies
from MONAIAuto3DSegLib.process import InferenceServer, LocalInference, ExitCode, SegmentationProcessInfo




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
        self.parent.contributors = ["Andras Lasso (PerkLab, Queen's University)", "Andres Diaz-Pinto (NVIDIA & KCL)", "Rudolf Bumm (KSGR Switzerland), Christian Herz (CHOP)"]
        self.parent.helpText = """
3D Slicer extension for segmentation using MONAI Auto3DSeg AI model.
See more information in the <a href="https://github.com/lassoan/SlicerMONAIAuto3DSeg">extension documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Andras Lasso (PerkLab, Queen's University).
The module uses <a href="https://github.com/Project-MONAI/tutorials/blob/main/MONAIAuto3DSeg/README.md">MONAI Auto3DSeg model</a>.
"""

        self.terminologyName = None
        self.anatomicContextName = None

        slicer.app.connect("startupCompleted()", self.configureDefaultTerminology)
        slicer.app.connect("startupCompleted()", self.registerSampleData)

    def configureDefaultTerminology(self):
        moduleDir = os.path.dirname(self.parent.path)
        terminologyFilePath = os.path.join(moduleDir, "Resources", "SegmentationCategoryTypeModifier-MONAIAuto3DSeg.term.json")
        anatomicContextFilePath = os.path.join(moduleDir, "Resources", "AnatomicRegionAndModifier-MONAIAuto3DSeg.term.json")
        tlogic = slicer.modules.terminologies.logic()
        self.terminologyName = tlogic.LoadTerminologyFromFile(terminologyFilePath)
        self.anatomicContextName = tlogic.LoadAnatomicContextFromFile(anatomicContextFilePath)

    def registerSampleData(self):
        """
        Add data sets to Sample Data module.
        """

        # For each sample data set: specify data set name and sha256 file content
        sampleDataSets = [
            [
                "ProstateX-0000", """
                3940564b147b7bcda52b6b305be3d39f7c9fb901b81caa9863c72ef6ff113dfb *ProstateX-0000-t2-tse-tra.nrrd
                774a3891a506f534b5d60ab9831587803e18da0f2daa343997d09221942bf329 *ProstateX-0000-t2-tse-sag.nrrd
                628b928f818350acdb95952daa107d089a6c9101cf74d51fb129ab9b5d490e59 *ProstateX-0000-t2-tse-cor.nrrd
                62a00ffa4fab7a3d53e8cc17409cb0a6c7e6ebdaea694ecab78c7d8c55b6d4d3 *ProstateX-0000-adc.nrrd
                """
            ],
            [
                "msd-prostate-01", """
                3745533b4bddd6f713d651ae54085fc91f00baab0860f279c8de1960b364ab88 *msd-prostate-01-adc.nrrd
                b85b2e145168ea5d4265b3a9f17ec1fbe6de81b23bf833181021fcdbf6816723 *msd-prostate-01-t2.nrrd
                """
            ],
            [
                "ProstateX-0000", """
                3940564b147b7bcda52b6b305be3d39f7c9fb901b81caa9863c72ef6ff113dfb *ProstateX-0000-t2-tse-tra.nrrd
                774a3891a506f534b5d60ab9831587803e18da0f2daa343997d09221942bf329 *ProstateX-0000-t2-tse-sag.nrrd
                628b928f818350acdb95952daa107d089a6c9101cf74d51fb129ab9b5d490e59 *ProstateX-0000-t2-tse-cor.nrrd
                62a00ffa4fab7a3d53e8cc17409cb0a6c7e6ebdaea694ecab78c7d8c55b6d4d3 *ProstateX-0000-adc.nrrd
                """
            ],
            [
                "ICH-ADAPT2", """
                40e9f1cb82dd68e8bd19dccf0ad116c8cb2eb67a8cfadf3ad9155642e4851d89 *ICH-ADAPT2.nii.gz
                """
            ],
            [
                "BraTS-GLI-00001-000", """
                4399faadcc45c8a4541313cdf88aced7d835ed59ac3078d950e0eac293d603f5 *BraTS-GLI-00001-000-t1c.nii.gz
                e860924b936e301ddeba20409fbb59dde322475cb49328f1b46a9235c792e73e *BraTS-GLI-00001-000-t1n.nii.gz
                82aed8546af5e6d8d94fd91c56227abdcf6120130390d1556c4342a208980604 *BraTS-GLI-00001-000-t2f.nii.gz
                4cd389cc57d12134a30a898c66532228126b9a7d0600ee578e82a32144528b51 *BraTS-GLI-00001-000-t2w.nii.gz
                """
            ],
            [
                "BraTS-MEN-00000-000", """
                4cb970e92edcec52c5c4d72568a17c41006a663391ab76cb871a5054f76c4e37 *BraTS-MEN-00000-000-t1c.nii.gz
                794d184402972c78a8a4be2b889f3683c80878f748912704fd23edc3ce6fa9dd *BraTS-MEN-00000-000-t1n.nii.gz
                e82bf0f3e1870e61500de40d834d92da64021c634e146d225597c2ff3a682cbd *BraTS-MEN-00000-000-t2f.nii.gz
                4d6d7aa361be09cb2b8d2e411fa26499301b5edf454acae1cd442f5c53904f75 *BraTS-MEN-00000-000-t2w.nii.gz
                """
            ],
            [
                "BraTS-MET-00002-000", """
                360b294b428d335c90f340ca727d8fbce4a7ae3aaac9fc61929f4cd71a0aa90b *BraTS-MET-00002-000-t1c.nii.gz
                bc8dcc7e2ae37b1ef6a231c1ee9e1fc355a21322f95572a78b59773c18459654 *BraTS-MET-00002-000-t1n.nii.gz
                863838bfc13259ee9de1cf31e9cb6db1a3a6489863bbe7efd1238f630dce08bf *BraTS-MET-00002-000-t2f.nii.gz
                3076031451a334c29242cbab9b65017c6acd89d5dda9a7566bb79d2325c51df9 *BraTS-MET-00002-000-t2w.nii.gz
                """
            ],
            [
                "BraTS-PED-00030-000", """
                9fd304f9a7c691266b22efeb2610b879c454e0e31df758c32ec039ad2c1acde7 *BraTS-PED-00030-000-t1c.nii.gz
                8c801c464b548b9acd49244db3e31895945c7a26abc1c7bbb34aac72706b7b9d *BraTS-PED-00030-000-t1n.nii.gz
                7c69f8a39b722f3bdd8ed943f5c6aa2aa20aec011727bc237d074aef01c774ff *BraTS-PED-00030-000-t2f.nii.gz
                88914c9e4aafacb21c642132a46c7f12096af958d8b915a80e023d5924fff8d8 *BraTS-PED-00030-000-t2w.nii.gz
                """
            ],
            [
                "BraTS-SSA-00002-000", """
                b980aab6d6fb2e95f01e6f6c964d94a89ef32e717448e1b1c101e163219042b1 *BraTS-SSA-00002-000-t1c.nii.gz
                cd78460e4225a1f145756c7fcda12a517ab3d13f62c52d8b287d78310e520cf7 *BraTS-SSA-00002-000-t1n.nii.gz
                b7067abe232daafbf5c46c9707462edc961cdb30cb5eb445e5d4407e0da70c08 *BraTS-SSA-00002-000-t2f.nii.gz
                4d9810cd217a9504f8aa4313a0edec4a091ecb145bad09dab5b7955688639420 *BraTS-SSA-00002-000-t2w.nii.gz
                """
            ]
        ]

        import SampleData
        iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')
        for sampleDataSet in sampleDataSets:
            sampleName = sampleDataSet[0]
            filenamesWithChecksums = sampleDataSet[1].split("\n")
            uris = []
            filenames = []
            nodeNames = []
            checksums = []
            for filenamesWithChecksum in filenamesWithChecksums:
                # filenamesWithChecksum = '                b980aab6d6fb2e95f01e6f6c964d94a89ef32e717448e1b1c101e163219042b1 *BraTS-SSA-00002-000-t1c.nii.gz'
                filenamesWithChecksum = filenamesWithChecksum.strip()
                if not filenamesWithChecksum:
                    continue
                checksum, filename = filenamesWithChecksum.split(" *")
                uris.append(f"https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/download/TestingData/{filename}")
                filenames.append(filename)
                nodeNames.append(filename.split(".")[0])
                checksums.append(f"SHA256:{checksum}")

            SampleData.SampleDataLogic.registerCustomSampleDataSource(
                category="MONAIAuto3DSeg",
                sampleName=sampleName,
                uris=uris,
                fileNames=filenames,
                nodeNames=nodeNames,
                thumbnailFileName=os.path.join(iconsPath, f"{sampleName}.jpg"),
                checksums=checksums
            )


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
    PROCESSING_COMPLETED = 4
    PROCESSING_CANCEL_REQUESTED = 5
    PROCESSING_FAILED = 6

    PROCESSING_STATES = {
        PROCESSING_IDLE: "Idle",
        PROCESSING_STARTING: "Starting...",
        PROCESSING_IN_PROGRESS: "In Progress",
        PROCESSING_IMPORT_RESULTS: "Importing Results",
        PROCESSING_COMPLETED: "Processing Finished",
        PROCESSING_CANCEL_REQUESTED: "Cancelling...",
        PROCESSING_FAILED: "Processing Failed"
    }

    @staticmethod
    def getHumanReadableProcessingState(state):
        try:
            return MONAIAuto3DSegWidget.PROCESSING_STATES[state]
        except KeyError:
            return "Unknown State"

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
        self._webServer = None

    def onReload(self):
        logging.debug(f"Reloading {self.moduleName}")
        if self._webServer:
            self._webServer.killProcess()

        packageName ="MONAIAuto3DSegLib"
        submoduleNames = ['dependency_handler', 'model_database', 'process', 'utils']
        import imp
        f, filename, description = imp.find_module(packageName)
        package = imp.load_module(packageName, f, filename, description)
        for submoduleName in submoduleNames:
            f, filename, description = imp.find_module(submoduleName, package.__path__)
            try:
                imp.load_module(packageName + '.' + submoduleName, f, filename, description)
            finally:
                f.close()
        super().onReload()

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

        self.ui.downloadSampleDataToolButton.setIcon(qt.QIcon(self.resourcePath("Icons/radiology.svg")))

        self.inputNodeSelectors = [self.ui.inputNodeSelector0, self.ui.inputNodeSelector1, self.ui.inputNodeSelector2, self.ui.inputNodeSelector3]
        self.inputNodeLabels = [self.ui.inputNodeLabel0, self.ui.inputNodeLabel1, self.ui.inputNodeLabel2, self.ui.inputNodeLabel3]

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MONAIAuto3DSegLogic()
        self.logic.logCallback = self.addLog
        self.logic.startResultImportCallback = self.onProcessImportStarted
        self.logic.endResultImportCallback = self.onProcessImportEnded
        self.logic.processingCompletedCallback = self.onProcessingCompleted

        self.ui.remoteProcessingCheckBox.checked = qt.QSettings().value(f"{self.moduleName}/remoteProcessing", False)

        self.ui.progressBar.hide()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        for inputNodeSelector in self.inputNodeSelectors:
            inputNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.fullTextSearchCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.cpuCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.showAllModelsCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.useStandardSegmentNamesCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.ui.modelSearchBox.connect("textChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.modelComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.ui.segmentationShow3DButton.setSegmentationNode)

        # Buttons
        self.ui.downloadSampleDataToolButton.connect("clicked(bool)", self.onDownloadSampleData)
        self.ui.packageInfoUpdateButton.connect("clicked(bool)", self.onPackageInfoUpdate)
        self.ui.packageUpgradeButton.connect("clicked(bool)", self.onPackageUpgrade)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.browseToModelsFolderButton.connect("clicked(bool)", self.onBrowseModelsFolder)
        self.ui.deleteAllModelsButton.connect("clicked(bool)", self.onClearModelsFolder)

        self.ui.serverComboBox.lineEdit().setPlaceholderText("Enter server address")
        self.ui.serverComboBox.currentIndexChanged.connect(self.onRemoteServerButtonToggled)
        self.ui.remoteProcessingCheckBox.toggled.connect(self.onRemoteProcessingCheckBoxToggled)
        self.ui.remoteServerButton.toggled.connect(self.onRemoteServerButtonToggled)

        self.ui.serverButton.toggled.connect(self.onServerButtonToggled)
        self.ui.portSpinBox.valueChanged.connect(self.updateParameterNodeFromGUI)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.updateServerUrlGUIFromSettings()

        self.updateGUIFromParameterNode()

        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IDLE)

        # Make the model search box in focus by default so users can just start typing to find the model they need
        qt.QTimer.singleShot(0, self.ui.modelSearchBox.setFocus)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        if self._webServer:
            self._webServer.killProcess()
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
        if not self._parameterNode.GetNodeReference("InputNode0"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputNode0", firstVolumeNode.GetID())

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
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        try:
            self.ui.modelSearchBox.text = self._parameterNode.GetParameter("ModelSearchText")

            searchWords = self._parameterNode.GetParameter("ModelSearchText").lower().split()
            fullTextSearch = self._parameterNode.GetParameter("FullTextSearch") == "true"
            showAllModels = self._parameterNode.GetParameter("ShowAllModels") == "true"
            self.ui.modelComboBox.clear()
            for model in self.logic.models:

                if model.get("deprecated"):
                    if showAllModels:
                        modelTitle = f"{model['title']} (v{model['version']}) -- deprecated"
                    else:
                        # Do not show deprecated models
                        continue
                else:
                    if showAllModels:
                        modelTitle = f"{model['title']} (v{model['version']})"
                    else:
                        modelTitle = model['title']

                if searchWords:
                    textToSearchIn = modelTitle.lower()
                    if fullTextSearch:
                        textToSearchIn += " " + model.get("description").lower() + " " + model.get("imagingModality").lower()
                        segmentNames = model.get("segmentNames")
                        if segmentNames:
                            segmentNames = " ".join(segmentNames)
                            textToSearchIn += " " + segmentNames.lower()
                    if not all(word in textToSearchIn for word in searchWords):
                        continue

                itemIndex = self.ui.modelComboBox.count
                self.ui.modelComboBox.addItem(modelTitle)
                item = self.ui.modelComboBox.item(itemIndex)
                item.setData(qt.Qt.UserRole, model["id"])
                item.setData(qt.Qt.ToolTipRole, "<html>" + model.get("details") + "</html>")

            modelId = self._parameterNode.GetParameter("Model")
            currentModelSelectable = self._setCurrentModelId(modelId)
            if not currentModelSelectable:
                modelId = ""
            sampleDataAvailable = self.logic.model(modelId).get("inputs") if modelId else False
            self.ui.downloadSampleDataToolButton.visible = sampleDataAvailable

            self.ui.fullTextSearchCheckBox.checked = fullTextSearch
            self.ui.cpuCheckBox.checked = self._parameterNode.GetParameter("CPU") == "true"
            self.ui.showAllModelsCheckBox.checked = showAllModels
            self.ui.useStandardSegmentNamesCheckBox.checked = self._parameterNode.GetParameter("UseStandardSegmentNames") == "true"
            self.ui.outputSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSegmentation"))

            state = self._processingState
            if state == MONAIAuto3DSegWidget.PROCESSING_IDLE:
                self.ui.applyButton.text = "Apply"
                inputErrorMessages = []  # it will contain text if the inputs are not valid
                if self.ui.remoteProcessingCheckBox.checked and not self.ui.remoteServerButton.checked:
                    inputErrorMessages.append("Connect to server or disable remote processing.")
                    self.ui.modelComboBox.enabled = False
                else:
                    self.ui.modelComboBox.enabled = True
                if modelId:
                    modelInputs = self.logic.model(modelId)["inputs"]
                else:
                    modelInputs = []
                    inputErrorMessages.append("Select a model.")
                inputNodes = []  # list of output nodes so far, for checking for duplicates
                for inputIndex in range(len(self.inputNodeSelectors)):
                    inputNodeSelector = self.inputNodeSelectors[inputIndex]
                    inputNodeLabel = self.inputNodeLabels[inputIndex]
                    if inputIndex < len(modelInputs):
                        inputNodeLabel.visible = True
                        inputTitle = modelInputs[inputIndex]["title"]
                        inputNodeLabel.text = f"{inputTitle}:"
                        inputNodeSelector.visible = True
                        inputNode = self._parameterNode.GetNodeReference("InputNode" + str(inputIndex))
                        inputNodeSelector.setCurrentNode(inputNode)
                        if inputIndex == 0 and inputNode:
                            self.ui.outputSegmentationSelector.baseName = inputNode.GetName() + " segmentation"
                        if not inputNode:
                            inputErrorMessages.append(f"Select {inputTitle}.")
                        else:
                            if inputNode in inputNodes:
                                inputErrorMessages.append(f"'{inputTitle}' does not have a unique input ('{inputNode.GetName()}' is already used as another input).")
                            inputNodes.append(inputNode)
                    else:
                        inputNodeLabel.visible = False
                        inputNodeSelector.visible = False

                if inputErrorMessages:
                    self.ui.applyButton.toolTip = "\n".join(inputErrorMessages)
                    self.ui.applyButton.enabled = False
                else:
                    self.ui.applyButton.toolTip = "Start segmentation"
                    self.ui.applyButton.enabled = True

            elif state == MONAIAuto3DSegWidget.PROCESSING_STARTING:
                self.ui.applyButton.toolTip = "Please wait while the segmentation is being initialized"
                self.ui.applyButton.enabled = False
            elif state == MONAIAuto3DSegWidget.PROCESSING_IN_PROGRESS:
                self.ui.applyButton.text = "Cancel"
                self.ui.applyButton.toolTip = "Cancel in-progress segmentation"
                self.ui.applyButton.enabled = True
            elif state == MONAIAuto3DSegWidget.PROCESSING_IMPORT_RESULTS:
                self.ui.applyButton.toolTip = "Please wait while the segmentation result is being imported"
                self.ui.applyButton.enabled = False
            elif state == MONAIAuto3DSegWidget.PROCESSING_CANCEL_REQUESTED:
                self.ui.applyButton.toolTip = "Please wait for the segmentation to be cancelled"
                self.ui.applyButton.enabled = False

            remoteConnection = self.ui.remoteServerButton.checked

            # if remoteConnection:
            #     self.ui.serverCollapsibleButton.collapsed = True

            self.ui.portSpinBox.value = int(self._parameterNode.GetParameter("ServerPort"))

            self.ui.serverAddressTitleLabel.visible = self._webServer is not None
            self.ui.serverAddressLabel.visible = self._webServer is not None
            if self._webServer:
                self.ui.serverAddressLabel.text = self._webServer.getAddressUrl()

            self.ui.browseToModelsFolderButton.enabled = not remoteConnection
            self.ui.useStandardSegmentNamesCheckBox.enabled = not remoteConnection
            self.ui.cpuCheckBox.enabled = not remoteConnection
            self.ui.showAllModelsCheckBox.enabled = not remoteConnection
            self.ui.deleteAllModelsButton.enabled = not remoteConnection
            self.ui.packageUpgradeButton.enabled = not remoteConnection

            serverRunning = self._webServer is not None and self._webServer.isRunning()
            self.ui.serverButton.checked = serverRunning
            self.ui.serverButton.text = "Running ..." if serverRunning else "Start server"
        finally:
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

        try:

            for inputIndex in range(len(self.inputNodeSelectors)):
                inputNodeSelector = self.inputNodeSelectors[inputIndex]
                self._parameterNode.SetNodeReferenceID("InputNode" + str(inputIndex), inputNodeSelector.currentNodeID)

            self._parameterNode.SetParameter("ModelSearchText", self.ui.modelSearchBox.text)
            modelId = self._currentModelId()
            if modelId:
                # Only save model ID if valid, otherwise it is temporarily filtered out in the selector
                self._parameterNode.SetParameter("Model", modelId)
            self._parameterNode.SetParameter("FullTextSearch", "true" if self.ui.fullTextSearchCheckBox.checked else "false")
            self._parameterNode.SetParameter("CPU", "true" if self.ui.cpuCheckBox.checked else "false")
            self._parameterNode.SetParameter("ShowAllModels", "true" if self.ui.showAllModelsCheckBox.checked else "false")
            self._parameterNode.SetParameter("UseStandardSegmentNames", "true" if self.ui.useStandardSegmentNamesCheckBox.checked else "false")
            self._parameterNode.SetNodeReferenceID("OutputSegmentation", self.ui.outputSegmentationSelector.currentNodeID)
            self._parameterNode.SetParameter("ServerPort", str(self.ui.portSpinBox.value))

        finally:
            self._parameterNode.EndModify(wasModified)

    def addLog(self, text):
        """Append text to log window
        """
        if len(self.ui.statusLabel.html) > 1024 * 256:
            self.ui.statusLabel.clear()
            self.ui.statusLabel.insertHtml("Log cleared\n")
        self.ui.statusLabel.insertHtml(text)
        self.ui.statusLabel.insertPlainText("\n")
        self.ui.statusLabel.ensureCursorVisible()
        self.ui.statusLabel.repaint()

        # self.ui.statusLabel.appendPlainText(text)
        # slicer.app.processEvents()  # force update

    def updateProgress(self, state):
        if state == self.PROCESSING_IDLE:
            qt.QTimer.singleShot(1000, self.ui.progressBar.hide)
            self.ui.progressBar.setRange(0,0)
        else:
            self.ui.progressBar.setRange(0,4)
            self.ui.progressBar.show()
            self.ui.progressBar.value = state
            self.ui.progressBar.setFormat(text := self.getHumanReadableProcessingState(state))
            self.addLog(text)

    def addServerLog(self, *args):
        for arg in args:
            if self.ui.logConsoleCheckBox.checked:
                print(arg)
            if self.ui.logGuiCheckBox.checked:
                self.addLog(arg)

    def setProcessingState(self, state):
        self._processingState = state
        self.updateGUIFromParameterNode()
        self.updateProgress(state)
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

        with slicer.util.tryWithErrorDisplay("Processing Failed. Check logs for more information.", waitCursor=True):
            try:
                # Create new segmentation node, if not selected yet
                if not self.ui.outputSegmentationSelector.currentNode():
                    self.ui.outputSegmentationSelector.addNode()

                self.logic.useStandardSegmentNames = self.ui.useStandardSegmentNamesCheckBox.checked

                # Compute output
                inputNodes = []
                for inputNodeSelector in self.inputNodeSelectors:
                    if inputNodeSelector.visible:
                        inputNodes.append(inputNodeSelector.currentNode())

                self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IN_PROGRESS)
                self._segmentationProcessInfo = self.logic.process(inputNodes, self.ui.outputSegmentationSelector.currentNode(),
                    self._currentModelId(), self.ui.cpuCheckBox.checked, waitForCompletion=False)

            except Exception as e:
                self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_FAILED)
                self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IDLE)
                raise

    def onCancel(self):
        with slicer.util.tryWithErrorDisplay("Failed to cancel processing.", waitCursor=True):
            self.logic.cancelProcessing()
            self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_CANCEL_REQUESTED)

    def onProcessImportStarted(self, customData):
        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IMPORT_RESULTS)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        slicer.app.processEvents()

    def onProcessImportEnded(self, customData):
        qt.QApplication.restoreOverrideCursor()
        slicer.app.processEvents()

    def onProcessingCompleted(self, returnCode, customData):
        if returnCode == 0:
            m = "\nProcessing finished."
        elif returnCode == ExitCode.USER_CANCELLED:
            m = "\nProcessing was cancelled."
        else:
            m = f"\nProcessing failed with error code {returnCode}. Please check logs for further information."
        self.addLog(m)
        self.setProcessingState(MONAIAuto3DSegWidget.PROCESSING_IDLE)
        self._segmentationProcessInfo = None

    def _currentModelId(self):
        itemIndex = self.ui.modelComboBox.currentRow
        item = self.ui.modelComboBox.item(itemIndex)
        if not item:
            return ""
        return item.data(qt.Qt.UserRole)

    def _setCurrentModelId(self, modelId):
        for itemIndex in range(self.ui.modelComboBox.count):
            item = self.ui.modelComboBox.item(itemIndex)
            if item.data(qt.Qt.UserRole) == modelId:
                self.ui.modelComboBox.setCurrentRow(itemIndex)
                return True
        return False

    def onDownloadSampleData(self):
        with slicer.util.tryWithErrorDisplay("Failed to retrieve model information", waitCursor=True):
            model = self.logic.model(self._currentModelId())

        sampleDataName = model.get("sampleData")
        if not sampleDataName:
            slicer.util.messageBox("No sample data is available for this model.")
            return

        if type(sampleDataName) == list:
            # For now, always just use the first data set if multiple data sets are provided
            sampleDataName = sampleDataName[0]

        with slicer.util.tryWithErrorDisplay("Failed to download sample data", waitCursor=True):
            import SampleData
            loadedSampleNodes = SampleData.SampleDataLogic().downloadSamples(sampleDataName)
            inputs = model.get("inputs")

        if not loadedSampleNodes:
            slicer.util.messageBox(f"Failed to load sample data set '{sampleDataName}'.")
            return

        inputNodes = self.logic.assignInputNodesByName(inputs, loadedSampleNodes)
        for inputIndex, inputNode in enumerate(inputNodes):
            if inputNode:
                self.inputNodeSelectors[inputIndex].setCurrentNode(inputNode)

    def onPackageInfoUpdate(self):
        self.ui.packageInfoTextBrowser.plainText = ""
        with slicer.util.tryWithErrorDisplay("Failed to get MONAI package version information", waitCursor=True):
            self.ui.packageInfoTextBrowser.plainText = self.logic.getMONAIPythonPackageInfo().rstrip()

    def onPackageUpgrade(self):
        restartRequired = True
        with slicer.util.tryWithErrorDisplay("Failed to upgrade MONAI", waitCursor=True):
            restartRequired = self.logic.setupPythonRequirements(upgrade=True)
        self.onPackageInfoUpdate()
        if restartRequired:
            if not slicer.util.confirmOkCancelDisplay(f"This MONAI update requires a 3D Slicer restart.","Press OK to restart."):
                raise ValueError("Restart was cancelled.")
            else:
                slicer.util.restart()

    def onBrowseModelsFolder(self):
        self.logic.createModelsDir()
        qt.QDesktopServices().openUrl(qt.QUrl.fromLocalFile(self.logic.modelsPath))

    def onClearModelsFolder(self):
        if not os.path.exists(self.logic.modelsPath):
            slicer.util.messageBox("There are no downloaded models.")
            return
        if not slicer.util.confirmOkCancelDisplay("All downloaded model files will be deleted. The files will be automatically downloaded again as needed."):
            return
        self.logic.deleteAllModels()
        slicer.util.messageBox("Downloaded models are deleted.")

    def onRemoteServerButtonToggled(self):
        if self.ui.remoteServerButton.checked and self.ui.serverComboBox.currentText != '':
            self.ui.remoteServerButton.text = "Connected"
            self.logic = RemoteMONAIAuto3DSegLogic()
            self.logic.server_address = self.ui.serverComboBox.currentText
            try:
                models = self.logic.models
                self.addLog(f"Remote Server Connected {self.logic.server_address}. {len(models)} models are available.")
            except:
                slicer.util.warningDisplay(
                    f"Connection to remote server '{self.logic.server_address}' failed. "
                    f"Please check address, port, and connection."
                )
                self.ui.remoteServerButton.checked = False
                return
            self.saveServerUrl()
        else:
            self.ui.remoteServerButton.checked = False
            self.ui.remoteServerButton.text = "Connect"
            self.logic = MONAIAuto3DSegLogic()

        self.logic.startResultImportCallback = self.onProcessImportStarted
        self.logic.endResultImportCallback = self.onProcessImportEnded
        self.logic.processingCompletedCallback = self.onProcessingCompleted
        self.updateGUIFromParameterNode()

    def onServerButtonToggled(self, toggled):
        with slicer.util.tryWithErrorDisplay("Failed to start server.", waitCursor=True):
            if toggled:
                # TODO: improve error reporting if installation of requirements fails

                self.ui.statusLabel.plainText = ""
                self.logic.setupPythonRequirements()

                if not self._webServer or not self._webServer.isRunning() :
                    import platform
                    from pathlib import Path
                    slicer.util.pip_install("psutil python-multipart fastapi slowapi uvicorn[standard]")

                    hostName = platform.node()
                    port = str(self.ui.portSpinBox.value)

                    self._webServer = InferenceServer(
                        logCallback=self.addServerLog,
                        completedCallback=self.onServerCompleted
                    )
                    self._webServer.hostName = hostName
                    self._webServer.port = port
                    self._webServer.start()
                    if self._webServer.isRunning():
                        self.addLog("Server started")
            else:
                if self._webServer is not None and self._webServer.isRunning():
                    self._webServer.stop()
                    self._webServer = None
        self.updateGUIFromParameterNode()

    def onServerCompleted(self, processInfo=None):
        returnCode = processInfo.procReturnCode
        if returnCode == ExitCode.USER_CANCELLED:
            m = "\nServer was stopped."
        else:
            m = f"\nProcessing failed with error code {returnCode}. Try again with `Log to GUI` for more details."
        self.addLog(m)
        self.ui.serverButton.setChecked(False)

    def serverUrl(self):
        serverUrl = self.ui.serverComboBox.currentText.strip()
        if not serverUrl:
            serverUrl = "http://127.0.0.1:8000"
        return serverUrl.rstrip("/")

    def saveServerUrl(self):
        settings = qt.QSettings()
        serverUrl = self.ui.serverComboBox.currentText
        settings.setValue(f"{self.moduleName}/serverUrl", serverUrl)
        serverUrlHistory = self._getServerUrlHistory(serverUrl, settings)
        serverUrlHistory.insert(0, serverUrl) # Save current server URL to the top of history
        serverUrlHistory = serverUrlHistory[:10]  # keep up to first 10 elements
        settings.setValue(f"{self.moduleName}/serverUrlHistory", ";".join(serverUrlHistory))
        self.updateServerUrlGUIFromSettings()

    def _getServerUrlHistory(self, serverUrl, settings):
        serverUrlHistory = settings.value(f"{self.moduleName}/serverUrlHistory")
        if serverUrlHistory:
            serverUrlHistory = serverUrlHistory.split(";")
        else:
            serverUrlHistory = []
        try:
            serverUrlHistory.remove(serverUrl)
        except ValueError:
            pass
        return serverUrlHistory

    def updateServerUrlGUIFromSettings(self):
        settings = qt.QSettings()
        serverUrlHistory = settings.value(f"{self.moduleName}/serverUrlHistory")

        wasBlocked = self.ui.serverComboBox.blockSignals(True)
        self.ui.serverComboBox.clear()
        if serverUrlHistory:
            self.ui.serverComboBox.addItems(serverUrlHistory.split(";"))
        self.ui.serverComboBox.setCurrentText(settings.value(f"{self.moduleName}/serverUrl"))
        self.ui.serverComboBox.blockSignals(wasBlocked)

    def onRemoteProcessingCheckBoxToggled(self, checked):
        # Disconnect remote server button if remote processing state is changed
        self.ui.remoteServerButton.setChecked(False)
        settings = qt.QSettings()
        settings.setValue(f"{self.moduleName}/remoteProcessing", checked)
        self.updateGUIFromParameterNode()

#
# MONAIAuto3DSegLogic
#


class MONAIAuto3DSegLogic(ScriptedLoadableModuleLogic, ModelDatabase):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    DEPENDENCY_HANDLER = SlicerPythonDependencies()

    @staticmethod
    def assignInputNodesByName(inputs, loadedSampleNodes):

        def findFirstNodeByNamePattern(namePattern, nodes):
            import fnmatch
            for node in nodes:
                if fnmatch.fnmatchcase(node.GetName(), namePattern):
                    return node
            return None

        inputNodes = []
        for inputIndex, input in enumerate(inputs):
            namePattern = input.get("namePattern")
            if namePattern:
                matchingNode = findFirstNodeByNamePattern(namePattern, loadedSampleNodes)
            else:
                matchingNode = loadedSampleNodes[inputIndex] if inputIndex < len(loadedSampleNodes) else \
                    loadedSampleNodes[0]
            inputNodes.append(matchingNode)
        return inputNodes

    @staticmethod
    def getLoadedTerminologyNames():
        import vtk
        terminologyNames = vtk.vtkStringArray()
        terminologiesLogic = slicer.util.getModuleLogic("Terminologies")
        terminologiesLogic.GetLoadedTerminologyNames(terminologyNames)

        return [terminologyNames.GetValue(idx) for idx in range(terminologyNames.GetNumberOfValues())]

    @staticmethod
    def getLoadedAnatomicContextNames():
        import vtk
        anatomicContextNames = vtk.vtkStringArray()
        terminologiesLogic = slicer.util.getModuleLogic("Terminologies")
        terminologiesLogic.GetLoadedAnatomicContextNames(anatomicContextNames)

        return [anatomicContextNames.GetValue(idx) for idx in range(anatomicContextNames.GetNumberOfValues())]

    @staticmethod
    def _terminologyPropertyTypes(terminologyName):
        """Get label terminology property types defined in from MONAI Auto3DSeg terminology.
        Terminology entries are either in DICOM or MONAI Auto3DSeg "Segmentation category and type".

        # List of property type codes that are specified by in the terminology.
        #
        # Codes are stored as a list of strings containing coding scheme designator and code value of the property type,
        # separated by "^" character. For example "SCT^123456".

        """
        terminologiesLogic = slicer.util.getModuleLogic("Terminologies")
        terminologyPropertyTypes = []

        # Get anatomicalStructureCategory from the MONAI Auto3DSeg terminology
        anatomicalStructureCategory = slicer.vtkSlicerTerminologyCategory()
        numberOfCategories = terminologiesLogic.GetNumberOfCategoriesInTerminology(terminologyName)
        for cIdx in range(numberOfCategories):
            terminologiesLogic.GetNthCategoryInTerminology(terminologyName, cIdx, anatomicalStructureCategory)

            # Retrieve all anatomicalStructureCategory property type codes
            terminologyType = slicer.vtkSlicerTerminologyType()
            numberOfTypes = terminologiesLogic.GetNumberOfTypesInTerminologyCategory(terminologyName,
                                                                                     anatomicalStructureCategory)
            for tIdx in range(numberOfTypes):
                if terminologiesLogic.GetNthTypeInTerminologyCategory(terminologyName, anatomicalStructureCategory, tIdx,
                                                                      terminologyType):
                    terminologyPropertyTypes.append(
                        terminologyType.GetCodingSchemeDesignator() + "^" + terminologyType.GetCodeValue())

        return terminologyPropertyTypes

    @staticmethod
    def _anatomicRegions(anatomicContextName):
        """Get anatomic regions defined in from MONAI Auto3DSeg terminology.
        Terminology entries are either in DICOM or MONAI Auto3DSeg "Anatomic codes".
        """
        anatomicRegions = []

        terminologiesLogic = slicer.util.getModuleLogic("Terminologies")
        if not hasattr(terminologiesLogic, "GetNumberOfRegionsInAnatomicContext"):
            # This Slicer version does not have GetNumberOfRegionsInAnatomicContext method,
            # do not add the region modifier (the only impact is that the modifier will not be selectable
            # when editing the terminology on the GUI)
            return anatomicRegions

        # Retrieve all anatomical region codes
        regionObject = slicer.vtkSlicerTerminologyType()
        numberOfRegions = terminologiesLogic.GetNumberOfRegionsInAnatomicContext(anatomicContextName)
        for i in range(numberOfRegions):
            if terminologiesLogic.GetNthRegionInAnatomicContext(anatomicContextName, i, regionObject):
                anatomicRegions.append(regionObject.GetCodingSchemeDesignator() + "^" + regionObject.GetCodeValue())

        return anatomicRegions

    @staticmethod
    def getSegmentLabelColor(terminologyEntryStr):
        """Get segment label and color from terminology"""

        def labelColorFromTypeObject(typeObject):
            """typeObject is a terminology type or type modifier"""
            label = typeObject.GetSlicerLabel() if typeObject.GetSlicerLabel() else typeObject.GetCodeMeaning()
            rgb = typeObject.GetRecommendedDisplayRGBValue()
            return label, (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        tlogic = slicer.modules.terminologies.logic()

        terminologyEntry = slicer.vtkSlicerTerminologyEntry()
        if not tlogic.DeserializeTerminologyEntry(terminologyEntryStr, terminologyEntry):
            raise RuntimeError(f"Failed to deserialize terminology string: {terminologyEntryStr}")

        numberOfTypes = tlogic.GetNumberOfTypesInTerminologyCategory(terminologyEntry.GetTerminologyContextName(),
                                                                     terminologyEntry.GetCategoryObject())
        foundTerminologyEntry = slicer.vtkSlicerTerminologyEntry()
        for typeIndex in range(numberOfTypes):
            tlogic.GetNthTypeInTerminologyCategory(terminologyEntry.GetTerminologyContextName(),
                                                   terminologyEntry.GetCategoryObject(), typeIndex,
                                                   foundTerminologyEntry.GetTypeObject())
            if terminologyEntry.GetTypeObject().GetCodingSchemeDesignator() != foundTerminologyEntry.GetTypeObject().GetCodingSchemeDesignator():
                continue
            if terminologyEntry.GetTypeObject().GetCodeValue() != foundTerminologyEntry.GetTypeObject().GetCodeValue():
                continue
            if terminologyEntry.GetTypeModifierObject() and terminologyEntry.GetTypeModifierObject().GetCodeValue():
                # Type has a modifier, get the color from there
                numberOfModifiers = tlogic.GetNumberOfTypeModifiersInTerminologyType(
                    terminologyEntry.GetTerminologyContextName(), terminologyEntry.GetCategoryObject(),
                    terminologyEntry.GetTypeObject())
                foundMatchingModifier = False
                for modifierIndex in range(numberOfModifiers):
                    tlogic.GetNthTypeModifierInTerminologyType(terminologyEntry.GetTerminologyContextName(),
                                                               terminologyEntry.GetCategoryObject(),
                                                               terminologyEntry.GetTypeObject(),
                                                               modifierIndex,
                                                               foundTerminologyEntry.GetTypeModifierObject())
                    if terminologyEntry.GetTypeModifierObject().GetCodingSchemeDesignator() != foundTerminologyEntry.GetTypeModifierObject().GetCodingSchemeDesignator():
                        continue
                    if terminologyEntry.GetTypeModifierObject().GetCodeValue() != foundTerminologyEntry.GetTypeModifierObject().GetCodeValue():
                        continue
                    return labelColorFromTypeObject(foundTerminologyEntry.GetTypeModifierObject())
                continue
            return labelColorFromTypeObject(foundTerminologyEntry.GetTypeObject())

        raise RuntimeError(f"Color was not found for terminology {terminologyEntryStr}")

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        ModelDatabase.__init__(self)

        self.logCallback = None
        self.startResultImportCallback = None
        self.endResultImportCallback = None
        self.processingCompletedCallback = None
        self.useStandardSegmentNames = True

        # process that will used to run inference either remotely or locally
        self._bgProcess = None

        # For testing the logic without actually running inference, set self.debugSkipInferenceTempDir to the location
        # where inference result is stored and set self.debugSkipInference to True.
        # Disabling this flag preserves input and output data after execution is completed,
        # which can be useful for troubleshooting.
        self.clearOutputFolder = True
        self.debugSkipInference = False
        self.debugSkipInferenceTempDir = r"c:\Users\andra\AppData\Local\Temp\Slicer\__SlicerTemp__2024-01-16_15+26+25.624"

    def log(self, text):
        logging.info(text)
        if self.logCallback:
            self.logCallback(text)

    def getMONAIPythonPackageInfo(self):
        return self.DEPENDENCY_HANDLER.installedMONAIPythonPackageInfo()

    def setupPythonRequirements(self, upgrade=False):
        self.DEPENDENCY_HANDLER.setupPythonRequirements(upgrade)
        return True

    def labelDescriptions(self, modelName):
        """Return mapping from label value to label description.
        Label description is a dict containing "name" and "terminology".
        Terminology string uses Slicer terminology entry format - see specification at
        https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmentations.html#terminologyentry-tag
        """
        labelsFilePath = self.modelPath(modelName).joinpath("labels.csv")
        return self._labelDescriptions(labelsFilePath)

    def _labelDescriptions(self, labelsFilePath):
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
        import csv
        with open(labelsFilePath, "r") as f:
            reader = csv.reader(f)
            columnNames = next(reader)
            # Loop through the rows of the csv file
            for row in reader:
                # Determine segmentation category (DICOM or MONAIAuto3DSeg)
                terminologyPropertyTypeStr = (  # Example: SCT^23451007
                  row[columnNames.index("SegmentedPropertyTypeCodeSequence.CodingSchemeDesignator")]
                  + "^" + row[columnNames.index("SegmentedPropertyTypeCodeSequence.CodeValue")])
                terminologyName = None

                # If property the code is found in this list then the terminology will be used,
                for tName in self.getLoadedTerminologyNames():
                    propertyTypes = self._terminologyPropertyTypes(tName)
                    if terminologyPropertyTypeStr in propertyTypes:
                        terminologyName = tName
                        break

                # NB: DICOM terminology will be used otherwise. Note: the DICOM terminology does not contain all the
                # necessary items and some items are incomplete (e.g., don't have color or 3D Slicer label).
                if not terminologyName:
                    terminologyName = "Segmentation category and type - DICOM master list"

                # Determine the anatomic context name (DICOM or MONAIAuto3DSeg)
                anatomicRegionStr = (  # Example: SCT^279245009
                  row[columnNames.index("AnatomicRegionSequence.CodingSchemeDesignator")]
                  + "^" + row[columnNames.index("AnatomicRegionSequence.CodeValue")])
                anatomicContextName = None
                for aName in self.getLoadedAnatomicContextNames():
                    if anatomicRegionStr in self._anatomicRegions(aName):
                        anatomicContextName = aName
                if not anatomicContextName:
                    anatomicContextName = "Anatomic codes - DICOM master list"

                terminologyEntryStr = (
                  terminologyName
                  + "~"
                  # Property category: "SCT^123037004^Anatomical Structure" or "SCT^49755003^Morphologically Altered Structure"
                  + "^".join(getCodeString("SegmentedPropertyCategoryCodeSequence", columnNames, row))
                  + "~"
                  # Property type: "SCT^23451007^Adrenal gland", "SCT^367643001^Cyst", ...
                  + "^".join(getCodeString("SegmentedPropertyTypeCodeSequence", columnNames, row))
                  + "~"
                  # Property type modifier: "SCT^7771000^Left", ...
                  + "^".join(getCodeString("SegmentedPropertyTypeModifierCodeSequence", columnNames, row))
                  + "~"
                  + anatomicContextName
                  + "~"
                  # Anatomic region (set if category is not anatomical structure): "SCT^64033007^Kidney", ...
                  + "^".join(getCodeString("AnatomicRegionSequence", columnNames, row))
                  + "~"
                  # Anatomic region modifier: "SCT^7771000^Left", ...
                  + "^".join(getCodeString("AnatomicRegionModifierSequence", columnNames, row))
                )

                # Store the terminology string for this structure
                labelValue = int(row[columnNames.index("LabelValue")])
                name = row[columnNames.index("Name")]
                labelDescriptions[labelValue] = {"name": name, "terminology": terminologyEntryStr}
        return labelDescriptions

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Model"):
            parameterNode.SetParameter("Model", self.defaultModel)
        if not parameterNode.GetParameter("UseStandardSegmentNames"):
            parameterNode.SetParameter("UseStandardSegmentNames", "true")
        if not parameterNode.GetParameter("ServerPort"):
            parameterNode.SetParameter("ServerPort", str(8891))

    def process(self, inputNodes, outputSegmentation, model=None, cpu=False, waitForCompletion=True, customData=None):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputNodes: input nodes in a list
        :param outputSegmentation: output segmentation to write to
        :param model: one of self.models
        :param cpu: use CPU instead of GPU
        :param waitForCompletion: if True then the method waits for the processing to finish
        :param customData: any custom data to identify or describe this processing request, it will be returned in the process completed callback when waitForCompletion is False
        """

        if not self.DEPENDENCY_HANDLER.dependenciesInstalled:
            with slicer.util.tryWithErrorDisplay("Failed to install required dependencies.", waitCursor=True):
                self.DEPENDENCY_HANDLER.setupPythonRequirements()

        if not inputNodes:
            raise ValueError("Input nodes are invalid")

        if not outputSegmentation:
            raise ValueError("Output segmentation is invalid")

        if model is None:
            model = self.defaultModel

        modelPath = self.modelPath(model)

        segmentationProcessInfo = SegmentationProcessInfo()

        logging.info("Processing started")

        if self.debugSkipInference:
            self.clearOutputFolder = False
            # For debugging, use a fixed temporary folder
            tempDir = self.debugSkipInferenceTempDir
        else:
            # Create new empty folder
            tempDir = slicer.util.tempDirectory()

        # Get Python executable path
        import shutil
        pythonSlicerExecutablePath = shutil.which("PythonSlicer")
        if not pythonSlicerExecutablePath:
            raise RuntimeError("Python was not found")

        # Write input volume to file
        inputFiles = []
        for inputIndex, inputNode in enumerate(inputNodes):
            if inputNode.IsA('vtkMRMLScalarVolumeNode'):
                inputImageFile = tempDir + f"/input-volume{inputIndex}.nrrd"
                logging.info(f"Writing input file to {inputImageFile}")
                volumeStorageNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
                volumeStorageNode.SetFileName(inputImageFile)
                volumeStorageNode.UseCompressionOff()
                volumeStorageNode.WriteData(inputNode)
                slicer.mrmlScene.RemoveNode(volumeStorageNode)
                inputFiles.append(inputImageFile)
            else:
                raise ValueError(f"Input node type {inputNode.GetClassName()} is not supported")

        outputSegmentationFile = tempDir + "/output-segmentation.nrrd"
        modelPtFile = modelPath.joinpath("model.pt")
        inferenceScriptPyFile = os.path.join(self.moduleDir, "Scripts", "auto3dseg_segresnet_inference.py")
        auto3DSegCommand = [ pythonSlicerExecutablePath, str(inferenceScriptPyFile),
            "--model-file", str(modelPtFile),
            "--image-file", inputFiles[0],
            "--result-file", str(outputSegmentationFile) ]
        for inputIndex in range(1, len(inputFiles)):
            auto3DSegCommand.append(f"--image-file-{inputIndex+1}")
            auto3DSegCommand.append(inputFiles[inputIndex])

        logging.info("Creating segmentations with MONAIAuto3DSeg AI...")
        logging.info(f"Auto3DSeg command: {auto3DSegCommand}")

        additionalEnvironmentVariables = None
        if cpu:
            additionalEnvironmentVariables = {"CUDA_VISIBLE_DEVICES": "-1"}
            logging.info(f"Additional environment variables: {additionalEnvironmentVariables}")

        segmentationProcessInfo.tempDir = tempDir
        segmentationProcessInfo.inputNodes = inputNodes
        segmentationProcessInfo.outputSegmentation = outputSegmentation
        segmentationProcessInfo.outputSegmentationFile = outputSegmentationFile
        segmentationProcessInfo.model = model
        segmentationProcessInfo.customData = customData

        self._bgProcess = LocalInference(processInfo=segmentationProcessInfo, logCallback=self.log, completedCallback=self.onSegmentationProcessCompleted)
        if self.debugSkipInference:
            segmentationProcessInfo.procReturnCode = 0
            self.onSegmentationProcessCompleted(segmentationProcessInfo)
        else:
            self._bgProcess.run(auto3DSegCommand, additionalEnvironmentVariables=additionalEnvironmentVariables, waitForCompletion=waitForCompletion)

        return segmentationProcessInfo

    def onSegmentationProcessCompleted(self, segmentationProcessInfo: SegmentationProcessInfo):
        procReturnCode = segmentationProcessInfo.procReturnCode
        customData = segmentationProcessInfo.customData
        cancelRequested = procReturnCode == ExitCode.USER_CANCELLED
        if not cancelRequested:
            if procReturnCode == 0:
                outputSegmentation = segmentationProcessInfo.outputSegmentation
                if self.startResultImportCallback:
                    self.startResultImportCallback(customData)

                try: # Load result
                    logging.info("Importing segmentation results...")
                    self.readSegmentation(outputSegmentation, segmentationProcessInfo.outputSegmentationFile, segmentationProcessInfo.model)

                    # Set source volume - required for DICOM Segmentation export
                    inputVolume = segmentationProcessInfo.inputNodes[0]
                    if not inputVolume.IsA('vtkMRMLScalarVolumeNode'):
                        raise ValueError("First input node must be a scalar volume")
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
                        self.endResultImportCallback(customData)
            else:
                logging.info(f"Processing failed with return code {procReturnCode}")

        tempDir = segmentationProcessInfo.tempDir
        if self.clearOutputFolder:
            logging.info("Cleaning up temporary folder.")
            if os.path.isdir(tempDir):
                import shutil
                shutil.rmtree(tempDir)
        else:
            logging.info(f"Not cleaning up temporary folder: {tempDir}")

        # Report total elapsed time
        elapsedTime = time.time() - segmentationProcessInfo.startTime
        if cancelRequested:
            logging.info(f"Processing was cancelled after {elapsedTime:.2f} seconds.")
        else:
            if procReturnCode == 0:
                logging.info(f"Processing was completed in {elapsedTime:.2f} seconds.")
            else:
                logging.info(f"Processing failed after {elapsedTime:.2f} seconds.")

        if self.processingCompletedCallback:
            self.processingCompletedCallback(procReturnCode, customData)

    def cancelProcessing(self):
        if not self._bgProcess:
            return
        self._bgProcess.stop()

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
            terminologyEntryStr = labelValueToDescription[labelValue]["terminology"]
            segmentId = labelValueToDescription[labelValue]["name"]
            self.setTerminology(outputSegmentation, segmentId, terminologyEntryStr)

    def setTerminology(self, segmentation, segmentId, terminologyEntryStr):
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
                logging.info(str(e))


class RemoteMONAIAuto3DSegLogic(MONAIAuto3DSegLogic):

    DEPENDENCY_HANDLER = RemotePythonDependencies()

    @property
    def server_address(self):
        return self._server_address

    @server_address.setter
    def server_address(self, address):
        self.DEPENDENCY_HANDLER.server_address = address
        self._server_address = address
        self._models = []

    def __init__(self):
        self._server_address = None
        MONAIAuto3DSegLogic.__init__(self)
        self._models = []

    def getMONAIPythonPackageInfo(self):
        return self.DEPENDENCY_HANDLER.installedMONAIPythonPackageInfo()

    def setupPythonRequirements(self, upgrade=False):
        self.DEPENDENCY_HANDLER.setupPythonRequirements(upgrade)
        return False

    def loadModelsDescription(self):
        if not self._server_address:
            return []
        else:
            response = requests.get(self._server_address + "/models")
            json_data = json.loads(response.text)
            return json_data

    def labelDescriptions(self, modelName):
        """Return mapping from label value to label description.
                Label description is a dict containing "name" and "terminology".
                Terminology string uses Slicer terminology entry format - see specification at
                https://slicer.readthedocs.io/en/latest/developer_guide/modules/segmentations.html#terminologyentry-tag
                """
        if not self._server_address:
            return {}
        else:
            from pathlib import Path
            tempDir = slicer.util.tempDirectory()
            tempDir = Path(tempDir)
            outfile = tempDir / "labelDescriptions.csv"
            with requests.get(self._server_address + f"/labelDescriptions?id={modelName}", stream=True) as r:
                r.raise_for_status()

                with open(outfile, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            labelDescriptions = self._labelDescriptions(outfile)

            import shutil
            shutil.rmtree(tempDir)
            return labelDescriptions

    def process(self, inputNodes, outputSegmentation, modelId=None, cpu=False, waitForCompletion=True, customData=None):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputNodes: input nodes in a list
        :param outputVolume: thresholding result
        :param modelId: one of self.models
        :param cpu: use CPU instead of GPU
        :param waitForCompletion: if True then the method waits for the processing to finish
        :param customData: any custom data to identify or describe this processing request, it will be returned in the process completed callback when waitForCompletion is False
        """

        segmentationProcessInfo = SegmentationProcessInfo()
        logging.info("Processing started")

        tempDir = slicer.util.tempDirectory()
        outputSegmentationFile = tempDir + "/output-segmentation.nrrd"

        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=tempDir) as temp_dir:
            # Write input volume to file
            from pathlib import Path
            tempDir = Path(temp_dir)
            inputFiles = []
            for inputIndex, inputNode in enumerate(inputNodes):
                if inputNode.IsA('vtkMRMLScalarVolumeNode'):
                    inputImageFile = tempDir / f"input-volume{inputIndex}.nrrd"
                    logging.info(f"Writing input file to {inputImageFile}")
                    volumeStorageNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
                    volumeStorageNode.SetFileName(inputImageFile)
                    volumeStorageNode.UseCompressionOff()
                    volumeStorageNode.WriteData(inputNode)
                    slicer.mrmlScene.RemoveNode(volumeStorageNode)
                    inputFiles.append(inputImageFile)
                else:
                    raise ValueError(f"Input node type {inputNode.GetClassName()} is not supported")

            logging.info(f"Initiating Inference on {self._server_address}")
            files = {}

            for idx, inputFile in enumerate(inputFiles, start=1):
                name = "image_file"
                if idx > 1:
                    name = f"{name}_{idx}"
                files[name] = open(inputFile, 'rb')

            r = None
            try:
                with requests.post(self._server_address + f"/infer?model_name={modelId}", files=files) as r:
                    r.raise_for_status()

                    with open(outputSegmentationFile, "wb") as binary_file:
                        for chunk in r.iter_content(chunk_size=8192):
                            binary_file.write(chunk)

                    segmentationProcessInfo.procReturnCode = 0
            except requests.exceptions.HTTPError as e:
                from http import HTTPStatus
                status = HTTPStatus(e.response.status_code)
                logging.debug(f"Server response content: {r.content}")
                raise RuntimeError(status.description)
            finally:
                for f in files.values():
                    f.close()

        segmentationProcessInfo.tempDir = tempDir
        segmentationProcessInfo.inputNodes = inputNodes
        segmentationProcessInfo.outputSegmentation = outputSegmentation
        segmentationProcessInfo.outputSegmentationFile = outputSegmentationFile
        segmentationProcessInfo.model = modelId
        segmentationProcessInfo.customData = customData

        self.onSegmentationProcessCompleted(segmentationProcessInfo)

        return segmentationProcessInfo


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

        # Logic testing is disabled by default to not overload automatic build machines (pytorch is a huge package and computation
        # on CPU takes 5-10 minutes). Set testLogic to True to enable testing.
        testLogic = True

        if not testLogic:
            self.delayDisplay("Logic testing is disabled. Set testLogic to True to enable it.")
            return

        logic = MONAIAuto3DSegLogic()
        logic.logCallback = self._mylog

        self.delayDisplay("Set up required Python packages")
        logic.setupPythonRequirements()

        testResultsPath = logic.fileCachePath.joinpath("ModelsTestResults")
        if not os.path.exists(testResultsPath):
            os.makedirs(testResultsPath)

        import json
        modelsTestResultsJsonFilePath = os.path.join(testResultsPath.joinpath("ModelsTestResults.json"))
        if os.path.exists(modelsTestResultsJsonFilePath):
            # resume testing
            with open(modelsTestResultsJsonFilePath) as f:
              models = json.load(f)
        else:
            # start testing from scratch
            models = logic.models

        import PyTorchUtils
        pytorchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if pytorchLogic.cuda:
            # CUDA is available, test on both CPU and GPU
            configurations = [{"forceUseCPU": False}, {"forceUseCPU": True}]
        else:
            # CUDA is not available, only test on CPU
            configurations = [{"forceUseCPU": True}]

        for configurationIndex, configuration in enumerate(configurations):
            forceUseCpu = configuration["forceUseCPU"]
            configurationName = "CPU" if forceUseCpu else "GPU"

            for modelIndex, model in enumerate(models):
                if model.get("deprecated"):
                    # Do not teset deprecated models
                    continue

                segmentationTimePropertyName = "segmentationTimeSec"+configurationName
                if segmentationTimePropertyName in models[modelIndex]:
                    # Skip already tested models
                    continue

                self.delayDisplay(f"Testing {model['title']} (v{model['version']})")
                slicer.mrmlScene.Clear()

                # Download sample data for model input

                sampleDataName = model.get("sampleData")
                if not sampleDataName:
                    self.delayDisplay(f"Sample data not available for {model['title']}")
                    continue

                if type(sampleDataName) == list:
                    # For now, always just use the first data set if multiple data sets are provided
                    sampleDataName = sampleDataName[0]

                import SampleData
                loadedSampleNodes = SampleData.SampleDataLogic().downloadSamples(sampleDataName)
                if not loadedSampleNodes:
                    raise RuntimeError(f"Failed to load sample data set '{sampleDataName}'.")

                # Set model inputs
                inputs = model.get("inputs")
                inputNodes = logic.assignInputNodesByName(inputs, loadedSampleNodes)

                outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

                # Run the segmentation

                self.delayDisplay(f"Running segmentation for {model['title']}...")
                startTime = time.time()
                logic.process(inputNodes, outputSegmentation, model["id"], forceUseCpu)
                segmentationTimeSec = time.time() - startTime

                # Save segmentation time (rounded to 0.1 sec) into model description
                models[modelIndex][segmentationTimePropertyName] = round(segmentationTimeSec * 10) / 10

                # Save all segment names into model description
                labelDescriptions = logic.labelDescriptions(model["id"])
                segmentNames = []
                for terminology in labelDescriptions.values():
                    contextName, category, typeStr, typeModifier, anatomicContext, region, regionModifier = terminology["terminology"].split("~")
                    typeName = typeStr.split("^")[2]
                    typeModifierName = typeModifier.split("^")[2]
                    if typeModifierName:
                        typeName = f"{typeModifierName} {typeName}"
                    regionName = region.split("^")[2]
                    regionModifierName = regionModifier.split("^")[2]
                    if regionModifierName:
                        regionName = f"{regionModifierName} {regionName}"
                    name = f"{typeName} in {regionName}" if regionName else typeName
                    segmentNames.append(name)
                models[modelIndex]["segmentNames"] = segmentNames

                sliceScreenshotFilename, rotate3dScreenshotFilename = self._writeScreenshots(outputSegmentation, testResultsPath, model["id"]+"-"+configurationName)
                if configurationIndex == 0:
                    # Use screenshot computed during the first configuration
                    models[modelIndex]["segmentationResultsScreenshot2D"] = sliceScreenshotFilename.name
                    models[modelIndex]["segmentationResultsScreenshot3D"] = rotate3dScreenshotFilename.name

                # Write results to file (to allow accessing the results before all tests complete)
                with open(modelsTestResultsJsonFilePath, 'w') as f:
                    json.dump(models, f, indent=2)

        logic.updateModelsDescriptionJsonFilePathFromTestResults(modelsTestResultsJsonFilePath)
        self._writeTestResultsToMarkdown(modelsTestResultsJsonFilePath)

        self.delayDisplay("Test passed")

    def _mylog(self,text):
        print(text)

    def _writeScreenshots(self, segmentationNode, outputPath, baseName, numberOfImages=25, lightboxColumns=5, numberOfVideoFrames=50):
        import ScreenCapture
        cap = ScreenCapture.ScreenCaptureLogic()

        sliceScreenshotFilename = outputPath.joinpath(f"{baseName}-slices.png")
        rotate3dScreenshotFilename = outputPath.joinpath(f"{baseName}-rotate3d.gif")  # gif, mp4, png
        videoLengthSec = 5

        # Capture slice sweep
        sliceScreenshotsFilenamePattern = outputPath.joinpath("slices_%04d.png")
        cap.showViewControllers(False)
        slicer.app.layoutManager().resetSliceViews()
        sliceNode = slicer.util.getNode("vtkMRMLSliceNodeRed")
        sliceOffsetMin, sliceOffsetMax = cap.getSliceOffsetRange(sliceNode)
        sliceOffsetStart = sliceOffsetMin + (sliceOffsetMax - sliceOffsetMin) * 0.05
        sliceOffsetEnd = sliceOffsetMax - (sliceOffsetMax - sliceOffsetMin) * 0.05
        cap.captureSliceSweep(
            sliceNode, sliceOffsetStart, sliceOffsetEnd, numberOfImages,
            sliceScreenshotsFilenamePattern.parent, sliceScreenshotsFilenamePattern.name,
            captureAllViews=None, transparentBackground=False)
        cap.showViewControllers(True)

        # Create lightbox image
        cap.createLightboxImage(lightboxColumns,
            sliceScreenshotsFilenamePattern.parent,
            sliceScreenshotsFilenamePattern.name,
            numberOfImages,
            sliceScreenshotFilename)
        cap.deleteTemporaryFiles(sliceScreenshotsFilenamePattern.parent, sliceScreenshotsFilenamePattern.name, numberOfImages)

        # Capture 3D rotation
        rotate3dScreenshotsFilenamePattern = outputPath.joinpath("rotate3d_%04d.png")
        segmentationNode.CreateClosedSurfaceRepresentation()
        segmentationNode.GetDisplayNode().SetOpacity3D(0.6)

        if rotate3dScreenshotFilename.suffix.lower() == ".png":
            video = False
            numberOfImages3d = numberOfImages
        else:
            video = True
            numberOfImages3d = numberOfVideoFrames
            if rotate3dScreenshotFilename.suffix.lower() == ".gif":
                # animated GIF
                extraOptions = "-filter_complex palettegen,[v]paletteuse"
            elif rotate3dScreenshotFilename.suffix.lower() == ".mp4":
                # H264 high-quality
                extraOptions = "-codec libx264 -preset slower -crf 18 -pix_fmt yuv420p"
            else:
                raise ValueError(f"Unsupported format: {rotate3dScreenshotFilename.suffix}")

        viewLabel = "1"
        viewNode = slicer.vtkMRMLViewLogic().GetViewNode(slicer.mrmlScene, viewLabel)
        viewNode.SetBackgroundColor(0,0,0)
        viewNode.SetBackgroundColor2(0,0,0)
        viewNode.SetAxisLabelsVisible(False)
        viewNode.SetBoxVisible(False)
        cap.showViewControllers(False)
        slicer.app.layoutManager().resetThreeDViews()
        cap.capture3dViewRotation(viewNode, -180, 180, numberOfImages3d, ScreenCapture.AXIS_YAW, rotate3dScreenshotsFilenamePattern.parent, rotate3dScreenshotsFilenamePattern.name)
        cap.showViewControllers(True)

        if video:
            cap.createVideo(numberOfImages3d/videoLengthSec, extraOptions, rotate3dScreenshotsFilenamePattern.parent, rotate3dScreenshotsFilenamePattern.name, rotate3dScreenshotFilename)
        else:
            cap.createLightboxImage(lightboxColumns,
                rotate3dScreenshotsFilenamePattern.parent,
                rotate3dScreenshotsFilenamePattern.name,
                numberOfImages3d,
                rotate3dScreenshotFilename)

        cap.deleteTemporaryFiles(rotate3dScreenshotsFilenamePattern.parent, rotate3dScreenshotsFilenamePattern.name, numberOfImages3d)

        return sliceScreenshotFilename, rotate3dScreenshotFilename

    def _writeTestResultsToMarkdown(self, modelsTestResultsJsonFilePath, modelsTestResultsMarkdownFilePath=None, screenshotUrlBase=None):
        if modelsTestResultsMarkdownFilePath is None:
            modelsTestResultsMarkdownFilePath = modelsTestResultsJsonFilePath.replace(".json", ".md")
        if screenshotUrlBase is None:
            screenshotUrlBase = "https://github.com/lassoan/SlicerMONAIAuto3DSeg/releases/download/ModelsTestResults/"

        import json
        with open(modelsTestResultsJsonFilePath) as f:
            modelsTestResults = json.load(f)

        with open(modelsTestResultsMarkdownFilePath, 'w', newline="\n") as f:
            f.write("# 3D Slicer MONAI Auto3DSeg models\n\n")
            # Write hardware information (only on Windows for now)
            if os.name == "nt":
                import subprocess
                cpu = subprocess.check_output('wmic cpu get name', stderr=open(os.devnull, 'w')).decode('utf-8').partition('Name')[2].strip(' \r\n')
                systemInfoStr = subprocess.check_output('systeminfo', stderr=open(os.devnull, 'w')).decode('utf-8')
                # System information has a line like this: "Total Physical Memory:     32,590 MB"
                import re
                ram = re.search(r"Total Physical Memory:(.+)", systemInfoStr).group(1).strip()
                f.write(f"Testing hardware: {cpu}, {ram}")
                import torch
                for i in range(torch.cuda.device_count()):
                    gpuProperties = torch.cuda.get_device_properties(i)
                    f.write(f", {gpuProperties.name} {round(torch.cuda.get_device_properties(0).total_memory/(2**30))}GB")
                f.write("\n\n")
            # Write test results
            for model in modelsTestResults:
                if model["deprecated"]:
                    continue
                title = f"{model['title']} (v{model['version']})"
                f.write(f"## {title}\n")
                f.write(f"{model['description']}\n\n")
                f.write(f"Processing time: {humanReadableTimeFromSec(model['segmentationTimeSecGPU'])} on GPU, {humanReadableTimeFromSec(model['segmentationTimeSecCPU'])} on CPU\n\n")
                f.write(f"Segment names: {', '.join(model['segmentNames'])}\n\n")
                f.write(f"![2D view]({screenshotUrlBase}{model['segmentationResultsScreenshot2D']})\n")
                f.write(f"![3D view]({screenshotUrlBase}{model['segmentationResultsScreenshot3D']})\n")
