import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, \
    ScriptedLoadableModuleTest
from slicer.util import VTKObservationMixin

from MonaiBundleHomeLib import MonaiBundleLogic, MonaiBundleWidget, PythonDependencyChecker


class MonaiBundleHome(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Monai Bundle Runner"
        self.parent.categories = ["Monai"]
        self.parent.dependencies = []
        self.parent.contributors = [""]
        self.parent.helpText = """Import and use MONAI bundles in 3D Slicer. \n
        This modules allows you test AI models from the <a href=https://monai.io/model-zoo.html>MONAI Model Zoo</a> 
        on your data withing 3D Slicer. For now, only segmentation tasks are supported."""
        self.parent.acknowledgementText = ""


class MonaiBundleHomeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Home Widget for KeriMedical.

    Responsible for the instantiation of the different widgets of the application
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = MonaiBundleLogic()
        self.ui = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self.layout.addWidget(MonaiBundleWidget() if PythonDependencyChecker.areDependenciesSatisfied()
            else PythonDependencyChecker.downloadDependencyWidget())
        self.layout.addStretch()

    def onReload(self):
        import importlib
        import MonaiBundleHomeLib
        import Testing
        importlib.reload(MonaiBundleHomeLib)
        importlib.reload(Testing)

        super(MonaiBundleHomeWidget, self).onReload()
        slicer.mrmlScene.Clear()


class MonaiBundleHomeTest(ScriptedLoadableModuleTest):
    """
    Unittest definition executable from the Slicer environment
    """

    def runTest(self):
        import unittest
        import Testing

        # Clear slicer scene before running the unittests
        slicer.mrmlScene.Clear()

        # Gather tests for the plugin and run them in a test suite
        testCases = []
        for elt in dir(Testing):
            attr = getattr(Testing, elt)
            if isinstance(attr, unittest.TestCase):
                testCases.append(attr)

        print("Discovered tests cases : ", testCases)
        suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(case) for case in testCases])
        unittest.TextTestRunner(verbosity=3).run(suite)

        # Clear scene after execution
        slicer.mrmlScene.Clear()
