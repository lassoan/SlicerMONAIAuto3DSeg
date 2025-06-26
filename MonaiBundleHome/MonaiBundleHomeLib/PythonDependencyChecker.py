import slicer

from .Utils import createButton


class PythonDependencyChecker:
    """
    Class responsible for installing the Modules dependencies
    """

    @classmethod
    def areDependenciesSatisfied(cls):
        try:
            import torch
            import monai
            return True
        except ImportError:
            return False

    @classmethod
    def installDependenciesIfNeeded(cls, progressDialog=None):
        if cls.areDependenciesSatisfied():
            return

        progressDialog = progressDialog or slicer.util.createProgressDialog(maximum=0)
        progressDialog.labelText = "Installing PyTorch"

        # nnUNet requires at least PyTorch version 2.0 to work (otherwise, inference will return torch._dynamo import
        # error).
        torch_version = ">=2.0.0"
        try:
            # Try to install the best available pytorch version for the environment using the PyTorch Slicer extension
            import PyTorchUtils
            PyTorchUtils.PyTorchUtilsLogic().installTorch(torchVersionRequirement=torch_version)
        except ImportError:
            # Fallback on default torch available on PIP
            slicer.util.pip_install(f"torch{torch_version}")

        progressDialog.labelText = "Installing monai"
        slicer.util.pip_install("monai[nibabel,itk,ignite,fire,einops]==1.3.0")

    @classmethod
    def downloadDependenciesAndRestart(cls):
        progressDialog = slicer.util.createProgressDialog(maximum=0)

        extensionManager = slicer.app.extensionsManagerModel()
        if not extensionManager.isExtensionInstalled("PyTorch"):
            progressDialog.labelText = "Installing the PyTorch Slicer extension"
            extensionManager.interactive = False  # avoid popups
            extensionManager.installExtensionFromServer("PyTorch")

        cls.installDependenciesIfNeeded(progressDialog)

        slicer.app.restart()

    @classmethod
    def downloadDependencyWidget(cls, parent=None):
        import qt
        widget = qt.QWidget(parent)
        layout = qt.QVBoxLayout(widget)

        error_msg = (
            "Pytorch and monai are required by this plugin.\n"
            "Please click on the Download button below to download\n"
            "and install these dependencies. (This may take several minutes)"
        )
        layout.addWidget(qt.QLabel(error_msg))
        layout.addWidget(createButton("Download dependencies and restart", cls.downloadDependenciesAndRestart))
        return widget
