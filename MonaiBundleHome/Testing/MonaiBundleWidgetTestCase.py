from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import slicer

from .Utils import MonaiBundleTestCase, download_ct_chest_sample_data, get_test_bundle_path
from MonaiBundleHomeLib import MonaiBundleWidget


class MonaiBundleWidgetTestCase(MonaiBundleTestCase):
    def test_can_be_displayed(self):
        widget = MonaiBundleWidget()
        widget.show()
        slicer.app.processEvents()

    @pytest.mark.slow
    def test_can_download_and_run_bundle_from_model_zoo(self):
        testVolumeNode = download_ct_chest_sample_data()

        with TemporaryDirectory() as tmp_dir:
            widget = MonaiBundleWidget()
            widget.modelZooSelectionBox.setCurrentIndex(0)
            widget.bundleStorageDir = Path(tmp_dir)
            widget.downloadModelFromZooButton.clicked.emit()
            widget.inputSelector.setCurrentNode(testVolumeNode)

            assert widget.runButton.isEnabled()
            widget.runButton.clicked.emit()

            nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
            assert nodes
            assert nodes[0].GetSegmentation().GetNumberOfSegments()

    @pytest.mark.slow
    def test_can_load_and_run_local_bundle(self):
        testVolumeNode = download_ct_chest_sample_data()
        testBundlePath = get_test_bundle_path()

        widget = MonaiBundleWidget()
        widget.inputSelector.setCurrentNode(testVolumeNode)
        widget.bundleRootDirSelector.setCurrentPath(testBundlePath)

        assert widget.runButton.isEnabled()
        widget.runButton.clicked.emit()

        nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        assert nodes
        assert nodes[0].GetSegmentation().GetNumberOfSegments()