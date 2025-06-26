from pathlib import Path
from unittest.mock import MagicMock

import pytest
from MonaiBundleHomeLib import MonaiBundleLogic
from tempfile import TemporaryDirectory

from .Utils import MonaiBundleTestCase, get_test_bundle_path

class MonaiBundleLogicTestCase(MonaiBundleTestCase):
    def setUp(self):
        super().setUp()
        self.logic = MonaiBundleLogic()
        self.testBundlePath = get_test_bundle_path()

    def test_can_load_bundle(self):
        bundleLoadedMock = MagicMock()
        self.logic.bundleLoaded.connect(bundleLoadedMock)
        self.logic.loadBundleConfig(self.testBundlePath.joinpath("configs", "inference.json"))
        bundleLoadedMock.assert_called_once()

    @pytest.mark.slow
    def test_can_download_bundle_from_zoo(self):
        with TemporaryDirectory() as tmp_dir:
            modelName = "brats_mri_segmentation"
            self.logic.downloadBundleFromModelZoo(modelName, "1.0.6", tmp_dir)
            assert Path(tmp_dir).joinpath(modelName, "configs", "metadata.json").is_file()
