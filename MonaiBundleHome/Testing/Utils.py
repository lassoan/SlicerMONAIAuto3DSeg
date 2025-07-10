import unittest
from pathlib import Path

import SampleData
import slicer


class MonaiBundleTestCase(unittest.TestCase):
    def setUp(self):
        self._clearScene()

    @staticmethod
    def _clearScene():
        slicer.app.processEvents()
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def tearDown(self):
        slicer.app.processEvents()


def _dataFolderPath():
    return Path(r"\\wheezy\DevApp\Projects\Cosy\MonaiBundle\TestingData")


def get_test_bundle_path():
    return _dataFolderPath().joinpath("TestBundles", "wholeBody_ct_segmentation")


def download_ct_chest_sample_data():
    from slicer.util import TESTING_DATA_URL

    dataSource = SampleData.SampleDataSource(
        nodeNames="CTChest",
        fileNames="CT-chest.nrrd",
        uris=TESTING_DATA_URL + "SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e"
    )
    loadedNode = SampleData.SampleDataLogic().downloadFromSource(dataSource)[0]
    return loadedNode
