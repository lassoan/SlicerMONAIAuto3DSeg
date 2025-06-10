import pytest

from MonaiBundleHomeLib import MonaiBundleWidget

class MonaiBundleWidgetTestCase:
    def test_can_be_displayed(self):
        widget = SegmentationWidget()
        widget.show()
        slicer.app.processEvents()

