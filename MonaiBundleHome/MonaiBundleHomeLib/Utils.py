import ctk
import qt


def createButton(name, callback=None, isCheckable=False, icon=None):
    """Helper function to create a button with a text, callback on click and checkable status

    Parameters
    ----------
    name: str
      Label of the button
    callback: Callable
      Called method when button is clicked
    isCheckable: bool
      If true, the button will be checkable

    Returns
    -------
    QPushButton
    """
    button = qt.QPushButton(name)
    if callback is not None:
        button.connect("clicked(bool)", callback)
    if icon:
        button.setIcon(icon)
    button.setCheckable(isCheckable)
    return button


def wrapInCollapsibleButton(layout, collapsibleText, isCollapsed=True, wrapInQFrame=False):
    """Wraps input layout into a collapsible button.
    collapsibleText is writen next to collapsible button. Initial collapsed status is customizable
    (collapsed by default)

    :returns ctkCollapsibleButton
    """
    collapsibleLayout = qt.QVBoxLayout()
    collapsibleLayout.setContentsMargins(0, 0, 0, 0)
    if wrapInQFrame:
        collapsibleLayout.addWidget(wrapInFrameArea(layout))
    else:
        collapsibleLayout.addLayout(layout)

    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = collapsibleText
    collapsibleButton.collapsed = isCollapsed
    collapsibleButton.setLayout(collapsibleLayout)
    return collapsibleButton


def wrapInFrameArea(item):
    frame = qt.QFrame()
    frame.setProperty("frameArea", True)
    try:
        frame.setLayout(item)
    except ValueError:
        layout = qt.QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(item)
    return frame


def messageBox(title, message, messageType=qt.QMessageBox.Information):
    """Opens a message box while handling the stay on top flag when main window is always on Top"""
    dlg = qt.QMessageBox(messageType, title, message)
    if strToBool(qt.QSettings().value("windowAlwaysOnTop")):
        dlg.setWindowFlag(qt.Qt.WindowStaysOnTopHint, True)
    dlg.exec()


def informationMessageBox(title, message):
    """Opens an information message box while handling the stay on top flag"""
    messageBox(title, message, qt.QMessageBox.Information)


def warningMessageBox(title, message):
    """Opens a warning message box while handling the stay on top flag"""
    messageBox(title, message, qt.QMessageBox.Warning)


def criticalMessageBox(title, message):
    """Opens a critical message box while handling the stay on top flag"""
    messageBox(title, message, qt.QMessageBox.Critical)


def strToBool(value):
    """Convert input string to bool"""
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ['true', 'yes', '1', 'y', 't']
    return False