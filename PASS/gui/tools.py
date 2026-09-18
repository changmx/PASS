"""Top-level tools workspace; calculations never change the active input."""
from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from PASS.gui.appearance import THEMES, icon
from PASS.gui.tool_beam import BeamCalculatorPage


class ToolNavigation(QFrame):
    """Exclusive page buttons using the configuration library's section style."""
    currentRowChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("libraryPanel")
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 6, 8)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = []
        for index, title in enumerate(("束流参数计算器", "共振线图", "RF bucket绘制", "相空间绘制及发射度计算", "磁铁参数换算", "激励计算")):
            button = QPushButton(title)
            button.setObjectName("librarySectionHeader")
            button.setProperty("depth", 0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setCheckable(True)
            button.setIconSize(QSize(16, 16))
            self.group.addButton(button, index)
            self.buttons.append(button)
            layout.addWidget(button)
        layout.addStretch()
        self.group.idClicked.connect(self.currentRowChanged)

    def setCurrentRow(self, index):
        if 0 <= index < len(self.buttons) and not self.buttons[index].isChecked():
            self.buttons[index].click()

    def set_theme(self, theme):
        for button, glyph in zip(self.buttons, ("calculator", "resonance", "rf_bucket", "phase_ellipse", "magnet", "exciter")):
            button.setIcon(icon(glyph, THEMES[theme]["muted"]))


class ToolsPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.tune = None
        self.rf = self.emittance = self.magnets = self.exciter = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.navigation = ToolNavigation()
        self.navigation.set_theme(self.theme)
        layout.addWidget(self.navigation)
        self.stack = QStackedWidget()
        self.beam = BeamCalculatorPage()
        self.stack.addWidget(self.beam)
        for _ in range(5):
            self.stack.addWidget(QWidget())
        layout.addWidget(self.stack, 1)
        self.navigation.currentRowChanged.connect(self._select)
        self.navigation.setCurrentRow(0)

    def _select(self, index):
        attribute = {1: "tune", 2: "rf", 3: "emittance", 4: "magnets", 5: "exciter"}.get(index)
        if attribute and getattr(self, attribute) is None:
            # Scientific plotting modules load only when their tool is opened.
            if index == 1:
                from PASS.gui.tool_tune import TuneDiagramPage
                page = TuneDiagramPage()
            elif index == 2:
                from PASS.gui.tool_rf import RFBucketPage
                page = RFBucketPage(lambda: self.beam.kinematics)
            elif index == 3:
                from PASS.gui.tool_emittance import EmittancePage
                page = EmittancePage(lambda: self.beam.kinematics)
            elif index == 4:
                from PASS.gui.tool_magnets import MagnetPage
                page = MagnetPage(lambda: self.beam.kinematics)
            else:
                from PASS.gui.tool_exciter import ExciterPage
                page = ExciterPage(lambda: self.beam.kinematics)
            setattr(self, attribute, page)
            old = self.stack.widget(index)
            self.stack.removeWidget(old)
            old.deleteLater()
            self.stack.insertWidget(index, page)
            page.set_theme(self.theme)
        self.stack.setCurrentIndex(index)

    def set_theme(self, theme):
        self.theme = theme
        self.navigation.set_theme(theme)
        for page in (self.tune, self.rf, self.emittance, self.magnets, self.exciter):
            if page is not None:
                page.set_theme(theme)
