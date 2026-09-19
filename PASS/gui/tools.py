"""Top-level tools workspace; calculations never change the active input."""
from PySide6.QtCore import QSize, Signal, QThread, QTimer
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget, QLabel, QApplication

from PASS.gui.appearance import THEMES, icon


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
        for index, title in enumerate(("束流参数计算器", "共振线图", "RF bucket绘制", "相空间绘制及发射度计算", "磁铁参数换算", "激励计算", "数据格式转换")):
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
        for button, glyph in zip(self.buttons, ("calculator", "resonance", "rf_bucket", "phase_ellipse", "magnet", "exciter", "folder")):
            button.setIcon(icon(glyph, THEMES[theme]["muted"]))


class ModulePreloader(QThread):
    """Prepare numerical dependencies off the GUI thread; never create widgets."""
    progress = Signal(str)

    def run(self):
        import importlib
        self.errors = []
        for name in ("pandas", "scipy.interpolate", "h5py", "tfs", "matplotlib.figure", "matplotlib.mathtext", "sdds", "turn_by_turn"):
            if self.isInterruptionRequested():
                break
            self.progress.emit("正在准备工具资源…")
            try:
                importlib.import_module(name)
            except Exception as exc:
                self.errors.append(f"{name}: {exc}")


class ToolsPage(QWidget):
    preparation_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.beam = self.tune = self.rf = self.emittance = self.magnets = self.exciter = self.conversion = None
        self._attributes = ("beam", "tune", "rf", "emittance", "magnets", "exciter", "conversion")
        self._pending = list(range(len(self._attributes)))
        self._failed = {}
        self._requested = 0
        self._ready = False
        self._closing = False
        self.pause_preload = lambda: False
        self._pending_file = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.navigation = ToolNavigation()
        self.navigation.set_theme(self.theme)
        layout.addWidget(self.navigation)
        self.stack = QStackedWidget()
        for _ in self._attributes:
            page = QWidget()
            body = QVBoxLayout(page)
            label = QLabel("正在准备此工具，请稍候…")
            label.setWordWrap(True)
            body.addWidget(label)
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)
        self.navigation.currentRowChanged.connect(self._select)
        self.navigation.setCurrentRow(0)
        self.preloader = ModulePreloader(self)
        self.preloader.progress.connect(self.preparation_changed)
        self.preloader.finished.connect(self._imports_ready)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._prepare_next)

    def start_preload(self):
        if not self.preloader.isRunning() and not self._ready and not self._closing:
            self.preloader.start()

    def _imports_ready(self):
        if not self._closing:
            self._ready = True
            self.timer.start(30)

    def _select(self, index):
        self._requested = index
        self.stack.setCurrentIndex(index)
        if hasattr(self, "timer") and self._ready:
            if index in self._failed:
                self._failed.pop(index)
                self._pending.append(index)
            self.timer.start(0)

    def _create_page(self, index):
        if index == 0:
            from PASS.gui.tool_beam import BeamCalculatorPage
            page = BeamCalculatorPage()
        elif index == 1:
            from PASS.gui.tool_tune import TuneDiagramPage
            page = TuneDiagramPage()
        elif index == 2:
            from PASS.gui.tool_rf import RFBucketPage
            page = RFBucketPage(lambda: self.beam.kinematics if self.beam else None)
        elif index == 3:
            from PASS.gui.tool_emittance import EmittancePage
            page = EmittancePage(lambda: self.beam.kinematics if self.beam else None)
        elif index == 4:
            from PASS.gui.tool_magnets import MagnetPage
            page = MagnetPage(lambda: self.beam.kinematics if self.beam else None)
        elif index == 5:
            from PASS.gui.tool_exciter import ExciterPage
            page = ExciterPage(lambda: self.beam.kinematics if self.beam else None)
        else:
            from PASS.gui.tool_conversion import ConversionPage
            page = ConversionPage()
        if hasattr(page, "set_theme") and getattr(page, "theme", None) != self.theme:
            page.set_theme(self.theme)
        setattr(self, self._attributes[index], page)
        old = self.stack.widget(index)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.stack.insertWidget(index, page)
        self.stack.setCurrentIndex(self._requested)
        if index == 6 and self._pending_file:
            path, self._pending_file = self._pending_file, None
            page.open_path(path)

    def _prepare_next(self):
        if self._closing or not self._pending:
            return
        requested = self._requested if self.isVisible() else None
        busy = self.pause_preload() or bool(self.conversion and self.conversion.busy) or QApplication.activeModalWidget() is not None
        if busy:
            self.timer.start(250)
            return
        index = requested if requested in self._pending else self._pending[0]
        self._pending.remove(index)
        try:
            self._create_page(index)
        except Exception as exc:
            self._failed[index] = str(exc)
            self.stack.widget(index).findChild(QLabel).setText(f"工具准备失败：{exc}\n再次选择此栏目可重试。")
        prepared = len(self._attributes) - len(self._pending) - len(self._failed)
        message = f"正在准备工具（{prepared}/{len(self._attributes)}）…" if self._pending else ("工具已就绪" if not self._failed else "部分工具未就绪，可进入对应栏目查看原因")
        self.preparation_changed.emit(message)
        if self._pending:
            self.timer.start(180)

    def open_conversion(self, path):
        self._pending_file = str(path)
        self.navigation.setCurrentRow(6)
        if self.conversion:
            self._pending_file = None
            self.conversion.open_path(path)

    def set_theme(self, theme):
        self.theme = theme
        self.navigation.set_theme(theme)
        for attribute in self._attributes:
            page = getattr(self, attribute)
            if page is not None and hasattr(page, "set_theme"):
                page.set_theme(theme)

    def shutdown(self):
        self._closing = True
        self.timer.stop()
        if self.conversion:
            self.conversion.shutdown()
        if self.preloader.isRunning():
            self.preloader.requestInterruption()
            return False
        return True
