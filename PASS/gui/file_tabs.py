"""Independent file tabs for plotting and numerical analysis."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from PASS.gui.widgets import file_dialog_directory


class _FileWorkspace(QWidget):
    """Retain closing pages until all of their background work has stopped."""

    shutdown_finished = Signal()

    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self._activated = False
        self._closing = False
        self._shutdown_notified = False
        self._pages = set()
        self._closing_pages = set()
        self._paths = {}
        self._wired_sources = set()
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        controls = QHBoxLayout()
        self.open_button = QPushButton("打开文件…")
        self.open_button.setToolTip("可同时选择多个文件；每个文件在独立标签页中打开。")
        self.open_button.clicked.connect(self.choose_files)
        self.new_button = QPushButton("新建标签页")
        self.new_button.clicked.connect(lambda: self.new_tab())
        controls.addWidget(self.open_button)
        controls.addWidget(self.new_button)
        controls.addWidget(QLabel("每个标签页独立保存数据、设置和视图"))
        controls.addStretch()
        root.addLayout(controls)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setElideMode(Qt.ElideMiddle)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._current_changed)
        self.tabs.tabBar().tabMoved.connect(self._refresh_labels)
        root.addWidget(self.tabs, 1)
        self.new_tab()

    @property
    def current_page(self):
        return self.tabs.currentWidget()

    @property
    def busy(self):
        return any(page.busy for page in self._pages)

    def activate(self):
        if self._closing:
            return
        self._activated = True
        self._current_changed()

    def _current_changed(self, *_args):
        if self._activated and not self._closing and self.current_page is not None:
            self._activate_page(self.current_page)

    def new_tab(self, *, kind=None):
        if self._closing:
            return None
        page = self._make_page(kind)
        self._pages.add(page)
        self._paths[page] = ""
        page.shutdown_finished.connect(lambda current=page: self._page_stopped(current))
        self._connect_page(page)
        self._apply_theme(page)
        index = self.tabs.addTab(page, "未打开文件")
        self.tabs.setCurrentIndex(index)
        self._refresh_labels()
        return page

    def _available_page(self, kind=None):
        candidates = [self.current_page] + [self.tabs.widget(index) for index in range(self.tabs.count())]
        for page in candidates:
            if page is not None and not page.busy and self._is_empty(page):
                self.tabs.setCurrentWidget(page)
                return page
        return self.new_tab(kind=kind)

    def _is_empty(self, page):
        return not self._paths[page]

    def _set_path(self, page, path):
        if page not in self._pages or page in self._closing_pages:
            return
        self._paths[page] = str(Path(path).expanduser().resolve()) if str(path).strip() else ""
        self._refresh_labels()

    def _refresh_labels(self, *_args):
        counts = {}
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            path = self._paths[page]
            name = Path(path).name if path else "未打开文件"
            counts[name] = counts.get(name, 0) + 1
            suffix = f" · {counts[name]}" if counts[name] > 1 else ""
            self.tabs.setTabText(index, name + suffix)
            self.tabs.setTabToolTip(index, path or "打开文件，或在页面内选择数据。")

    def close_tab(self, index):
        page = self.tabs.widget(index)
        if page is None or page in self._closing_pages:
            return
        kind = getattr(self, "current_kind", None)
        self._closing_pages.add(page)
        self.tabs.removeTab(index)
        page.hide()
        # removeTab does not destroy the page; keep ownership through shutdown.
        if page.shutdown():
            self._dispose_page(page)
        if not self._closing and not self.tabs.count():
            self.new_tab(kind=kind)
        self._refresh_labels()

    def _dispose_page(self, page):
        if page not in self._pages or page.busy:
            return
        self._closing_pages.discard(page)
        self._pages.remove(page)
        self._wired_sources.discard(page)
        self._paths.pop(page, None)
        page.deleteLater()

    def _page_stopped(self, page):
        if page not in self._pages or page.busy:
            return
        if page in self._closing_pages:
            self._dispose_page(page)
        if self._closing and not self.busy and not self._shutdown_notified:
            self._shutdown_notified = True
            self.shutdown_finished.emit()

    def shutdown(self):
        self._closing = True
        self.open_button.setEnabled(False)
        self.new_button.setEnabled(False)
        for page in tuple(self._pages):
            if page.shutdown() and page in self._closing_pages:
                self._dispose_page(page)
        return not self.busy

    def set_theme(self, theme):
        if self.theme == theme:
            return
        self.theme = theme
        for page in tuple(self._pages - self._closing_pages):
            self._apply_theme(page)

    def _apply_theme(self, page):
        page.update()


class PlotWorkspace(_FileWorkspace):
    """Open one independent PlotPage per file, retaining comparison within a tab."""

    def _make_page(self, kind):
        from PASS.gui.plotting import PlotPage

        page = PlotPage()
        page.comparison_load_button.setText("添加比较文件…")
        page.comparison_load_button.setToolTip("将文件加入当前标签页，作为叠加、差值或相对差的比较数据。")
        return page

    def _connect_page(self, page):
        page.load_finished.connect(lambda current=page: self._sync_path(current))
        page.result_selector.currentIndexChanged.connect(lambda _index, current=page: self._sync_path(current))

    def _activate_page(self, page):
        pass

    def _is_empty(self, page):
        return super()._is_empty(page) and not page.result_files

    def _sync_path(self, page):
        self._set_path(page, page.result_selector.currentData() or "")

    def open_paths(self, paths):
        if self._closing:
            return False
        self.activate()
        accepted = False
        for path in paths:
            page = self._available_page()
            self._set_path(page, path)
            page.load_paths_async([self._paths[page]])
            accepted = True
        return accepted

    def load_paths_async(self, paths):
        return self.open_paths(paths)

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "在独立绘图标签页中打开", file_dialog_directory(self),
                                                "Data files (*.csv *.tfs *.h5 *.hdf5);;All files (*)")
        self.open_paths(paths)


class AnalysisWorkspace(_FileWorkspace):
    """Keep every file's analysis controls, results and Matplotlib views separate."""

    @property
    def current_kind(self):
        page = self.current_page
        if page is None:
            return "spectrum"
        if page._activated:
            return "dynamic_aperture" if page.stack.currentWidget() is page.dynamic_aperture else page.task.currentData()
        return page.property("analysisKind") or "spectrum"

    def _make_page(self, kind):
        from PASS.gui.analysis import AnalysisPage

        kind = kind or self.current_kind
        previous = self.current_page
        turn = -1
        if kind == "dynamic_aperture" and previous is not None and previous.dynamic_aperture is not None and previous.dynamic_aperture._activated:
            turn = previous.dynamic_aperture.requested_turn.value()
        page = AnalysisPage()
        page.setProperty("analysisKind", kind)
        page.setProperty("requestedTurn", turn)
        return page

    def _connect_page(self, page):
        page.load_finished.connect(lambda current=page: self._sync_path(current))

    def _activate_page(self, page):
        page.activate()
        if page not in self._wired_sources:
            self._wired_sources.add(page)
            for source in page.sources.values():
                source.path.textChanged.connect(lambda _text, current=page: self._sync_path(current))
            page.navigation.currentRowChanged.connect(lambda _index, current=page: self._mode_changed(current))
            page.dynamic_aperture.analysis_finished.connect(lambda _value, current=page: self._sync_path(current))
            page.select_analysis(page.property("analysisKind"))
            if page.dynamic_aperture._activated:
                page.dynamic_aperture.requested_turn.setValue(page.property("requestedTurn"))
            self._mode_changed(page)

    def _mode_changed(self, page):
        if page.dynamic_aperture._activated and not page.dynamic_aperture.property("fileTabConnected"):
            page.dynamic_aperture.setProperty("fileTabConnected", True)
            page.dynamic_aperture.source.textChanged.connect(lambda _text, current=page: self._sync_path(current))
        self._sync_path(page)

    def _sync_path(self, page):
        if not page._activated:
            return
        if page.stack.currentWidget() is page.dynamic_aperture:
            path = page.dynamic_aperture.source.text()
        else:
            path = page.source.path.text()
        if path.strip():
            self._set_path(page, path)

    def _is_empty(self, page):
        if not super()._is_empty(page):
            return False
        if not page._activated:
            return True
        return not any(source.path.text().strip()
                       for source in page.sources.values()) and not (page.dynamic_aperture._activated and page.dynamic_aperture.source.text().strip())

    def _apply_theme(self, page):
        if page.theme == self.theme:
            return
        # Recolour existing artists without rebuilding axes or clearing zoom history.
        page.theme = self.theme
        if page.dynamic_aperture is not None:
            page.dynamic_aperture.set_theme(self.theme)
        if page._activated:
            from PASS.gui.analysis_data import style_figure

            page.navigation.set_theme(self.theme)
            style_figure(page.view.figure, self.theme)
            page.view.canvas.draw_idle()

    def select_analysis(self, kind):
        if kind not in ("spectrum", "particle_spectrum", "fma", "dynamic_aperture"):
            raise ValueError(f"Unknown analysis entry: {kind}")
        if self._closing:
            return False
        self.activate()
        return self.current_page.select_analysis(kind)

    def open_paths(self, paths, kind=None):
        if self._closing:
            return False
        self.activate()
        kind = kind or self.current_kind
        if kind not in ("spectrum", "particle_spectrum", "fma", "dynamic_aperture"):
            raise ValueError(f"Unknown analysis entry: {kind}")
        requested_turn = -1
        current = self.current_page
        if kind == "dynamic_aperture" and current.dynamic_aperture._activated:
            requested_turn = current.dynamic_aperture.requested_turn.value()
        accepted = False
        for path in paths:
            page = self._available_page(kind)
            self._activate_page(page)
            page.select_analysis(kind)
            if kind == "dynamic_aperture":
                page.dynamic_aperture.requested_turn.setValue(requested_turn)
            self._set_path(page, path)
            accepted = page.open_path(self._paths[page]) or accepted
        return accepted

    def open_path(self, path):
        return self.open_paths([path])

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "在独立分析标签页中打开", file_dialog_directory(self),
                                                "Analysis data (*.h5 *.hdf5 *.tfs *.csv *.tsv *.txt *.dat *.npy *.npz);;All files (*)")
        self.open_paths(paths)
