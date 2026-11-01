"""Batch file-opening routes for shell drops, including child widgets and tabs."""
from pathlib import Path
import zipfile

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget


def identify_file(path):
    path = Path(path)
    if path.is_dir():
        from PASS.gui.project import Project
        path = Project.resolve_path(path)
    if not path.is_file():
        raise ValueError("请选择本地文件或含 .passproj 文件的项目文件夹。")
    with path.open("rb") as stream:
        head = stream.read(512)
        if head.startswith(b"SDDS"):
            return "sdds"
        offset = 0
        while offset < path.stat().st_size:
            stream.seek(offset)
            if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                return "hdf5"
            offset = 512 if offset == 0 else offset * 2
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz", ".tsv", ".txt", ".dat"}:
        return "analysis_data"
    if suffix == ".passproj" or head.startswith(b"PK\x03\x04"):
        from PASS.gui.project import FORMAT_VERSION, JSON_LIMIT, read_json

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("manifest.json")
                if info.file_size > JSON_LIMIT:
                    raise ValueError("项目清单过大。")
                manifest = read_json(archive.read(info))
            expected_version = 2
        else:
            if path.stat().st_size > JSON_LIMIT:
                raise ValueError("项目清单过大。")
            manifest = read_json(path.read_bytes())
            expected_version = FORMAT_VERSION
        if manifest.get("format") != "pass-project" or type(
                manifest.get("format_version")) is not int or manifest["format_version"] != expected_version:
            raise ValueError("不支持的 PASS 项目格式或版本。")
        return "project"
    if suffix == ".json":
        from PASS.gui.project import JSON_LIMIT, read_json
        if path.stat().st_size > JSON_LIMIT:
            raise ValueError("输入 JSON 超过大小限制。")
        data = read_json(path.read_bytes())
        if not any(str(k).casefold() == "sequence" and isinstance(v, dict) for k, v in data.items()):
            raise ValueError("JSON 不是可识别的 PASS 输入：需要 Sequence 对象。")
        return "json"
    if suffix in {".csv", ".tfs"}:
        return suffix[1:]
    raise ValueError("无法识别文件。支持 PASS 项目、输入 JSON、SDDS、HDF5、TFS、CSV、TSV、TXT、DAT、NPY 和 NPZ。")


class FileDropRouter(QObject):

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner

    def eventFilter(self, watched, event):
        if event.type() not in {QEvent.DragEnter, QEvent.DragMove, QEvent.Drop}:
            return False
        if not isinstance(watched, QWidget) or not (watched is self.owner or self.owner.isAncestorOf(watched)):
            return False
        # Plain-text drags and internal table reordering retain their normal behavior.
        mime = event.mimeData()
        if not mime.hasUrls():
            return False
        urls = mime.urls()
        if not urls or any(not url.isLocalFile() for url in urls):
            event.ignore()
            return True
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if event.type() == QEvent.Drop:
            paths = [url.toLocalFile() for url in urls]
            # Drops can be delivered to the window instead of the child under the pointer.
            position = self.owner.mapFromGlobal(watched.mapToGlobal(event.position().toPoint()))
            target = self.owner.childAt(position) or watched
            converter = self.owner.tools.conversion
            into_converter = converter is not None and (target is converter or converter.isAncestorOf(target))
            analysis = self.owner.analysis
            into_analysis = target is analysis or analysis.isAncestorOf(target)
            plot = getattr(self.owner, "plot", None)
            into_plot = plot is not None and (target is plot or plot.isAncestorOf(target))
            # Window margins, navigation and tab headers follow the active workspace.
            stack = getattr(self.owner, "stack", None)
            if not (into_converter or into_analysis or into_plot) and stack is not None:
                into_analysis = stack.currentWidget() is analysis
                into_plot = plot is not None and stack.currentWidget() is plot
            options = dict(into_converter=into_converter, into_analysis=into_analysis, into_plot=into_plot)
            if len(paths) == 1:
                QTimer.singleShot(0, lambda: self.open_path(paths[0], **options))
            else:
                QTimer.singleShot(0, lambda: self.open_paths(paths, **options))
        return True

    def open_path(self, path, *, into_converter=False, into_analysis=False, into_plot=False):
        """Preserve the single-file entry point used by CLI and existing callers."""
        return self.open_paths([path], into_converter=into_converter, into_analysis=into_analysis, into_plot=into_plot)

    def open_paths(self, paths, *, into_converter=False, into_analysis=False, into_plot=False):
        """Identify the whole batch before opening one independent tab per data file."""
        paths = [str(path) for path in paths]
        if not paths:
            return False
        identified, errors = [], []
        for path in paths:
            try:
                identified.append((path, identify_file(path)))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                errors.append(f"{path}: {exc}")
        configuration = any(kind in {"project", "json"} for _, kind in identified)
        configuration |= any(Path(path).suffix.lower() in {".json", ".passproj"} or Path(path).is_dir() for path in paths)
        if len(paths) > 1 and configuration:
            self._report_errors(["项目或输入 JSON 每次只能打开一个，不能与数据文件一起拖入；本批次没有打开任何文件。", *errors])
            return False
        if not identified:
            self._report_errors(errors)
            return False
        try:
            path, kind = identified[0]
            if kind == "project":
                result = self.owner.open_project_path(path)
            elif kind == "json":
                result = self.owner.open_json_path(path)
            else:
                destination = "converter" if into_converter else "analysis" if into_analysis else "plot" if into_plot else None
                analysis_kind = None
                if destination is None:
                    if all(item_kind == "analysis_data" for _, item_kind in identified):
                        destination = "analysis"
                    elif all(item_kind == "sdds" for _, item_kind in identified) or (len(paths) == 1 and kind == "hdf5"):
                        destination = "converter"
                    else:
                        names = "\n".join(Path(item_path).name for item_path, _ in identified)
                        choices = ["绘图", "统计数据频谱", "PM 粒子频谱", "频率图分析 (FMA)", "动力学孔径", "预览与转换"]
                        action, ok = QInputDialog.getItem(self.owner, "选择文件用途", names, choices, 0, False)
                        if not ok:
                            self._report_errors(errors)
                            return False
                        analysis_kinds = {"统计数据频谱": "spectrum", "PM 粒子频谱": "particle_spectrum", "频率图分析 (FMA)": "fma", "动力学孔径": "dynamic_aperture"}
                        analysis_kind = analysis_kinds.get(action)
                        destination = "analysis" if analysis_kind is not None else "plot" if action == "绘图" else "converter"
                if destination == "converter":
                    if len(paths) > 1:
                        errors.append("预览与转换暂不支持同时打开多个文件，请每次拖入一个；本批次没有打开任何文件。")
                        result = False
                    else:
                        self.owner._show_page(4)
                        result = self.owner.tools.open_conversion(path)
                else:
                    supported = {"csv", "tfs", "hdf5"}
                    if destination == "analysis":
                        supported.add("analysis_data")
                    selected = []
                    for item_path, item_kind in identified:
                        if item_kind in supported:
                            selected.append(item_path)
                        else:
                            label = "绘图" if destination == "plot" else "数据分析"
                            errors.append(f"{item_path}: {label}不支持此文件类型，请使用其他数据工具。")
                    result = self._open_data(selected, destination, analysis_kind) if selected else False
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
            result = False
        self._report_errors(errors)
        return result is not False

    def _open_data(self, paths, destination, analysis_kind):
        if destination == "plot":
            self.owner._show_page(2)
            workspace = self.owner.plot
        else:
            self.owner._show_page(3)
            workspace = self.owner.analysis
        if hasattr(workspace, "open_paths"):
            if destination == "analysis" and analysis_kind is not None:
                return workspace.open_paths(paths, kind=analysis_kind)
            return workspace.open_paths(paths)
        # Legacy single pages are still useful when embedded outside MainWindow.
        if len(paths) > 1:
            raise ValueError("当前页面不支持多个文件，请使用带文件标签页的绘图或数据分析窗口。")
        if destination == "plot":
            return workspace.load_paths_async(paths)
        if analysis_kind is not None:
            workspace.select_analysis(analysis_kind)
        return workspace.open_path(paths[0])

    def _report_errors(self, errors):
        if errors:
            QMessageBox.warning(self.owner, "部分文件未打开", "\n\n".join(errors))
