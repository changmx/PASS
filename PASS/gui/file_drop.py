"""One file-opening route for shell drops, including child text/table widgets."""
import json
from pathlib import Path
import zipfile

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget


def identify_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("请拖入一个本地文件。")
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
    if suffix == ".passproj" or head.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("manifest.json")
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError("项目清单过大。")
            manifest = json.loads(archive.read(info))
            if manifest.get("format") != "pass-project":
                raise ValueError("不是 PASS 项目文件。")
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
    raise ValueError("无法识别文件。支持 PASS 项目、输入 JSON、SDDS、HDF5、TFS 和 CSV。")


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
        if len(urls) != 1 or not urls[0].isLocalFile():
            event.ignore()
            return True
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if event.type() == QEvent.Drop:
            path = urls[0].toLocalFile()
            converter = self.owner.tools.conversion
            into_converter = converter is not None and (watched is converter or converter.isAncestorOf(watched))
            QTimer.singleShot(0, lambda: self.open_path(path, into_converter=into_converter))
        return True

    def open_path(self, path, *, into_converter=False):
        try:
            kind = identify_file(path)
            if kind == "project":
                self.owner.open_project_path(path)
            elif kind == "json":
                self.owner.open_json_path(path)
            elif kind in {"sdds", "hdf5"} or into_converter:
                self.owner._show_page(3)
                self.owner.tools.open_conversion(path)
            else:
                action, ok = QInputDialog.getItem(self.owner, "选择文件用途", Path(path).name, ["预览与转换", "绘图"], 0, False)
                if not ok:
                    return
                if action == "绘图":
                    self.owner.plot.load_paths([path])
                    self.owner._show_page(2)
                else:
                    self.owner._show_page(3)
                    self.owner.tools.open_conversion(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            QMessageBox.warning(self.owner, "无法打开文件", str(exc))
