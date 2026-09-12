"""Application-wide editor themes, native window frames and vector indicators."""
import ctypes
from pathlib import Path
import re
import sys

from PySide6.QtCore import QByteArray, QEvent, QObject, QTemporaryDir, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap, QSyntaxHighlighter, QTextCharFormat, QWheelEvent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QComboBox, QWidget

THEMES = {
    # Core colors from Binaryify/OneDark-Pro, themes/OneDark-Pro.json.
    "dark": dict(bg="#282c34", panel="#21252b", input="#1d1f23", line="#3e4452",
                 text="#abb2bf", muted="#9da5b4", accent="#61afef", active="#2c313a",
                 section="#282c34", guide="#4b5363", error="#e06c75", warning="#e5c07b",
                 string="#98c379", number="#d19a66", keyword="#c678dd"),
    "light": dict(bg="#fafafa", panel="#f0f0f0", input="#ffffff", line="#d5d7dc",
                  text="#383a42", muted="#666a73", accent="#2965cf", active="#e4eaf5",
                  section="#e8e9ec", guide="#bec3cd", error="#c63440", warning="#8a5b0a",
                  string="#397b31", number="#986801", keyword="#a626a4"),
}

UI_FAMILIES = '"Segoe UI Variable Text", "Microsoft YaHei UI", "Segoe UI"'
CODE_FAMILIES = '"JetBrains Mono", "Consolas", "Cascadia Mono", "Microsoft YaHei UI"'
_indicator_dir = None


def code_font() -> QFont:
    """Prefer the upstream theme's font without requiring a font installation."""
    available = set(QFontDatabase.families())
    family = next((name for name in ("JetBrains Mono", "Consolas", "Cascadia Mono") if name in available), "monospace")
    font = QFont(family)
    font.setPixelSize(12)
    font.setStyleHint(QFont.Monospace)
    return font


def _indicator_url(glyph: str, color: str) -> str:
    """QSS needs a file URL; retain a private temporary SVG cache for its lifetime."""
    global _indicator_dir
    if _indicator_dir is None:
        _indicator_dir = QTemporaryDir()
        if not _indicator_dir.isValid():
            raise OSError("Unable to create the GUI indicator cache")
    path = Path(_indicator_dir.path()) / f"{glyph}-{color.lstrip('#')}.svg"
    if not path.exists():
        drawing = {"down": "M4 6l4 4 4-4", "check": "M3 8l3 3 7-7", "partial": "M3 8h10"}[glyph]
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
                        f'<path d="{drawing}" fill="none" stroke="{color}" stroke-width="1.8" '
                        'stroke-linecap="round" stroke-linejoin="round"/></svg>', encoding="utf-8")
    return f'url("{path.as_posix()}")'


class JsonHighlighter(QSyntaxHighlighter):
    """Highlight JSON tokens without changing input or covering text inside strings."""
    tokens = re.compile(r'"(?:[^"\\]|\\.)*"|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

    def __init__(self, document):
        super().__init__(document)
        self.theme = "dark"

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        colors = THEMES[self.theme]
        for match in self.tokens.finditer(text):
            value = match.group()
            if value.startswith('"'):
                role = "accent" if text[match.end():].lstrip().startswith(":") else "string"
            else:
                role = "keyword" if value in ("true", "false", "null") else "number"
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(colors[role]))
            # Qt offsets count UTF-16 code units, including astral Unicode in keys.
            offset = len(text[:match.start()].encode("utf-16-le")) // 2
            length = len(value.encode("utf-16-le")) // 2
            self.setFormat(offset, length, fmt)


def _native_title_bar(window: QWidget, theme: str) -> None:
    """Use DWM colors on Windows 11 and the dark-frame flag on Windows 10."""
    if sys.platform != "win32" or QApplication.platformName() != "windows":
        return
    try:
        hwnd = ctypes.c_void_p(int(window.winId()))
        setter = ctypes.windll.dwmapi.DwmSetWindowAttribute
        dark = ctypes.c_int(theme == "dark")
        for attribute in (20, 19):
            if setter(hwnd, attribute, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
                break
        # Unsupported attributes return HRESULT on older systems; Qt keeps its
        # native frame and normal move/resize/snap behavior on every platform.
        for attribute, role in ((35, "bg"), (36, "text")):
            color = QColor(THEMES[theme][role])
            value = ctypes.c_uint32(color.red() | color.green() << 8 | color.blue() << 16)
            setter(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    except (AttributeError, OSError):
        pass


class _WindowThemeController(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.theme = "dark"
        self._applying = False
        app.installEventFilter(self)

    def apply_window(self, window):
        if (self._applying or window.windowHandle() is None
                or window.windowType() not in (Qt.Window, Qt.Dialog)):
            return
        self._applying = True
        try:
            _native_title_bar(window, self.theme)
        finally:
            self._applying = False

    def update_window(self, window):
        self.apply_window(window)
        # Qt's Windows plugin can reset DWM attributes after WinIdChange/Show
        # processing. Reapply once after it finishes; the context cancels this
        # callback automatically if the dialog is destroyed before then.
        QTimer.singleShot(0, window, lambda: self.apply_window(window))

    def eventFilter(self, watched, event):
        # Apply to every selector, including lazy pages and Qt dialog controls.
        # Popup views still receive wheel events for browsing the open list.
        if event.type() == QEvent.Wheel and isinstance(watched, QComboBox):
            # An application filter stops Qt's normal parent propagation. Route
            # the wheel to the surrounding form so scrolling over a field works.
            parent = watched.parentWidget()
            while parent is not None:
                if isinstance(parent, QAbstractScrollArea):
                    viewport = parent.viewport()
                    forwarded = QWheelEvent(viewport.mapFromGlobal(event.globalPosition()),
                        event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                        event.buttons(), event.modifiers(), event.phase(), event.inverted(),
                        event.source(), event.pointingDevice())
                    QApplication.sendEvent(viewport, forwarded)
                    break
                parent = parent.parentWidget()
            event.ignore()
            return True
        if (event.type() in (QEvent.Show, QEvent.WinIdChange, QEvent.PaletteChange, QEvent.ThemeChange)
                and isinstance(watched, QWidget) and watched.isWindow()
                and watched.testAttribute(Qt.WA_WState_Created)
                and watched.windowHandle() is not None):
            self.update_window(watched)
        return False


def apply_application_theme(theme: str) -> None:
    """Cover existing and future dialogs, including unparented message boxes."""
    app = QApplication.instance()
    controller = getattr(app, "_pass_window_theme", None)
    if controller is None:
        controller = _WindowThemeController(app)
        app._pass_window_theme = controller
    controller.theme = theme
    # Native shell file pickers bypass Qt styling; the Qt picker uses this theme
    # while keeping the system's native outer frame.
    app.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    palette, rules = theme_palette(theme), stylesheet(theme)
    if app.palette() != palette:
        app.setPalette(palette)
    if app.styleSheet() != rules:
        app.setStyleSheet(rules)
    for window in app.topLevelWidgets():
        if window.windowHandle() is not None:
            controller.update_window(window)

PATHS = {
    "file": '<path d="M4 1.5h5l3 3v10H4zM9 1.5v3h3"/>',
    "folder": '<path d="M1.5 4V2.5h5l2 2H14v2M1.5 6.5h13l-2 7h-11z"/>',
    "save": '<path d="M2 2h10l2 2v10H2zM5 2v4h6V2M5 14V9h6v5"/>',
    "copy": '<path d="M5 5h9v9H5zM2 11V2h9"/>',
    "check": '<path d="M2.5 8l3.5 3.5L13.5 4"/>',
    "settings": '<path d="M2 4h12M2 8h12M2 12h12"/><circle cx="6" cy="4" r="1.5"/><circle cx="10" cy="8" r="1.5"/><circle cx="5" cy="12" r="1.5"/>',
    "beam": '<circle cx="8" cy="8" r="5.5"/><circle cx="8" cy="8" r="1.5"/><path d="M8 .5v2M8 13.5v2M.5 8h2M13.5 8h2"/>',
    "calculator": '<rect x="2.5" y="1" width="11" height="14" rx="1.5"/><path d="M5 4h6M5 7h.01M8 7h.01M11 7h.01M5 10h.01M8 10h.01M11 10v2.5M5 12.5h.01M8 12.5h.01"/>',
    "resonance": '<path d="M2 1.5V14h12.5M4 4l8 8M4 12l8-8M4 8h8"/><circle cx="11.5" cy="2" r="1.2"/>',
    "rf_bucket": '<path d="M1 8h14M3 8c2-8 8-8 10 0-2 8-8 8-10 0zM8 1v14"/>',
    "phase_ellipse": '<path d="M1 8h14M8 1v14"/><ellipse cx="8" cy="8" rx="3" ry="6.5" transform="rotate(35 8 8)"/>',
    "magnet": '<path d="M2 2h4v7a2 2 0 0 0 4 0V2h4v7a6 6 0 0 1-12 0zM2 5h4M10 5h4"/>',
    "exciter": '<path d="M1 8h1c2-8 4 8 6 0s4 8 6 0h1M1 14h14"/>',
    "list": '<path d="M6 3h8M6 8h8M6 13h8M2 3h1M2 8h1M2 13h1"/>',
    "tools": '<path d="M10 1.5a4 4 0 0 0-4.5 5L1.5 11a2 2 0 0 0 3 3L9 10a4 4 0 0 0 5.5-4.5L12 8 9 5z"/>',
    "optics": '<path d="M1 8h14M4 2v12M12 2v12M4 2l8 12M4 14L12 2"/>',
    "box": '<path d="M2 4l6-3 6 3v8l-6 3-6-3zM2 4l6 3 6-3M8 7v8"/>',
    "chart": '<path d="M2 1v13h13M4 10l3-4 3 2 4-5"/>',
    "layers": '<path d="M1.5 5l6.5-3 6.5 3L8 8zM1.5 8.5l6.5 3 6.5-3M1.5 12l6.5 3 6.5-3"/>',
    "plus": '<path d="M8 2v12M2 8h12"/>',
    "trash": '<path d="M2 4h12M6 1.5h4V4M4 4l.5 10h7L12 4M6.5 6.5v5M9.5 6.5v5"/>',
    "moon": '<path d="M12.7 11.5A6 6 0 0 1 4.5 3.3 6 6 0 1 0 12.7 11.5z"/>',
    "sun": '<circle cx="8" cy="8" r="3"/><path d="M8 0v2M8 14v2M0 8h2M14 8h2M2.3 2.3l1.4 1.4M12.3 12.3l1.4 1.4M2.3 13.7l1.4-1.4M12.3 3.7l1.4-1.4"/>',
    "archive": '<rect x="2" y="2" width="12" height="12" rx="1"/><path d="M7 2v3h2v3H7v3h2v3"/>',
}


def icon(name: str, color: str) -> QIcon:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><g fill="none" stroke="{color}" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round">{PATHS[name]}</g></svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def theme_palette(theme: str) -> QPalette:
    c = THEMES[theme]
    palette = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "input"), (QPalette.AlternateBase, "panel"),
                      (QPalette.WindowText, "text"), (QPalette.Text, "text"), (QPalette.Button, "panel"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "active"),
                      (QPalette.HighlightedText, "text"), (QPalette.ToolTipBase, "input"),
                      (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "muted"),
                      (QPalette.Mid, "line"), (QPalette.Link, "accent")):
        palette.setColor(role, QColor(c[key]))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, QColor(c["muted"]))
    return palette


def stylesheet(theme: str) -> str:
    c = dict(THEMES[theme], ui_fonts=UI_FAMILIES, code_fonts=CODE_FAMILIES)
    c.update(arrow=_indicator_url("down", c["text"]), arrow_disabled=_indicator_url("down", c["muted"]),
             tick=_indicator_url("check", c["text"]), tick_disabled=_indicator_url("check", c["muted"]),
             partial=_indicator_url("partial", c["text"]))
    return """
    QWidget { background: %(bg)s; color: %(text)s; font-family: %(ui_fonts)s; font-size: 12px; }
    QPlainTextEdit#codeEditor { font-family: %(code_fonts)s; font-size: 12px; }
    QLabel { background: transparent; }
    QLabel#brand { color: %(accent)s; font-size: 20px; font-weight: 500; }
    QLabel#muted, QLabel#runStat { color: %(muted)s; font-size: 12px; }
    QFrame#editorPanel, QFrame#propertyPanel, QFrame#libraryPanel { background: %(panel)s; border: 1px solid %(line)s; border-radius: 6px; }
    #propertyPanel QWidget { background-color: %(panel)s; }
    #libraryPanel QWidget { background-color: transparent; }
    #libraryPanel QWidget#librarySectionBody { border: 0; border-left: 1px solid %(guide)s; background: %(panel)s; }
    #propertyPanel QLabel { font-size: 12px; }
    QLabel#formTitle { font-size: 13px; font-weight: 500; }
    #propertyPanel QLineEdit#commandBadge { color: %(accent)s; background: transparent; border: 0; padding: 0 3px; font-size: 11px; }
    #propertyPanel QGroupBox#propertySection { background: %(panel)s; border: 1px solid %(line)s; border-left: 2px solid %(guide)s; border-radius: 4px; margin-top: 12px; padding: 5px 0 0 0; }
    #propertyPanel QGroupBox#propertySection::title { color: %(accent)s; background: %(panel)s; subcontrol-origin: margin; left: 8px; padding: 0 4px; font-weight: 500; }
    QPushButton, QToolButton { background: %(panel)s; border: 1px solid %(line)s; border-radius: 4px; padding: 5px 9px; }
    QPushButton:hover, QToolButton:hover { border-color: %(accent)s; background: %(active)s; }
    QPushButton:disabled, QToolButton:disabled { color: %(muted)s; }
    QPushButton#primary { background: %(active)s; border-color: %(accent)s; color: %(accent)s; }
    QPushButton#nav { border: 0; border-radius: 0; padding: 7px 15px; }
    QPushButton#nav:checked { border-bottom: 2px solid %(accent)s; color: %(accent)s; }
    #libraryPanel QPushButton { text-align: left; border: 0; border-radius: 3px; padding: 5px 7px; }
    #libraryPanel QPushButton#librarySectionHeader { background: %(section)s; font-weight: 500; padding: 6px 7px; }
    #libraryPanel QPushButton#librarySectionHeader[depth="1"] { background: transparent; font-weight: 400; }
    #libraryPanel QPushButton#librarySectionHeader:checked { color: %(accent)s; }
    #libraryPanel QPushButton:hover, #libraryPanel QToolButton:hover { background: %(active)s; }
    #libraryPanel QToolButton#librarySectionToggle { border: 0; padding: 0; background: %(section)s; }
    #libraryPanel QToolButton#librarySectionToggle[depth="1"] { background: transparent; }
    QScrollArea, QScrollArea#libraryScroll { border: 0; background: transparent; }
    QLineEdit, QComboBox, QPlainTextEdit, QTreeWidget, QListWidget, QTableWidget { background: %(input)s; border: 1px solid %(line)s; border-radius: 3px; selection-background-color: %(active)s; selection-color: %(text)s; }
    QLineEdit, QComboBox { padding: 3px 6px; min-height: 18px; }
    #propertyPanel QLineEdit, #propertyPanel QComboBox, #propertyPanel QPlainTextEdit { background: %(input)s; font-size: 12px; }
    #propertyPanel QLineEdit, #propertyPanel QComboBox { padding: 2px 5px; min-height: 18px; }
    #structuredField QTableWidget { font-size: 12px; background: %(input)s; }
    #structuredField QHeaderView::section { padding: 4px; font-size: 11px; }
    #structuredField QPushButton { padding: 3px 5px; font-size: 12px; }
    QSpinBox, QDoubleSpinBox { background: %(input)s; color: %(text)s; border: 1px solid %(line)s; border-radius: 3px; padding: 2px 4px; min-height: 18px; font-size: 12px; }
    QSpinBox:disabled, QDoubleSpinBox:disabled { color: %(muted)s; }
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: %(accent)s; }
    QLineEdit#readonlyField, QLineEdit:disabled, QComboBox:disabled { color: %(muted)s; }
    QComboBox, #propertyPanel QComboBox { padding-right: 29px; }
    QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; border: 0; border-left: 1px solid %(line)s; width: 23px; }
    QComboBox::down-arrow { image: %(arrow)s; width: 14px; height: 14px; }
    QComboBox::down-arrow:disabled { image: %(arrow_disabled)s; }
    QComboBox QAbstractItemView { background: %(input)s; color: %(text)s; border: 1px solid %(line)s; selection-background-color: %(active)s; selection-color: %(text)s; outline: 0; }
    QCheckBox { spacing: 6px; background: transparent; }
    QCheckBox::indicator, QAbstractItemView::indicator, QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid %(muted)s; border-radius: 2px; background: %(input)s; }
    QCheckBox::indicator:checked, QAbstractItemView::indicator:checked, QGroupBox::indicator:checked { image: %(tick)s; }
    QCheckBox::indicator:indeterminate, QAbstractItemView::indicator:indeterminate { image: %(partial)s; }
    QCheckBox::indicator:checked:disabled, QAbstractItemView::indicator:checked:disabled, QGroupBox::indicator:checked:disabled { image: %(tick_disabled)s; border-color: %(line)s; }
    QCheckBox::indicator:unchecked:disabled { border-color: %(line)s; }
    QGroupBox { border: 1px solid %(line)s; border-radius: 3px; margin-top: 11px; padding: 5px; }
    QGroupBox::title { color: %(muted)s; subcontrol-origin: margin; left: 6px; padding: 0 3px; font-size: 12px; }
    QTableWidget { gridline-color: %(line)s; alternate-background-color: %(panel)s; }
    QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected { background: %(active)s; color: %(text)s; }
    QHeaderView::section { background: %(input)s; color: %(muted)s; border: 0; border-right: 1px solid %(line)s; border-bottom: 1px solid %(line)s; padding: 5px 8px; font-size: 12px; }
    QTabWidget::pane { border: 0; }
    QTabBar::tab { background: %(panel)s; color: %(muted)s; padding: 7px 12px; border-bottom: 2px solid transparent; }
    QTabBar::tab:selected { color: %(accent)s; border-bottom-color: %(accent)s; }
    QScrollBar:vertical { background: %(panel)s; width: 8px; margin: 0; }
    QScrollBar::handle:vertical { background: %(line)s; min-height: 24px; border-radius: 3px; }
    QScrollBar:horizontal { background: %(panel)s; height: 8px; margin: 0; }
    QScrollBar::handle:horizontal { background: %(line)s; min-width: 24px; border-radius: 3px; }
    QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
    QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
    QSplitter::handle { background: %(bg)s; width: 8px; }
    QStatusBar { background: %(panel)s; color: %(muted)s; font-size: 11px; }
    QMenu { background: %(input)s; border: 1px solid %(line)s; padding: 6px; }
    QMenu::item { padding: 6px 24px 6px 24px; border-radius: 3px; }
    QMenu::indicator { width: 14px; height: 14px; }
    QMenu::indicator:checked { image: %(tick)s; }
    QMenu::indicator:checked:disabled { image: %(tick_disabled)s; }
    QMenu::item:selected { background: %(active)s; }
    QMenu::item:disabled { color: %(muted)s; }
    QMenu::separator { height: 1px; background: %(line)s; margin: 5px 2px; }
    QToolTip { background: %(input)s; color: %(text)s; border: 1px solid %(line)s; padding: 4px; }
    QLabel#syncStatus { color: %(muted)s; font-size: 11px; }
    QLabel#syncStatus[state="warning"] { color: %(accent)s; }
    QPushButton#validationStatus { border: 0; color: %(muted)s; font-size: 12px; }
    QPushButton#validationStatus[state="error"] { color: %(error)s; }
    QPushButton#validationStatus[state="warning"] { color: %(warning)s; }
    """ % c
