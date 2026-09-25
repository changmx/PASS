"""Shared Qt controls without dependencies on pages or parameter schemas."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QComboBox, QPushButton, QWidget


class Choice(QComboBox):

    def wheelEvent(self, event):
        event.ignore()

    def showPopup(self):
        # A narrow form column must not truncate the choices in its popup.
        view = self.view()
        view.ensurePolished()
        width = max((self.fontMetrics().horizontalAdvance(self.itemText(i)) for i in range(self.count())), default=0) + 64
        screen = self.screen().availableGeometry()
        view.setMinimumWidth(min(max(self.width(), width), screen.width() - 24))
        # Styled row padding is not included in Qt's default popup height.
        # With two choices this used to cut the second row in half.
        rows = min(self.count(), self.maxVisibleItems())
        height = sum(max(view.sizeHintForRow(i), self.fontMetrics().height()) for i in range(rows))
        view.setMinimumHeight(min(height + 2 * view.frameWidth(), screen.height() // 2))
        super().showPopup()


def button(text: str, object_name: str = "") -> QPushButton:
    item = QPushButton(text)
    if object_name:
        item.setObjectName(object_name)
    item.setCursor(Qt.PointingHandCursor)
    return item


class PropertyComboBox(Choice):
    """A property selector changed only through its drop-down list."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class BusyProgressBar(QWidget):
    """Small indeterminate progress bar rendered without native Qt styling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("busyProgress")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._offset = -0.25
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setFixedHeight(4)
        self.setMinimumWidth(80)

    def _advance(self) -> None:
        self._offset += 0.035
        if self._offset > 1.0:
            self._offset = -0.25
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start(30)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.palette().base())
        painter.drawRect(self.rect())
        width = max(48, int(self.width() * 0.22))
        x = int((self.width() + width) * self._offset - width)
        painter.fillRect(x, 0, width, self.height(), self.palette().link())
