from PySide6.QtWidgets import (
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
)


class NoWheelSpinBox(QSpinBox):

    def wheelEvent(self, event):

        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):

    def wheelEvent(self, event):

        event.ignore()


class NoWheelComboBox(QComboBox):

    def wheelEvent(self, event):

        event.ignore()
