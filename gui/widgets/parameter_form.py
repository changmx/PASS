from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QGridLayout,
    QSizePolicy,
    QVBoxLayout,
    QScrollArea,
)

from PySide6.QtCore import Qt

from gui.widgets.parameter_widget_factory import (ParameterWidgetFactory)


class ParameterForm(QWidget):

    def __init__(self, model):

        super().__init__()

        self.model = model

        self.widgets = {}

        self.setup_ui()

    # =========================================================
    # ui
    # =========================================================

    def setup_ui(self):

        # -----------------------------------------------------
        # root layout
        # -----------------------------------------------------

        root_layout = QVBoxLayout()

        self.setLayout(root_layout)

        # -----------------------------------------------------
        # scroll area
        # -----------------------------------------------------

        self.scroll = QScrollArea()

        self.scroll.setWidgetResizable(True)

        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root_layout.addWidget(self.scroll)

        # -----------------------------------------------------
        # content widget
        # -----------------------------------------------------

        self.content = QWidget()

        self.scroll.setWidget(self.content)

        # -----------------------------------------------------
        # grid layout
        # -----------------------------------------------------

        self.layout = QGridLayout()

        self.content.setLayout(self.layout)

        self.layout.setContentsMargins(20, 20, 20, 20)

        self.layout.setHorizontalSpacing(20)

        self.layout.setVerticalSpacing(12)

        self.layout.setColumnStretch(0, 0)

        self.layout.setColumnStretch(1, 1)

        # -----------------------------------------------------
        # rows
        # -----------------------------------------------------

        row = 0

        for name, parameter in vars(self.model).items():

            # label

            label = QLabel(parameter.description)

            label.setMinimumWidth(180)

            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # widget

            widget = (ParameterWidgetFactory.create_widget(parameter))

            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            # add

            self.layout.addWidget(label, row, 0)

            self.layout.addWidget(widget, row, 1)

            self.widgets[name] = widget

            row += 1

        # bottom stretch

        self.layout.setRowStretch(row, 1)
