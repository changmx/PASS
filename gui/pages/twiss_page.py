from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QCheckBox,
    QSpinBox,
)


class TwissPage(QWidget):

    def __init__(self, model):

        super().__init__()

        self.model = model

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()

        self.setLayout(layout)

        # ------------------------------------------------
        # twiss file
        # ------------------------------------------------

        self.twiss_label = QLabel("No Twiss File")

        layout.addWidget(self.twiss_label)

        self.twiss_button = QPushButton("Select Twiss File")

        self.twiss_button.clicked.connect(self.select_twiss)

        layout.addWidget(self.twiss_button)

        # ------------------------------------------------
        # error file
        # ------------------------------------------------

        self.error_label = QLabel("No Error File")

        layout.addWidget(self.error_label)

        self.error_button = QPushButton("Select Error File")

        self.error_button.clicked.connect(self.select_error)

        layout.addWidget(self.error_button)

        # ------------------------------------------------
        # interpolate
        # ------------------------------------------------

        self.interp_box = QCheckBox("Enable Interpolation")

        self.interp_box.setChecked(self.model.interpolate)

        self.interp_box.stateChanged.connect(self.update_interp)

        layout.addWidget(self.interp_box)

        # ------------------------------------------------
        # num slices
        # ------------------------------------------------

        self.slice_spin = QSpinBox()

        self.slice_spin.setMaximum(1000000)

        self.slice_spin.setValue(self.model.num_interp_slice)

        self.slice_spin.valueChanged.connect(self.update_slice)

        layout.addWidget(self.slice_spin)

        layout.addStretch()

    # ------------------------------------------------
    # select file
    # ------------------------------------------------

    def select_twiss(self):

        path, _ = QFileDialog.getOpenFileName(self, "Select Twiss File")

        if path:

            self.model.twiss_file = path

            self.twiss_label.setText(path)

    def select_error(self):

        path, _ = QFileDialog.getOpenFileName(self, "Select Error File")

        if path:

            self.model.error_file = path

            self.error_label.setText(path)

    # ------------------------------------------------
    # update
    # ------------------------------------------------

    def update_interp(self, value):

        self.model.interpolate = bool(value)

    def update_slice(self, value):

        self.model.num_interp_slice = value
