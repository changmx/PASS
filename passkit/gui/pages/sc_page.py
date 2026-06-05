from PySide6.QtWidgets import (
    QWidget,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QComboBox,
)


class SpaceChargePage(QWidget):

    def __init__(self, model):

        super().__init__()

        self.model = model

        self.setup_ui()

    def setup_ui(self):

        layout = QFormLayout()

        self.setLayout(layout)

        # ------------------------------------------------
        # enable
        # ------------------------------------------------

        self.enable_box = QCheckBox()

        self.enable_box.setChecked(self.model.enable)

        self.enable_box.stateChanged.connect(self.update_enable)

        layout.addRow("Enable", self.enable_box)

        # ------------------------------------------------
        # slices
        # ------------------------------------------------

        self.slice_spin = QSpinBox()

        self.slice_spin.setMaximum(100000)

        self.slice_spin.setValue(self.model.num_slices)

        self.slice_spin.valueChanged.connect(self.update_slices)

        layout.addRow("Slices", self.slice_spin)

        # ------------------------------------------------
        # field solver
        # ------------------------------------------------

        self.solver_combo = QComboBox()

        self.solver_combo.addItems([
            "PIC_FD_CUDSS",
            "PIC_FFT",
            "PIC_OPENMP",
        ])

        self.solver_combo.setCurrentText(self.model.field_solver)

        self.solver_combo.currentTextChanged.connect(self.update_solver)

        layout.addRow("Field Solver", self.solver_combo)

    # ------------------------------------------------
    # update
    # ------------------------------------------------

    def update_enable(self, value):

        self.model.enable = bool(value)

    def update_slices(self, value):

        self.model.num_slices = value

    def update_solver(self, value):

        self.model.field_solver = value
