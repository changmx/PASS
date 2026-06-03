from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

from gui.widgets.no_wheel import (
    NoWheelDoubleSpinBox,
    NoWheelComboBox,
)

from PySide6.QtCore import Signal


class EngineeringSpinBox(QWidget):

    value_changed = Signal(float)

    def __init__(self, parameter):

        super().__init__()

        self.parameter = parameter

        self.setup_ui()

    # =========================================================
    # ui
    # =========================================================

    def setup_ui(self):

        layout = QHBoxLayout()

        layout.setContentsMargins(0, 0, 0, 0)

        layout.setSpacing(4)

        self.setLayout(layout)

        # -----------------------------------------------------
        # mantissa
        # -----------------------------------------------------

        self.spin = NoWheelDoubleSpinBox()

        self.spin.setDecimals(6)

        self.spin.setMaximum(1e9)

        self.spin.setMinimum(-1e9)

        self.spin.setFixedHeight(28)

        self.spin.valueChanged.connect(self.update_parameter)

        layout.addWidget(self.spin)

        # -----------------------------------------------------
        # exponent
        # -----------------------------------------------------

        self.combo = NoWheelComboBox()

        self.exponents = []

        for i in [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]:

            text = f"1e{i}"

            self.exponents.append(text)

        self.combo.addItems(self.exponents)

        self.combo.setFixedHeight(28)

        self.combo.currentTextChanged.connect(self.change_scale)

        layout.addWidget(self.combo)

        self.setFixedHeight(32)

        # -----------------------------------------------------
        # init display
        # -----------------------------------------------------

        self.initialize_display()

    # =========================================================
    # initialize
    # =========================================================

    def initialize_display(self):

        value_si = self.parameter.value

        # -------------------------------------------------
        # use scientific_scale if provided
        # -------------------------------------------------

        if self.parameter.scientific_scale:

            scale_text = (self.parameter.scientific_scale)

        # -------------------------------------------------
        # otherwise auto choose
        # -------------------------------------------------

        else:

            exponent = 0

            value = abs(value_si)

            while value >= 1000 and exponent < 15:

                value /= 10

                exponent += 1

            scale_text = f"1e{exponent}"

        # -------------------------------------------------
        # set combo
        # -------------------------------------------------

        self.combo.setCurrentText(scale_text)

        # -------------------------------------------------
        # set spin
        # -------------------------------------------------

        factor = self.current_factor()

        self.spin.setValue(value_si / factor)

    # =========================================================
    # current factor
    # =========================================================

    def current_factor(self):

        text = self.combo.currentText()

        exponent = int(text[2:])

        return 10**exponent

    # =========================================================
    # update parameter
    # =========================================================

    def update_parameter(self):

        self.parameter.value = (self.spin.value() * self.current_factor())

        self.value_changed.emit(self.parameter.value)

    # =========================================================
    # scale changed
    # =========================================================

    def change_scale(self):

        value_si = self.parameter.value

        factor = self.current_factor()

        self.spin.blockSignals(True)

        self.spin.setValue(value_si / factor)

        self.spin.blockSignals(False)
