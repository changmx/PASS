from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
)

from passkit.gui.widgets.no_wheel import (
    NoWheelDoubleSpinBox,
    NoWheelComboBox,
)

from PySide6.QtCore import Signal

UNIT_TABLE = {
    "eV": {
        "eV": 1.0,
        "keV": 1e3,
        "MeV": 1e6,
        "GeV": 1e9,
        "J": 1.602176634e-19,
    },
    "m": {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "A": 1e-10,
        "pm": 1e-12,
        "fm": 1e-15,
    },
    "pi*m*rad": {
        "pi*m*rad": 1.0,
        "pi*mm*mrad": 1e-6,
        "pi*nm*rad": 1e-9,
    },
    "rad": {
        "rad": 1.0,
        "mrad": 1e-3,
        "deg": 180 / 3.141592653589793,
    },
    "V": {
        "V": 1.0,
        "kV": 1e3,
        "MV": 1e6,
    },
}


class UnitDoubleSpinBox(QWidget):

    value_changed_si = Signal(float)

    def __init__(self, parameter):

        super().__init__()

        self.parameter = parameter

        self.unit_table = UNIT_TABLE[parameter.unit]

        self.current_factor = (self.unit_table[parameter.display_unit])

        self.setup_ui()

    def setup_ui(self):

        layout = QHBoxLayout()

        layout.setContentsMargins(0, 0, 0, 0)

        layout.setSpacing(4)

        self.setLayout(layout)

        self.spin = NoWheelDoubleSpinBox()

        self.spin.setDecimals(10)

        self.spin.setMaximum(self.parameter.maximum)

        self.spin.setMinimum(self.parameter.minimum)

        self.spin.setFixedHeight(28)

        self.spin.valueChanged.connect(self.update_parameter)

        layout.addWidget(self.spin)

        self.combo = NoWheelComboBox()

        self.combo.addItems(self.unit_table.keys())

        self.combo.setCurrentText(self.parameter.display_unit)

        self.combo.setFixedHeight(28)

        self.combo.currentTextChanged.connect(self.change_unit)

        layout.addWidget(self.combo)

        self.setFixedHeight(32)

        self.refresh_display()

    def refresh_display(self):

        value = (self.parameter.value / self.current_factor)

        self.spin.setValue(value)

    def update_parameter(self):

        self.parameter.value = (self.spin.value() * self.current_factor)

    def change_unit(self, unit):

        value_si = self.parameter.value

        self.current_factor = (self.unit_table[unit])

        self.parameter.display_unit = unit

        self.refresh_display()
