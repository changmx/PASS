from PySide6.QtWidgets import (
    QLineEdit,
    QCheckBox,
)

from gui.widgets.no_wheel import (
    NoWheelSpinBox,
    NoWheelDoubleSpinBox,
    NoWheelComboBox,
)

from gui.widgets.unit_spinbox import (UnitDoubleSpinBox)

from gui.widgets.engineering_spinbox import (EngineeringSpinBox)

from gui.widgets.list_line_edit import (ListLineEdit)


class ParameterWidgetFactory:

    @staticmethod
    def finalize_widget(widget):

        widget.setMinimumHeight(32)

        widget.setMaximumHeight(32)

        return widget

    @staticmethod
    def create_widget(parameter):

        value = parameter.value

        # bool

        if isinstance(value, bool):

            widget = QCheckBox()

            widget.setChecked(value)

            widget.stateChanged.connect(lambda v: setattr(parameter, "value", bool(v)))

            return (ParameterWidgetFactory.finalize_widget(widget))

        # choices

        if parameter.choices:

            widget = NoWheelComboBox()

            widget.addItems(parameter.choices)

            widget.setCurrentText(value)

            widget.currentTextChanged.connect(lambda v: setattr(parameter, "value", v))

            return (ParameterWidgetFactory.finalize_widget(widget))

        # unit

        if parameter.unit:

            return UnitDoubleSpinBox(parameter)

        # scientific

        if parameter.scientific:

            return EngineeringSpinBox(parameter)

        # int

        if isinstance(value, int):

            widget = NoWheelSpinBox()

            QT_INT_MAX = 2147483647
            QT_INT_MIN = -2147483648

            # 处理最大值
            max_val = int(parameter.maximum)
            if max_val > QT_INT_MAX:
                max_val = QT_INT_MAX
            widget.setMaximum(max_val)

            # 处理最小值
            min_val = int(parameter.minimum)
            if min_val < QT_INT_MIN:
                min_val = QT_INT_MIN
            widget.setMinimum(min_val)

            widget.setValue(value)

            widget.valueChanged.connect(lambda v: setattr(parameter, "value", v))

            return (ParameterWidgetFactory.finalize_widget(widget))

        # float

        if isinstance(value, float):

            # engineering scale

            if parameter.scientific:

                return EngineeringSpinBox(parameter)

            # normal float

            widget = NoWheelDoubleSpinBox()

            widget.setDecimals(10)

            widget.setMaximum(parameter.maximum)

            widget.setMinimum(parameter.minimum)

            widget.setValue(value)

            widget.valueChanged.connect(lambda v: setattr(parameter, "value", v))

            return (ParameterWidgetFactory.finalize_widget(widget))

        if isinstance(value, list):

            return ListLineEdit(parameter)

        # string

        widget = QLineEdit()

        widget.setText(str(value))

        widget.textChanged.connect(lambda v: setattr(parameter, "value", v))

        return (ParameterWidgetFactory.finalize_widget(widget))
