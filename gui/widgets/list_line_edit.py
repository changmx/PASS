import ast

from PySide6.QtWidgets import (
    QLineEdit
)


class ListLineEdit(QLineEdit):

    def __init__(self, parameter):

        super().__init__()

        self.parameter = parameter

        self.setText(
            str(parameter.value)
        )

        self.editingFinished.connect(
            self.update_parameter
        )

    # =====================================================
    # update
    # =====================================================

    def update_parameter(self):

        text = self.text()

        try:

            value = ast.literal_eval(text)

            if isinstance(
                value,
                list,
            ):

                self.parameter.value = value

        except Exception:

            pass