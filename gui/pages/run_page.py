from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTextEdit,
)


class RunPage(QWidget):

    def __init__(self):

        super().__init__()

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()

        self.setLayout(layout)

        self.run_button = QPushButton("Run PASS")

        layout.addWidget(self.run_button)

        self.output_text = QTextEdit()

        self.output_text.setReadOnly(True)

        layout.addWidget(self.output_text)

        self.run_button.clicked.connect(self.run_simulation)

    def run_simulation(self):

        self.output_text.append("Simulation started...")

        # future:
        # PassRunner.run(...)
