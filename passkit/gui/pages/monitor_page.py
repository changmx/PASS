from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QCheckBox,
)


class MonitorPage(QWidget):

    def __init__(self, model):

        super().__init__()

        self.model = model

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()

        self.setLayout(layout)

        # ------------------------------------------------
        # stat
        # ------------------------------------------------

        self.stat_box = QCheckBox("Enable Stat Monitor")

        self.stat_box.setChecked(self.model.enable_stat_monitor)

        self.stat_box.stateChanged.connect(self.update_stat)

        layout.addWidget(self.stat_box)

        # ------------------------------------------------
        # dist
        # ------------------------------------------------

        self.dist_box = QCheckBox("Enable Distribution Monitor")

        self.dist_box.setChecked(self.model.enable_dist_monitor)

        self.dist_box.stateChanged.connect(self.update_dist)

        layout.addWidget(self.dist_box)

        # ------------------------------------------------
        # phase
        # ------------------------------------------------

        self.phase_box = QCheckBox("Enable Phase Monitor")

        self.phase_box.setChecked(self.model.enable_phase_monitor)

        self.phase_box.stateChanged.connect(self.update_phase)

        layout.addWidget(self.phase_box)

        layout.addStretch()

    # ------------------------------------------------
    # update
    # ------------------------------------------------

    def update_stat(self, value):

        self.model.enable_stat_monitor = bool(value)

    def update_dist(self, value):

        self.model.enable_dist_monitor = bool(value)

    def update_phase(self, value):

        self.model.enable_phase_monitor = bool(value)
