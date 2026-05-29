from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QStackedWidget,
    QPushButton,
    QFileDialog,
)

from gui.pages.main_para_page import (MainParaPage)

from gui.pages.injection_page import (InjectionPage)

from gui.widgets.console_widget import (ConsoleWidget)

from gui.utils.version import (read_pass_version)

from para.models.main_para import (MainPara)

from para.models.injection_para import (InjectionPara)

from para.builders.main_builder import (MainBuilder)

from para.builders.injection_builder import (InjectionBuilder)

from para.builders.simulation_builder import (SimulationBuilder)

from para.exporters.json_exporter import (JsonExporter)

from gui.core.logger import logger

import os


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        logger.enable_gui = True

        version = read_pass_version()

        self.setWindowTitle(f"PASS v{version}")

        self.resize(1400, 900)

        self.setup_models()

        self.setup_ui()

    def setup_models(self):

        self.main_para = MainPara()

        self.injection_para = InjectionPara()

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        root_layout = QVBoxLayout()

        central.setLayout(root_layout)

        # toolbar

        toolbar_layout = QHBoxLayout()

        root_layout.addLayout(toolbar_layout)

        self.generate_button = QPushButton("Generate JSON")

        self.generate_button.clicked.connect(self.generate_json)

        toolbar_layout.addWidget(self.generate_button)

        self.run_button = QPushButton("Run PASS")

        toolbar_layout.addWidget(self.run_button)

        toolbar_layout.addStretch()

        # center

        center_layout = QHBoxLayout()

        root_layout.addLayout(center_layout)

        # nav

        self.nav = QListWidget()

        center_layout.addWidget(self.nav, 1)

        # pages

        self.stack = QStackedWidget()

        center_layout.addWidget(self.stack, 4)

        # console

        self.console = ConsoleWidget()

        self.console.setMaximumHeight(200)

        root_layout.addWidget(self.console)

        self.setup_pages()

        self.setup_navigation()

    def setup_pages(self):

        self.main_page = MainParaPage(self.main_para)

        self.inj_page = InjectionPage(self.injection_para)

        self.stack.addWidget(self.main_page)

        self.stack.addWidget(self.inj_page)

    def setup_navigation(self):

        self.nav.addItem("Main Parameters")

        self.nav.addItem("Injection")

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        self.nav.setCurrentRow(0)

    def generate_json(self):

        try:

            default_output_dir = os.path.join(os.getcwd(), "input")

            default_filename = "beam0.json"

            path, _ = QFileDialog.getSaveFileName(self, "Save JSON", os.path.join(default_output_dir, default_filename), "JSON (*.json)")

            if not path:

                return

            builder = SimulationBuilder()

            builder.set_main_para(MainBuilder.build(self.main_para))

            builder.add_module("Injection", InjectionBuilder.build(self.injection_para))

            data = builder.build()

            JsonExporter.export(data, path)

            # self.console.log(f"JSON generated:\n{path}")
            logger.info(f"JSON generated: {path}")

        except Exception as e:

            # self.console.error(str(e))
            logger.error(str(e))
