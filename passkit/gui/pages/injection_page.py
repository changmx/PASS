from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
)

from PySide6.QtCore import Qt

from passkit.gui.widgets.parameter_widget_factory import (ParameterWidgetFactory)

from passkit.para.models.injection_para import (
    BunchPara,
    InjectionPara,
)


class BunchTabContent(QWidget):
    """
    Content widget for a single bunch's tab.
    Displays all parameters inside a grid layout within a scroll area.
    """

    def __init__(self, bunch: BunchPara, bunch_index: int):
        super().__init__()

        self.bunch = bunch
        self.bunch_index = bunch_index
        self.widgets = {}  # name -> widget

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        # scroll area in case bunch has many parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        grid = QGridLayout()
        content.setLayout(grid)

        grid.setContentsMargins(20, 20, 20, 20)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        row = 0
        for name, param in vars(self.bunch).items():

            label = QLabel(param.description)
            label.setMinimumWidth(200)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            widget = ParameterWidgetFactory.create_widget(param)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)

            self.widgets[name] = widget

            row += 1

        # bottom stretch
        grid.setRowStretch(row, 1)

        # connect conditional disabling rules
        self._connect_disable_rules()

    # =========================================================
    # disable rules
    # =========================================================

    def _connect_disable_rules(self):
        """
        Build signal connections from each source parameter's widget
        to all target widgets according to each Parameter's disables list.
        """
        # Collect all rules: (src_name, tgt_name, when_value)
        self._rules = []
        for src_name, src_param in vars(self.bunch).items():
            for rule in (src_param.disables or []):
                for tgt_name in rule.targets:
                    self._rules.append((src_name, tgt_name, rule.when_value))

        if not self._rules:
            return

        # Collect all source names
        self._source_names = {src for src, _, _ in self._rules}

        # Connect each source widget's signal to the master update function
        for src_name in self._source_names:
            src_widget = self.widgets.get(src_name)
            if src_widget is None:
                continue

            if hasattr(src_widget, 'stateChanged'):
                src_widget.stateChanged.connect(self._update_from_rules)
            elif hasattr(src_widget, 'currentTextChanged'):
                src_widget.currentTextChanged.connect(self._update_from_rules)
            elif hasattr(src_widget, 'textChanged'):
                src_widget.textChanged.connect(self._update_from_rules)
            elif hasattr(src_widget, 'valueChanged'):
                src_widget.valueChanged.connect(self._update_from_rules)

        # Apply initial state
        self._update_from_rules()

    def _update_from_rules(self):
        """
        Iteratively compute disabled states, handling cascade:
        If a source is disabled, its rules should not apply.
        """
        disabled_sources = set()
        disabled_targets = set()
        changed = True

        while changed:
            changed = False

            new_disabled_targets = set()
            new_disabled_sources = set(disabled_sources)

            for src_name, tgt_name, when_value in self._rules:
                # Skip rules whose source is disabled
                if src_name in new_disabled_sources:
                    continue

                src_val = getattr(self.bunch, src_name).value
                if src_val == when_value:
                    new_disabled_targets.add(tgt_name)
                    # If target is also a source, mark it as disabled source
                    if tgt_name in self._source_names and tgt_name not in new_disabled_sources:
                        new_disabled_sources.add(tgt_name)
                        changed = True

            # Check if state changed
            if (new_disabled_targets != disabled_targets or new_disabled_sources != disabled_sources):
                disabled_targets = new_disabled_targets
                disabled_sources = new_disabled_sources
                changed = True

        # Apply to all target widgets
        all_touched = set(tgt for _, tgt, _ in self._rules)
        for tgt_name in all_touched:
            widget = self.widgets.get(tgt_name)
            if widget is not None:
                widget.setEnabled(tgt_name not in disabled_targets)


class InjectionPage(QWidget):
    """
    Injection page supporting multiple bunches using closable tabs.

    Each bunch is displayed in a separate tab with a close button (✕).
    User can click the close button to delete that bunch directly.
    """

    def __init__(self, model: InjectionPara):

        super().__init__()

        self.model = model
        self.tab_widget: QTabWidget | None = None

        self.setup_ui()

    # =========================================================
    # ui
    # =========================================================

    def setup_ui(self):

        root = QVBoxLayout()
        self.setLayout(root)

        # -----------------------------------------------------
        # toolbar (add bunch only)
        # -----------------------------------------------------

        toolbar = QHBoxLayout()
        root.addLayout(toolbar)

        self.add_btn = QPushButton("+ Add Bunch")
        self.add_btn.clicked.connect(self.add_bunch)
        toolbar.addWidget(self.add_btn)

        toolbar.addStretch()

        # -----------------------------------------------------
        # tab widget (each bunch is a closable tab)
        # -----------------------------------------------------

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        root.addWidget(self.tab_widget)

        # rebuild tabs from model
        self._rebuild_from_model()

    # =========================================================
    # add / close
    # =========================================================

    def add_bunch(self):
        """Add a new bunch to the model and rebuild tabs."""

        new_index = len(self.model.bunches)
        bunch = BunchPara()
        self.model.bunches.append(bunch)

        self._rebuild_from_model()
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    def close_tab(self, index: int):
        """Close the tab at the given index and remove its bunch from the model."""

        if len(self.model.bunches) <= 1:
            QMessageBox.information(self, "Cannot Remove", "At least one bunch is required.")
            return

        self.model.bunches.pop(index)
        self._rebuild_from_model()

    # =========================================================
    # rebuild
    # =========================================================

    def _rebuild_from_model(self):
        """Clear all tabs and rebuild from the model's bunch list."""

        self.tab_widget.clear()

        for i, bunch in enumerate(self.model.bunches):
            tab = BunchTabContent(bunch, i)
            self.tab_widget.addTab(tab, f"Bunch {i}")
