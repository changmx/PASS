"""Compact Cartesian scan editing inside an Injection bunch form."""

from PySide6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QVBoxLayout, QWidget

from PASS.gui.structured import IntegerSpinBox, StructuredField


class ScanGridEditor(StructuredField):
    """Keep an optional grid and incomplete form drafts without allocating particles."""

    def __init__(self, value=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("生成 x–y 扫描网格")
        layout.addWidget(self.enabled)
        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        controls_layout.addLayout(form)
        self.x_min, self.x_max = QLineEdit("-10"), QLineEdit("10")
        self.y_min, self.y_max = QLineEdit("-10"), QLineEdit("10")
        self.n_x, self.n_y = IntegerSpinBox(41, minimum=1), IntegerSpinBox(41, minimum=1)
        for name, first, last, count in (("x / mm", self.x_min, self.x_max, self.n_x), ("y / mm", self.y_min, self.y_max, self.n_y)):
            first.setMinimumWidth(56)
            last.setMinimumWidth(56)
            count.setMaximumWidth(72)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSizeConstraint(QLayout.SetMinimumSize)
            for field in (first, QLabel("至"), last, QLabel("点数"), count):
                row_layout.addWidget(field)
            form.addRow(name, row)
        self.dp_values = QLineEdit("-0.01, 0, 0.01")
        self.dp_values.setToolTip("逗号或空格分隔，例如 -0.003, 0, 0.003；0.003 表示动量偏差 +0.3%。")
        form.addRow("dp 列表", self.dp_values)
        self.px, self.py, self.z = QLineEdit("0"), QLineEdit("0"), QLineEdit("0")
        fixed = QWidget()
        fixed_layout = QHBoxLayout(fixed)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.setSizeConstraint(QLayout.SetMinimumSize)
        for label, field in (("px", self.px), ("py", self.py), ("z / m", self.z)):
            fixed_layout.addWidget(QLabel(label))
            fixed_layout.addWidget(field)
        form.addRow("固定坐标", fixed)
        # QFormLayout spanning rows can shrink wrapped text in nested forms.
        # A vertical layout propagates each label's height-for-width instead.
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        controls_layout.addWidget(self.summary)
        self.note = QLabel("每个 dp 重复完整 x–y 网格；粒子在偏移之后插入。与手动坐标、指定粒子文件三选一。")
        self.note.setWordWrap(True)
        controls_layout.addWidget(self.note)
        layout.addWidget(self.controls)
        self._text_fields = {key: getattr(self, key) for key in ("x_min", "x_max", "y_min", "y_max", "dp_values", "px", "py", "z")}
        if value is not None:
            from PASS.para.schema.bunch import ScanGridConfig

            value = ScanGridConfig.model_validate(value).model_dump(by_alias=True)
            for axis in ("x", "y"):
                bounds = value[f"{axis.upper()} range (m)"]
                getattr(self, f"{axis}_min").setText(str(bounds[0] * 1e3))
                getattr(self, f"{axis}_max").setText(str(bounds[1] * 1e3))
                getattr(self, f"n_{axis}").setValue(value[f"Number of {axis} points"])
            self.dp_values.setText(", ".join(str(item) for item in value["dp values"]))
            for key in ("px", "py", "z"):
                getattr(self, key).setText(str(value["z (m)" if key == "z" else key]))
            self.enabled.setChecked(True)
        for field in self._text_fields.values():
            field.textChanged.connect(self._edited)
        self.n_x.valueChanged.connect(self._edited)
        self.n_y.valueChanged.connect(self._edited)
        self.n_x.lineEdit().textEdited.connect(self._edited)
        self.n_y.lineEdit().textEdited.connect(self._edited)
        self.enabled.toggled.connect(self._edited)
        self._edited()

    def get_value(self):
        if not self.enabled.isChecked():
            return None
        from PASS.para.schema.bunch import ScanGridConfig

        dp_values = [float(value) for value in self.dp_values.text().replace("，", ",").replace(",", " ").split()]
        grid = ScanGridConfig(x_range=[float(self.x_min.text()) * 1e-3, float(self.x_max.text()) * 1e-3],
                              y_range=[float(self.y_min.text()) * 1e-3, float(self.y_max.text()) * 1e-3],
                              num_x=int(self.n_x.text()),
                              num_y=int(self.n_y.text()),
                              dp_values=dp_values,
                              px=float(self.px.text()),
                              py=float(self.py.text()),
                              z=float(self.z.text()))
        return grid.model_dump(by_alias=True)

    def _edited(self, *_args):
        self.controls.setVisible(self.enabled.isChecked())
        try:
            grid = self.get_value()
            if grid is None:
                self.summary.clear()
            else:
                n_x, n_y, n_dp = grid["Number of x points"], grid["Number of y points"], len(grid["dp values"])
                self.summary.setText(f"{n_x} × {n_y} × {n_dp} = {n_x * n_y * n_dp:,} 个粒子")
        except (ValueError, TypeError) as exc:
            self.summary.setText("请填写有效网格：" + str(exc))
        self.changed.emit()

    def capture_draft(self):
        return dict(enabled=self.enabled.isChecked(),
                    n_x=self.n_x.text(),
                    n_y=self.n_y.text(),
                    fields={
                        key: field.text()
                        for key, field in self._text_fields.items()
                    })

    def restore_draft(self, state):
        for key, text in state["fields"].items():
            self._text_fields[key].setText(text)
        for key in ("n_x", "n_y"):
            field = getattr(self, key)
            text = str(state[key])
            try:
                field.setValue(int(text))
            except ValueError:
                pass
            field.lineEdit().setText(text)
        self.enabled.setChecked(state["enabled"])
        self._edited()
