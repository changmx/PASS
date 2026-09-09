"""Focused interaction tests for the configuration page."""

import os
from collections import Counter
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton

from PASS.gui.app import ConfigPage, PropertyComboBox


def _page() -> ConfigPage:
    app = QApplication.instance() or QApplication([])
    page = ConfigPage()
    page.data = {
        "Sequence": {
            "late": {"S (m)": 4.0, "Command": "Drift", "Length (m)": 1.0},
            "early": {"S (m)": 1.0, "Command": "Quadrupole", "K1L": 0.2},
        }
    }
    page._sync_editor()
    page._refresh_tree()
    return page


def test_browsing_component_does_not_modify_sequence():
    page = _page()
    before = dict(page.data["Sequence"])

    page.select_command("RFCavity")

    assert page.data["Sequence"] == before
    assert page._pending_command == "RFCavity"
    assert page.insert_button.isHidden() is False


def test_confirmed_insert_and_sequence_table_filter():
    page = _page()

    page.select_command("RFCavity")
    page._name_field.setText("rf_ip1")
    page.insert_pending_command()

    assert len(page.data["Sequence"]) == 3
    assert page.data["Sequence"]["rf_ip1"]["Command"] == "RFCavity"
    assert [page.sequence_table.item(row, 0).text() for row in range(page.sequence_table.rowCount())] == [
        "early",
        "late",
        "rf_ip1",
    ]

    page.sequence_filter.setText("quad")
    assert page.sequence_table.isRowHidden(0) is False
    assert page.sequence_table.isRowHidden(1) is True
    assert page.sequence_table.isRowHidden(2) is True

    page.sequence_filter.clear()
    page.sequence_command_filter.setCurrentText("Quadrupole")
    assert page.sequence_table.isRowHidden(0) is False
    assert page.sequence_table.isRowHidden(1) is True
    assert page.sequence_table.isRowHidden(2) is True


def test_duplicate_and_delete_sequence_item(monkeypatch):
    page = _page()
    page._select_sequence_item("early")

    page.duplicate_selected()
    assert "early_copy" in page.data["Sequence"]

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    page._select_sequence_item("early_copy")
    page.delete_selected()
    assert "early_copy" not in page.data["Sequence"]


def test_sequence_table_row_selects_matching_command():
    page = _page()

    # Rows are sorted by S (m), so the first row is the early quadrupole.
    page._sequence_row_clicked(0, 0)

    assert page._selected_path == ("Sequence", "early")
    assert page.form_title.text().endswith("early")


def test_sequence_name_can_be_renamed_from_the_property_form():
    page = _page()
    page._select_sequence_item("early")
    page._name_field.setText("qf1")

    page.apply_form()

    assert "early" not in page.data["Sequence"]
    assert page.data["Sequence"]["qf1"]["Command"] == "Quadrupole"
    assert page._selected_path == ("Sequence", "qf1")


def test_component_library_is_grouped_and_uses_complete_injection_template():
    page = _page()

    assert list(page.library_sections) == [
        "必需项", "Twiss 与光学", "序列工具", "元件", "监测与诊断", "物理模块"
    ]
    injection = page._command_template("Injection")

    assert injection["Command"] == "Injection"
    assert "bunch0" in injection
    quadrupole = page._command_template("Quadrupole")
    assert {"Aperture type", "Aperture value", "Integrator", "Field error KNL"} <= set(quadrupole)


def test_overview_has_two_left_aligned_sections_and_compact_validation_status():
    page = _page()
    assert page.tree_section.header.isChecked() is False
    assert page.sequence_section.header.isChecked() is True
    assert page.tree_section.body.isVisible() is False
    assert page.sequence_section.header.text() == "执行序列详情（Sequence）"
    assert page.tree_section.header.text() == "全局配置与 Sequence"
    assert page.tree_section.header.styleSheet() == "text-align: left;"
    assert page.sequence_section.header.styleSheet() == "text-align: left;"
    page.data["Backend (gpu/cpu)"] = "cpu"
    page._refresh_tree()

    assert [page.editor_tabs.tabText(index) for index in range(page.editor_tabs.count())] == ["配置概览", "JSON 源码"]
    assert page.tree.topLevelItem(0).text(0) == "全局配置"
    assert page.sequence_table.rowCount() == 2
    assert page.validation_label.text() == "配置检查：1 项问题"
    assert "未找到 Injection command" in page.validation_label.toolTip()

    page.data["Sequence"]["injection"] = page._command_template("Injection")
    page._refresh_tree()
    assert page.validation_label.text() == "配置有效"


def test_property_form_uses_typed_fields_and_keeps_command_read_only():
    page = _page()
    command = {
        "Command": "Quadrupole",
        "Is ramping": False,
        "Integrator": "adaptive",
        "Field error KNL": [0.1, 0.2],
        "Aperture": {"type": "circle", "value": [0.02]},
    }
    page.data["Sequence"]["qf"] = command
    page._refresh_tree()
    page._select_sequence_item("qf")

    assert isinstance(page._form_fields["Command"], QLineEdit)
    assert page._form_fields["Command"].isReadOnly()
    assert isinstance(page._form_fields["Is ramping"], QCheckBox)
    assert isinstance(page._form_fields["Integrator"], QComboBox)
    assert isinstance(page._form_fields["Field error KNL"], QPlainTextEdit)
    assert isinstance(page._form_fields["Aperture"], QPlainTextEdit)

    page._form_fields["Field error KNL"].setPlainText("[0.3]")
    page.apply_form()
    assert page.data["Sequence"]["qf"]["Field error KNL"] == [0.3]


def test_property_selectors_ignore_mouse_wheel_even_with_focus():
    page = _page()
    page.data["Sequence"]["early"]["Integrator"] = "adaptive"
    page._refresh_tree()
    page._select_sequence_item("early")
    selector = page._form_fields["Integrator"]

    class WheelEvent:
        ignored = False

        def ignore(self):
            self.ignored = True

    event = WheelEvent()
    selector.setFocus()
    current = selector.currentText()
    selector.wheelEvent(event)

    assert isinstance(selector, PropertyComboBox)
    assert selector.currentText() == current
    assert event.ignored is True


def test_navigation_keeps_or_discards_unconfirmed_form_edits(monkeypatch):
    page = _page()
    page._select_sequence_item("early")
    page._form_fields["K1L"].setText("0.4")

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
    page._select_sequence_item("late")
    assert page._selected_path == ("Sequence", "early")
    assert page._form_dirty is True

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    page._select_sequence_item("late")
    assert page._selected_path == ("Sequence", "late")
    assert page._form_dirty is False


def test_new_project_uses_complete_main_config_and_global_form():
    page = ConfigPage()

    assert {
        "Beam Name", "Circumference (m)", "Number of turns", "Timing", "Sequence",
    } <= set(page.data)
    assert "Is space charge" not in page.data
    assert "Space-charge simulation parameters" not in page.data
    page._populate_root_configuration()
    assert "Beam Name" in page._form_fields
    assert "Sequence" not in page._form_fields
    assert "Timing" not in page._form_fields
    assert isinstance(page._timing_fields["Mode"], QComboBox)


def test_global_config_is_a_required_left_library_entry_and_timing_is_typed():
    page = ConfigPage()

    assert "配置与组件库" in [label.text() for label in page.findChildren(QLabel)]
    assert any(item.text() == "全局配置  · 必需" for item in page.findChildren(QPushButton))
    page.configure_global()
    assert page.form_title.text() == "全局配置"
    assert page.form_layout.rowCount() >= 6
    page._timing_fields["Mode"].setCurrentText("turn")
    page._timing_fields["Log Interval"].setText("25")
    page.apply_form()

    assert page.data["Timing"] == {
        "Mode": "turn", "Log Interval": 25, "Warmup Turns": 1, "Include IO": True,
    }
    assert "配置检查" in page.validation_label.text()


def test_space_charge_global_editor_uses_new_named_configuration_schema():
    page = ConfigPage()
    page.configure_space_charge()

    assert page.form_title.text() == "空间电荷 · 计算配置"
    assert page.data["Space charge"]["Enabled"] is True
    assert page._active_space_charge_configuration == "default"
    assert set(page._space_charge_fields) == {
        "Slice set", "Nx", "Ny", "Grid Width X (m)", "Grid Width Y (m)",
        "Method", "Solver", "Particle Deposition Method",
        "Grid Half Width X (m)", "Grid Half Width Y (m)",
        "Center X (m)", "Center Y (m)", "Angle (rad)", "Sigma (m)",
        "Sigma X (m)", "Sigma Y (m)", "Radius (m)", "Semi-axis A (m)", "Semi-axis B (m)",
    }
    assert page._space_charge_fields["Solver"].itemText(0) == "fft_free_space"
    assert [
        page._space_charge_fields["Particle Deposition Method"].itemText(index)
        for index in range(page._space_charge_fields["Particle Deposition Method"].count())
    ] == ["CIC", "TSC"]

    page.data["Sequence"]["sc_ip"] = page._command_template("SpaceCharge")
    page._space_charge_name_field.setText("round_pipe")
    assert page._space_charge_selector.currentText() == "round_pipe"
    page._space_charge_fields["Nx"].setText("65")
    page._space_charge_fields["Solver"].setCurrentText("dst_dirichlet")
    page.apply_form()

    configuration = page.data["Space charge"]["Configurations"]["round_pipe"]
    assert configuration["Nx"] == 65
    assert configuration["Solver"] == "dst_dirichlet"
    assert page.data["Sequence"]["sc_ip"]["Configuration"] == "round_pipe"
    assert "default" not in page.data["Space charge"]["Configurations"]


def test_space_charge_command_only_contains_point_parameters_and_named_reference():
    page = ConfigPage()
    assert any(item.text() == "插入计算点" for item in page.space_charge_menu.findChildren(QPushButton))
    page.configure_space_charge()
    page._space_charge_name_field.setText("round_pipe")
    page.apply_form()
    page.select_command("SpaceCharge")

    assert isinstance(page._form_fields["Configuration"], PropertyComboBox)
    assert page._form_fields["Configuration"].currentText() == "round_pipe"
    assert set(page._form_fields) == {
        "S (m)", "Command", "Configuration", "SC length (m)",
        "Aperture type", "Aperture value",
        "Save field", "Save potential", "Save density", "Save turns",
    }
    assert "Nx" not in page._form_fields
    assert "Field solver" not in page._form_fields

    page.configure_global()
    assert "Space charge" not in page._form_fields


def test_space_charge_named_configurations_can_be_added_copied_and_deleted(monkeypatch):
    page = ConfigPage()
    page.configure_space_charge()

    page.add_space_charge_configuration()
    assert page._active_space_charge_configuration == "configuration_1"
    assert set(page.data["Space charge"]["Configurations"]) == {"default", "configuration_1"}

    page.copy_space_charge_configuration()
    assert page._active_space_charge_configuration == "configuration_1_copy"
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    page.delete_space_charge_configuration()
    assert "configuration_1_copy" not in page.data["Space charge"]["Configurations"]


def test_deleting_referenced_space_charge_configuration_requires_replacement(monkeypatch):
    page = ConfigPage()
    page.configure_space_charge()
    page.add_space_charge_configuration()
    page.data["Sequence"]["sc_a"] = page._command_template("SpaceCharge")
    page.data["Sequence"]["sc_b"] = page._command_template("SpaceCharge")
    page._space_charge_selector.setCurrentText("default")

    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("configuration_1", True))
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    page.delete_space_charge_configuration()

    assert "default" not in page.data["Space charge"]["Configurations"]
    assert page.data["Sequence"]["sc_a"]["Configuration"] == "configuration_1"
    assert page.data["Sequence"]["sc_b"]["Configuration"] == "configuration_1"
    page._select_sequence_item("sc_a")
    assert [
        page._form_fields["Configuration"].itemText(index)
        for index in range(page._form_fields["Configuration"].count())
    ] == ["configuration_1"]


def test_deleting_only_referenced_space_charge_configuration_is_blocked(monkeypatch):
    page = ConfigPage()
    page.configure_space_charge()
    page.data["Sequence"]["sc_a"] = page._command_template("SpaceCharge")
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2]))

    page.delete_space_charge_configuration()

    assert "default" in page.data["Space charge"]["Configurations"]
    assert "sc_a" in messages[0]


def test_validation_status_uses_tooltip_for_missing_injection_detail():
    page = ConfigPage()

    assert page.validation_label.text() == "配置检查：1 项问题"
    assert "未找到 Injection command" in page.validation_label.toolTip()
    page.data["Sequence"]["injection"] = page._command_template("Injection")
    page._refresh_tree()
    assert page.validation_label.text() == "配置有效"


def test_injection_uses_structured_bunch_editor_and_normalizes_groups():
    page = _page()
    page.data["Sequence"]["injection"] = page._command_template("Injection")
    page._refresh_tree()
    page._select_sequence_item("injection")

    assert "bunch0" not in page._form_fields
    assert page._bunch_selector is not None
    assert page._bunch_selector.currentText() == "bunch0"
    page.add_bunch()
    assert page.data["Sequence"]["injection"]["Harmonic Number"] == 2
    assert page.data["Sequence"]["injection"]["bunch1"]["Harmonic ID of this bunch"] == 1

    page.copy_bunch()
    assert page.data["Sequence"]["injection"]["Harmonic Number"] == 3
    page.delete_bunch()
    assert page.data["Sequence"]["injection"]["Harmonic Number"] == 2
    assert [page.data["Sequence"]["injection"][f"bunch{i}"]["Harmonic ID of this bunch"] for i in range(2)] == [0, 1]


def test_twiss_template_and_madx_import_append_unique_items(tmp_path, monkeypatch):
    page = _page()
    twiss = page._command_template("Twiss")
    assert twiss["Command"] == "Twiss"
    assert twiss["S previous (m)"] == twiss["S (m)"]

    source = tmp_path / "ring.tfs"
    source.write_text("placeholder", encoding="utf-8")

    class FakeItem:
        def model_dump(self, by_alias):
            return {"Command": "Drift", "S (m)": 1.0, "Length (m)": 1.0}

    calls = {}

    def fake_reader(path, **kwargs):
        calls.update(kwargs)
        return [FakeItem()], ["late"], 42.0

    monkeypatch.setattr("PASS.para.madx.read_madx_elements", fake_reader)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    page.configure_madx_import()
    page._madx_fields["Twiss TFS 文件"].setText(str(source))
    page.import_madx_twiss()

    assert "late_2" in page.data["Sequence"]
    assert page.data["Circumference (m)"] == 42.0
    assert calls["is_merge_drift"] is True


def test_madx_import_modes_are_selected_by_entry_point():
    page = ConfigPage()

    page.configure_madx_elements()
    assert "mode" not in page._madx_fields
    assert page._madx_fields["merge_drift"].currentText() == "是"
    assert page._madx_fields["merge_drift"].isEnabled()

    page.configure_madx_twiss()
    assert "mode" not in page._madx_fields
    assert page._madx_fields["merge_drift"].currentText() == "是"
    assert page._madx_fields["merge_drift"].isEnabled()


def test_sequence_validation_reports_the_invalid_command_field():
    page = _page()
    page.data["Sequence"]["bad_drift"] = {
        "S (m)": 2.0,
        "Command": "Drift",
        "Length (m)": -0.5,
    }

    issues = page._validate_configuration()

    assert any("Sequence.bad_drift" in issue and "Length (m)" in issue for issue in issues)


def test_invalid_json_reports_location_and_moves_editor_cursor(monkeypatch):
    page = _page()
    page.editor.setPlainText('{\n  "Sequence":\n}')
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))

    assert page.apply_json() is False

    assert "第 3 行" in messages[0]
    assert page.editor.textCursor().position() == 16


def test_madx_preview_does_not_mutate_sequence_and_import_reuses_preview(tmp_path, monkeypatch):
    page = _page()
    source = tmp_path / "ring.tfs"
    source.write_text("placeholder", encoding="utf-8")

    class FakeItem:
        def model_dump(self, by_alias):
            return {"Command": "Drift", "S (m)": 3.0, "Length (m)": 1.0}

    calls = []

    def fake_reader(path, **kwargs):
        calls.append((path, kwargs))
        return [FakeItem()], ["preview_drift"], 42.0

    messages = []
    monkeypatch.setattr("PASS.para.madx.read_madx_elements", fake_reader)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[1:]))
    page.configure_madx_elements()
    page._madx_fields["Twiss TFS 文件"].setText(str(source))
    before = deepcopy(page.data["Sequence"])

    page.preview_madx_import()

    assert page.data["Sequence"] == before
    assert len(calls) == 1
    assert "导入预览" in messages[0][0]
    page.import_madx_twiss()
    assert len(calls) == 1
    assert "preview_drift" in page.data["Sequence"]


def test_madx_preview_formats_element_and_twiss_source_statistics():
    counts = Counter({"drift": 12, "quadrupole": 4, "sextupole": 2, "monitor": 3, "hkicker": 1, "foo": 5})

    element_summary = ConfigPage._format_madx_type_counts(counts)
    twiss_summary = ConfigPage._format_madx_type_counts(counts, twiss_points=True)

    assert "Drift：12" in element_summary
    assert "四极铁：4" in element_summary
    assert "监测器：3" in element_summary
    assert "水平校正铁：1" in element_summary
    assert "其他（FOO）：5" in element_summary
    assert "四极铁处 Twiss 点：4" in twiss_summary
    assert counts["quadrupole"] == 4
