"""Isolated conversion jobs: cancellable without blocking or importing Qt."""
from dataclasses import asdict
import json
from pathlib import Path
import sys


def _preview_with_structure(source, selection, inspection_signature):
    from PASS.tool.data_conversion import file_signature, inspect_file, preview_file

    before = file_signature(source)
    if inspection_signature is None or inspection_signature == before:
        result = preview_file(source, selection, check_sdds=True)
        if result["signature"] != before:
            raise ValueError("文件在预览时发生变化，请重新预览。")
        return result
    inspection = inspect_file(source)
    if inspection["signature"] != before or file_signature(source) != before:
        raise ValueError("文件在重新读取结构时发生变化，请重新预览。")
    try:
        fields = inspection["fields"]
        groups = []
        if inspection["format"] in {"csv", "tfs"}:
            groups.append((selection.columns or [], {field["name"] for field in fields if field["kind"] == "column"}))
        elif inspection["format"] == "hdf5":
            groups.append((selection.datasets, {field["name"] for field in fields if field["kind"] == "dataset"}))
        elif inspection["format"] == "sdds":
            groups.append((selection.bpms or [], {field["bpm_name"] for field in fields if field["kind"] == "bpm"}))
            groups.append((selection.bunch_ids or [], set(inspection.get("bunch_ids", []))))
        missing = [str(name) for selected, available in groups for name in selected if name not in available]
        if missing:
            raise ValueError("来源结构已变化，所选字段已删除或更名：" + ", ".join(missing) + "。请取消失效选择或重新选择字段。")
        result = preview_file(source, selection, check_sdds=True)
    except Exception as exc:
        if file_signature(source) != before:
            raise ValueError("文件在重新预览时发生变化，请重试。") from exc
        # Publish fresh choices even when a deleted selection prevents preview.
        return {"inspection": inspection, "preview_error": f"{type(exc).__name__}: {exc}"}
    if result["signature"] != before:
        raise ValueError("文件在重新预览时发生变化，请重试。")
    return {**result, "inspection": inspection}


def execute(request):
    source = request["source"]
    action = request["action"]
    if action in {"wake_preview", "wake_convert"}:
        from PASS.tool.wake_conversion import convert_wake_file, preview_wake_file
        if action == "wake_preview":
            return preview_wake_file(source, **request["options"])
        return convert_wake_file(source,
                                 request["destination"],
                                 expected_sha256=request["expected_sha256"],
                                 overwrite=request.get("overwrite", False),
                                 **request["options"])
    from PASS.tool.data_conversion import DataSelection, convert_file, inspect_file, file_signature
    if action == "inspect":
        return inspect_file(source)
    selection = DataSelection(**request.get("selection", {}))
    if action == "preview":
        return _preview_with_structure(source, selection, request.get("inspection_signature"))
    if action == "plan":
        if request["signature"] != file_signature(source):
            raise ValueError("文件在预览后发生变化，请重新打开并预览。")
        target = Path(request["destination"])
        paths = [target]
        if request.get("csv_metadata", True) and target.suffix.lower() == ".csv":
            paths += [Path(str(path) + ".metadata.json") for path in paths.copy()]
        return {"paths": [str(p) for p in paths], "existing": [str(p) for p in paths if p.exists()]}
    if action == "convert":

        def progress(index, label, rows):
            print(json.dumps({"progress": f"{label}: {rows:,} rows"}), flush=True)

        result = convert_file(source,
                              request["destination"],
                              selection,
                              overwrite=request.get("overwrite", False),
                              csv_metadata=request.get("csv_metadata", True),
                              sdds_mode=request.get("sdds_mode", "binary"),
                              expected_signature=request["signature"],
                              progress=progress,
                              staging_directory=request.get("staging_directory"))
        return asdict(result)
    raise ValueError("Unknown conversion action")


def main():
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        result = execute(request)
        print(json.dumps({"result": result}, ensure_ascii=True, allow_nan=False), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
