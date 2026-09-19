"""Isolated conversion jobs: cancellable without blocking or importing Qt."""
from dataclasses import asdict
import json
from pathlib import Path
import sys


def execute(request):
    from PASS.tool.data_conversion import DataSelection, convert_file, inspect_file, preview_file, file_signature
    source = request["source"]
    action = request["action"]
    if action == "inspect":
        return inspect_file(source)
    selection = DataSelection(**request.get("selection", {}))
    if action == "preview":
        return preview_file(source, selection, check_sdds=True)
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
