"""Structured diagnostics and lossless, strict JSON decoding."""
from dataclasses import asdict, dataclass, field
import json
import math


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    path: tuple
    message: str
    source: str = ""

    @property
    def pointer(self):
        return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in self.path)

    def __str__(self):
        prefix = f"{self.source}: " if self.source else ""
        return f"{prefix}{self.pointer or '/'}: {self.message}"


@dataclass
class ValidationReport:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    command_count: int = 0
    full: bool = True

    @property
    def errors(self):
        return [item for item in self.diagnostics if item.severity == "error"]

    @property
    def warnings(self):
        return [item for item in self.diagnostics if item.severity == "warning"]

    @property
    def ok(self):
        return not self.errors

    def add(self, path, code, message, severity="error", source=""):
        issue = Diagnostic(severity, code, tuple(path), str(message), source)
        if issue not in self.diagnostics:
            self.diagnostics.append(issue)

    def to_dict(self):
        return {"valid": self.ok, "full": self.full, "commands": self.command_count,
                "errors": len(self.errors), "warnings": len(self.warnings),
                "checked_files": self.checked_files,
                "diagnostics": [{**asdict(item), "pointer": item.pointer} for item in self.diagnostics]}

    def text(self):
        summary = f"{len(self.errors)} 错误，{len(self.warnings)} 警告；检查 {self.command_count} 个命令、{len(self.checked_files)} 个文件"
        return summary + "\n" + "\n".join(f"[{d.severity} / {d.code}] {d}" for d in self.diagnostics)


class _Pairs(list):
    """Keep duplicate object members until their full path is known."""


def parse_json(content: str | bytes, report=None):
    report = report if report is not None else ValidationReport()
    try:
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        value = json.loads(content, object_pairs_hook=_Pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        report.add((), "json.syntax", str(exc))
        return None, report

    def unpack(value, path):
        if isinstance(value, _Pairs):
            result, seen = {}, set()
            for key, child in value:
                if key.casefold() in seen:
                    report.add((*path, key), "json.duplicate", "重复的 JSON 键（包括大小写冲突）；引擎会覆盖其中一个值")
                seen.add(key.casefold())
                result[key] = unpack(child, (*path, key))
            return result
        if isinstance(value, list):
            return [unpack(child, (*path, i)) for i, child in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            report.add(path, "json.nonfinite", "必须是有限数值；不允许 NaN、Infinity 或数值溢出")
        return value

    try:
        value = unpack(value, ())
    except RecursionError:
        report.add((), "json.depth", "JSON 嵌套过深")
        return None, report
    if not isinstance(value, dict):
        report.add((), "json.root", "JSON 根节点必须是对象")
        return None, report
    return value, report
