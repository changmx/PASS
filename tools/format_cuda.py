"""Check or format embedded CUDA with clang-format 23 and unchanged C++ tokens."""

import argparse
import ast
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tokenize


def _find_clang_format(requested):
    requested = requested or os.environ.get("CLANG_FORMAT")
    if requested:
        candidates = [requested]
    else:
        candidates = [shutil.which(name) for name in ("clang-format", "clang-format-23")]
        for folder in (".vscode", ".vscode-insiders"):
            extensions = Path.home() / folder / "extensions"
            candidates.extend(sorted(extensions.glob("ms-vscode.cpptools-*/LLVM/bin/clang-format*"), reverse=True))
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run([str(candidate), "--version"], capture_output=True, text=True, check=True)
            if re.search(r"clang-format version 23\.", result.stdout):
                return str(candidate), result.stdout.strip()
            errors.append(f"{candidate}: {result.stdout.strip()}")
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{candidate}: {error}")
    detail = "\n".join(errors)
    raise ValueError("clang-format 23.x is required. Use --clang-format, CLANG_FORMAT, PATH, or the VS Code C/C++ extension.\n" + detail)


def _tokenize_cpp(source):
    pattern = (r'//[^\n]*|/\*[\s\S]*?\*/'
               r'|(?:u8|[uUL])?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)"(?:[A-Za-z_]\w*)?'
               r'|(?:u8|[uUL])?"(?:\\[\s\S]|[^"\\])*"(?:[A-Za-z_]\w*)?'
               r"|(?:u8|[uUL])?'(?:\\[\s\S]|[^'\\])*'(?:[A-Za-z_]\w*)?"
               r"|(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*"
               r"|[a-zA-Z_]\w*"
               r"|%:%:|<=>|>>=|<<=|->\*|\.\.\.|##|::|\.\*|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\|"
               r"|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<:|:>|<%|%>|%:|\S")
    # C++ raw strings retain physical backslash-newline pairs.
    parts = []
    start = 0
    for match in re.finditer(pattern, source):
        if match["delimiter"] is not None:
            parts.append(re.sub(r"\\\r?\n", "", source[start:match.start()]))
            parts.append(match[0])
            start = match.end()
    parts.append(re.sub(r"\\\r?\n", "", source[start:]))
    source = "".join(parts)
    return [match[0] for match in re.finditer(pattern, source)]


def _read_directives(source):
    source = re.sub(r"\\\r?\n", "", source)
    directives = []
    for line in source.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        tokens = _tokenize_cpp(line)
        function_macro = re.match(r"\s*#\s*define\s+\w+\(", line) is not None
        # Include paths and stringification can depend on whitespace within tokens.
        if re.match(r"\s*#\s*(include|pragma|error|warning)\b", line) or "#" in tokens[1:]:
            directives.append((line.strip(), function_macro))
        else:
            directives.append((tokens, function_macro))
    return directives


def _verify_cpp(before, after):
    tokens_before = _tokenize_cpp(before)
    directives_before = _read_directives(before)
    if tokens_before != _tokenize_cpp(after):
        raise ValueError("formatting changed C++ tokens")
    if directives_before != _read_directives(after):
        raise ValueError("formatting changed preprocessor directives")
    if before != after:
        if "__LINE__" in tokens_before:
            raise ValueError("source uses __LINE__; review line-sensitive formatting manually")
        for directive in directives_before:
            tokens = directive[0] if isinstance(directive[0], list) else _tokenize_cpp(directive[0])
            if "#" in tokens[1:]:
                raise ValueError("source uses macro stringification; review whitespace manually")


def _run_clang_format(source, executable, config):
    result = subprocess.run(
        [executable, "--assume-filename=kernel.cu", f"--style=file:{config}"],
        input=source,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return result.stdout


def _format_kernel_macros(source, executable, config):
    lines = source.splitlines(keepends=True)
    output = []
    macro_names = []
    index = 0
    while index < len(lines):
        header = lines[index]
        if not re.match(r"\s*#\s*define\b", header) or not header.rstrip().endswith("\\"):
            output.append(header)
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].rstrip().endswith("\\"):
            index += 1
        if index == len(lines):
            raise ValueError("unterminated macro continuation")
        index += 1
        block = "".join(lines[start:index])
        if not re.search(r"\b__(global|device)__\b", block):
            output.append(block)
            continue
        body = re.sub(r"\\\r?\n", "", "".join(lines[start + 1:index])).strip()
        replacements = {}

        def replace_paste(match):
            name = f"pass_format_macro_{len(replacements)}"
            if name in source:
                raise ValueError("macro placeholder conflicts with source")
            replacements[name] = match[0]
            return name

        body = re.sub(r"\b\w+(?:\s*##\s*\w+)+\b", replace_paste, body)
        formatted = _run_clang_format(body, executable, config).rstrip()
        for name, value in replacements.items():
            formatted = formatted.replace(name, value)
        body_lines = formatted.splitlines()
        rendered = [header.rstrip().removesuffix("\\").rstrip() + " \\\n"]
        rendered.extend("    " + line + (" \\\n" if i < len(body_lines) - 1 else "\n") for i, line in enumerate(body_lines))
        replacement = "".join(rendered)
        _verify_cpp(block, replacement)
        output.append(replacement)
        macro_names.append(re.match(r"\s*#\s*define\s+(\w+)", header)[1])
    formatted = "".join(output)
    for name in macro_names:
        formatted = re.sub(rf"\b{re.escape(name)}\(\s*(\w+)\s*\)", rf"{name}(\1)", formatted)
    return formatted


def _format_cuda(source, executable, config):
    output = _run_clang_format(source, executable, config)
    output = _format_kernel_macros(output, executable, config)
    if source.startswith("\n") and not output.startswith("\n"):
        output = "\n" + output
    _verify_cpp(source, output)
    return output


def _format_python(source, executable, config):
    tree = ast.parse(source)
    dynamic_parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
        and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    }
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(row, column):
        return starts[row - 1] + len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))

    edits = []
    skipped = []
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or id(node) in docstrings:
            continue
        if id(node) in dynamic_parts:
            if re.search(r"\b__(global|device)__\b", node.value):
                skipped.append((node.lineno, "dynamic f-string; review manually"))
            continue
        if not re.search(r"\b__(global|device)__\s+\w", node.value):
            continue
        literal = ast.get_source_segment(source, node)
        strings = [token for token in tokenize.generate_tokens(io.StringIO(literal).readline) if token.type == tokenize.STRING]
        match = re.match(r"(?i)(r?)(\"\"\"|''')", literal)
        if not match or len(strings) != 1:
            skipped.append((node.lineno, "single-line or concatenated fragment; review manually"))
            continue
        output = _format_cuda(node.value, executable, config)
        count += 1
        if output == node.value:
            continue
        quote = match[2]
        body = output if match[1] else output.replace("\\", "\\\\")
        if match[1] and quote in body:
            raise ValueError(f"line {node.lineno}: formatted source contains the Python quote delimiter")
        if not match[1]:
            body = body.replace(quote, "\\" + quote)
        replacement = match[0] + body + quote
        if ast.literal_eval(replacement) != output:
            raise ValueError(f"line {node.lineno}: Python string value changed while escaping")
        edits.append((offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset), replacement))
        node.value = output
    output = source
    for start, end, replacement in sorted(edits, reverse=True):
        output = output[:start] + replacement + output[end:]
    if ast.dump(ast.parse(output)) != ast.dump(tree):
        raise ValueError("formatting changed Python syntax outside the selected strings")
    return output, count, skipped


def _collect_files(paths):
    files = set()
    excluded = {"tests", "output", "runs", "build", "__pycache__", "node_modules", "venv"}
    for path in paths:
        path = path.resolve()
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            for folder, directories, names in os.walk(path):
                directories[:] = [name for name in directories if name not in excluded and not name.startswith(".")]
                files.update(Path(folder) / name for name in names if name.endswith(".py"))
        else:
            raise ValueError(f"not a Python file or directory: {path}")
    return sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", help="Python files or directories; defaults to PASS/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report formatting differences without writing (default)")
    mode.add_argument("--write", action="store_true", help="write changes after validating all selected files")
    parser.add_argument("--clang-format", help="path or command for clang-format 23.x")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        executable, version = _find_clang_format(args.clang_format)
        files = _collect_files(args.paths or [root / "PASS"])
        changes = []
        block_count = 0
        skipped_count = 0
        for path in files:
            raw = path.read_bytes()
            encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
            source = raw.decode(encoding).replace("\r\n", "\n")
            try:
                output, count, skipped = _format_python(source, executable, root / ".clang-format")
            except (ValueError, SyntaxError, tokenize.TokenError) as error:
                raise ValueError(f"{path}: {error}") from error
            block_count += count
            skipped_count += len(skipped)
            for line, reason in skipped:
                print(f"SKIP {path}:{line}: {reason}")
            if output != source:
                if b"\r\n" in raw:
                    output = output.replace("\n", "\r\n")
                changes.append((path, output.encode(encoding)))
        print(version)
        for path, data in changes:
            if args.write:
                path.write_bytes(data)
            print(f"{'FORMATTED' if args.write else 'WOULD FORMAT'} {path}")
        print(f"{len(files)} Python files, {block_count} CUDA blocks, {len(changes)} changed files, {skipped_count} manual-review fragments.")
        return 0 if args.write or not changes else 1
    except (OSError, ValueError, SyntaxError, tokenize.TokenError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
