"""
Safe(r) execution of LLM-generated analysis code.

Two layers of defence:

1. Static check (`check_code`) - parse the code and reject anything outside a
   small allow-list: only data-science imports, no eval/exec/open/__import__,
   no dunder access (blocks `().__class__.__subclasses__()` style escapes), no
   pickle/SQL/HTML readers, no absolute paths, '..' or URLs in string literals.

2. Isolated runtime (`run_code`) - the code runs in a *separate* Python
   process (`python -I`, so no user site-packages / env-based imports) with:
     * a fresh temp directory as cwd, holding *copies* of the data files,
     * a wall-clock timeout (process killed) and CPU / memory / file-size rlimits,
     * networking disabled (socket functions replaced before user code runs),
     * a scrubbed environment: no API keys or other secrets are inherited.
   Matplotlib figures left open are saved as PNGs and handed back.

This is defence in depth for a single-user analysis tool, not a hardened
multi-tenant jail - see DECISIONS.md for what a production deployment would add
(containers / gVisor / seccomp).
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "math", "statistics", "datetime", "re", "json",
    "collections", "itertools", "functools", "string", "textwrap", "warnings",
    "decimal", "fractions", "calendar", "operator", "typing", "dataclasses",
}
BANNED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit", "help", "memoryview",
}
BANNED_ATTRS = {
    "read_pickle", "to_pickle", "read_sql", "read_sql_query", "read_sql_table", "to_sql",
    "read_html", "read_xml", "read_clipboard", "to_clipboard", "system", "popen", "load",
    "fromfile", "tofile", "memmap", "f_globals", "gi_frame",
}

MAX_OUTPUT_CHARS = 20_000

RUNNER = r'''
import sys, io, traceback, warnings
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
    resource.setrlimit(resource.RLIMIT_FSIZE, ({fsize}, {fsize}))
except Exception:
    pass

import socket
def _no_network(*a, **k):
    raise PermissionError("network access is disabled in the sandbox")
class _NoSocket(socket.socket):  # stays a class: some libraries subclass / isinstance-check it
    def __init__(self, *a, **k):
        _no_network()
socket.socket = _NoSocket
socket.create_connection = _no_network
socket.getaddrinfo = _no_network

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
warnings.filterwarnings("ignore")

src = open("analysis.py", encoding="utf-8").read()
status = 0
try:
    exec(compile(src, "analysis.py", "exec"), {{"__name__": "__main__"}})
except SystemExit:
    pass
except BaseException:
    tb = traceback.format_exc()
    # Keep only frames from the user's script: library internals just add noise.
    lines = tb.splitlines()
    keep, skip = [], False
    for ln in lines:
        if ln.startswith('  File "'):
            skip = 'File "analysis.py"' not in ln
        elif not ln.startswith("  "):
            skip = False  # "Traceback ..." header / the exception message itself
        if not skip:
            keep.append(ln)
    print("\n".join(keep), file=sys.stderr)
    status = 1
finally:
    for i, num in enumerate(plt.get_fignums(), 1):
        try:
            plt.figure(num).savefig(f"_figure_{{i}}.png", dpi=110, bbox_inches="tight")
        except Exception as e:
            print(f"[sandbox] could not save figure {{i}}: {{e}}", file=sys.stderr)
sys.stdout.flush()
sys.exit(status)
'''


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""                 # one-line summary of what went wrong
    figures: list[Path] = field(default_factory=list)
    duration_s: float = 0.0
    timed_out: bool = False
    violations: list[str] = field(default_factory=list)

    def for_llm(self) -> str:
        """Compact description of the outcome, suitable for a prompt."""
        if self.violations:
            return "Code was REJECTED by the safety check:\n- " + "\n- ".join(self.violations)
        parts = [f"Execution {'succeeded' if self.ok else 'FAILED'} in {self.duration_s:.1f}s."]
        if self.timed_out:
            parts.append(self.error)
        if self.stdout.strip():
            parts.append("STDOUT:\n" + _truncate(self.stdout, 8000))
        if not self.ok and self.stderr.strip():
            parts.append("ERROR:\n" + _truncate(self.stderr, 4000))
        if self.figures:
            parts.append(f"Charts saved: {', '.join(p.name for p in self.figures)}")
        return "\n\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters truncated] ...\n" + text[-half:]


# ------------------------------------------------------------------ static check
def check_code(code: str) -> list[str]:
    """Return a list of safety violations (empty list = code may run)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    problems: list[str] = []

    def bad(node: ast.AST, msg: str) -> None:
        problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # Literal pieces in the middle of an f-string (f"{x}/month") cannot start a path or URL.
    inner_fstring_parts = {
        id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for v in n.values[1:]
        if isinstance(v, ast.Constant)
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    bad(node, f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                bad(node, f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_CALLS:
                bad(node, f"use of '{node.id}' is not allowed")
            elif node.id.startswith("__") and node.id != "__name__":
                bad(node, f"access to '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                bad(node, f"dunder attribute '{node.attr}' is not allowed")
            elif node.attr in BANNED_ATTRS:
                bad(node, f"'.{node.attr}' is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inner_fstring_parts:
            v = node.value.strip()
            if _ABS_PATH.match(v) or "../" in v or "..\\" in v:
                bad(node, f"file paths outside the working directory are not allowed: {v[:60]!r}")
            elif "://" in v:
                bad(node, f"URLs are not allowed (no network): {v[:60]!r}")
    return problems


# "/etc/x", "~/x", "C:\\x" - but not a lone "/" used as a separator or in a label.
_ABS_PATH = re.compile(r"^(/[\w.\-]|~[/\\]|[A-Za-z]:[/\\])")


# ---------------------------------------------------------------- execution
def _clean_env(workdir: Path) -> dict[str, str]:
    """Minimal environment: nothing from the parent (no API keys) except PATH."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(workdir / ".mpl"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
    }


def run_code(
    code: str,
    data_files: list[Path] | None = None,
    timeout: int = 60,
    figures_dir: Path | None = None,
    figure_prefix: str = "chart",
    memory_mb: int = 2048,
) -> ExecResult:
    """
    Check and run `code` in an isolated subprocess.

    Data files are copied into the temp working directory, so the code should
    open them by bare file name (e.g. `pd.read_csv("sales.csv")`). Figures are
    moved to `figures_dir` (if given) as <figure_prefix>_<n>.png.
    """
    violations = check_code(code)
    if violations:
        return ExecResult(ok=False, error="code rejected by safety check", violations=violations)

    with tempfile.TemporaryDirectory(prefix="analyst_sandbox_") as tmp:
        workdir = Path(tmp)
        for f in data_files or []:
            shutil.copy2(f, workdir / Path(f).name)
        (workdir / "analysis.py").write_text(code, encoding="utf-8")
        (workdir / "_runner.py").write_text(
            RUNNER.format(cpu=timeout + 5, mem=memory_mb * 1024 * 1024, fsize=200 * 1024 * 1024),
            encoding="utf-8",
        )

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "_runner.py"],
                cwd=workdir,
                env=_clean_env(workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(
                ok=False,
                stdout=_truncate(out, MAX_OUTPUT_CHARS),
                error=f"Timed out after {timeout}s - the code is too slow (avoid row-by-row Python loops).",
                duration_s=round(time.time() - start, 2),
                timed_out=True,
            )
        duration = round(time.time() - start, 2)

        figures: list[Path] = []
        pngs = sorted(workdir.glob("_figure_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        # Also keep charts the code saved itself (then closed) under its own names.
        pngs += sorted(p for p in workdir.glob("*.png") if not p.name.startswith("_figure_"))
        if figures_dir is not None and pngs:
            figures_dir.mkdir(parents=True, exist_ok=True)
            for i, png in enumerate(pngs, 1):
                dest = figures_dir / f"{figure_prefix}_{i}.png"
                shutil.move(str(png), dest)
                figures.append(dest)

        stderr = proc.stderr.strip()
        ok = proc.returncode == 0
        error = ""
        if not ok:
            last = [ln for ln in stderr.splitlines() if ln.strip()]
            error = last[-1] if last else f"process exited with code {proc.returncode}"
            if proc.returncode < 0:
                error = f"process killed by signal {-proc.returncode} (memory/CPU limit?)"
        return ExecResult(
            ok=ok,
            stdout=_truncate(proc.stdout, MAX_OUTPUT_CHARS),
            stderr=_truncate(stderr, MAX_OUTPUT_CHARS),
            error=error,
            figures=figures,
            duration_s=duration,
        )
