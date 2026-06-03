import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Checkov discovery
# ---------------------------------------------------------------------------

def _probe(argv: list[str]) -> bool:
    """
    Returns True if *argv* invokes a working Checkov.

    Checkov's --version exits with code 1 on some builds, so we do NOT rely
    on returncode.  Instead we check that the output contains a version
    string and no fatal import error.

    On Windows, .cmd files cannot be passed to CreateProcess directly
    (WinError 193).  Callers must wrap them via _wrap_cmd() before passing
    here so they arrive as ["cmd.exe", "/c", "<path>"].
    """
    try:
        result = subprocess.run(
            argv + ["--version"],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        combined = (result.stdout or "") + (result.stderr or "")

        # A broken install always prints ModuleNotFoundError / ImportError.
        if "ModuleNotFoundError" in combined or "ImportError" in combined:
            return False

        # A working install prints something like "checkov 3.x.y"
        # Accept any output that contains a digit (version number).
        import re
        return bool(re.search(r"\d+\.\d+", combined))

    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _wrap_cmd(path: Path) -> list[str]:
    """
    Returns the correct argv prefix to invoke *path*.

    - .cmd / .bat  →  ["cmd.exe", "/c", str(path)]   (Windows shell scripts)
    - anything else →  [str(path)]
    """
    if _IS_WINDOWS and path.suffix.lower() in (".cmd", ".bat"):
        return ["cmd.exe", "/c", str(path)]
    return [str(path)]


def _find_checkov_cmd() -> list[str] | None:
    """
    Returns the argv prefix needed to invoke Checkov, trying strategies
    in order of reliability:

      1. python -m checkov   – same interpreter as this script; works when
                               checkov is installed in the active venv.
      2. checkov on PATH     – works for clean global / venv installs.
      3. Explicit file hunt  – searches .venv, pipx, and local-bin locations.
                               .cmd files are wrapped in ``cmd.exe /c``.

    Returns None when no working Checkov can be found.
    """
    # ── Strategy 1: current interpreter ─────────────────────────────────────
    # This is the most reliable option when checkov is pip-installed into the
    # same venv that is running this script (e.g. D:\..\.venv).
    candidate = [sys.executable, "-m", "checkov"]
    if _probe(candidate):
        print(f"   🔧 Checkov found via: {' '.join(candidate)}")
        return candidate

    # ── Strategy 2: PATH lookup ──────────────────────────────────────────────
    import shutil
    checkov_on_path = shutil.which("checkov")
    if checkov_on_path:
        wrapped = _wrap_cmd(Path(checkov_on_path))
        if _probe(wrapped):
            print(f"   🔧 Checkov found on PATH: {checkov_on_path}")
            return wrapped

    # ── Strategy 3: explicit filesystem search ───────────────────────────────
    search_paths: list[Path] = []

    # The active venv (most common case for project-local installs).
    # sys.executable is e.g. D:\project\.venv\Scripts\python.exe
    venv_scripts = Path(sys.executable).parent   # …\Scripts or …/bin
    for name in ("checkov.cmd", "checkov.exe", "checkov"):
        search_paths.append(venv_scripts / name)

    # %USERPROFILE%\.local\bin  (pipx default on Windows)
    local_bin = Path.home() / ".local" / "bin"
    search_paths += [
        local_bin / "checkov.cmd",
        local_bin / "checkov.exe",
        local_bin / "checkov",
    ]

    # pipx venvs – Scripts\ (Windows) and bin/ (Linux/macOS)
    pipx_home = Path(os.environ.get("PIPX_HOME", Path.home() / ".local" / "pipx"))
    for sub in ("Scripts", "bin"):
        venv_dir = pipx_home / "venvs" / "checkov" / sub
        search_paths += [
            venv_dir / "checkov.cmd",
            venv_dir / "checkov.exe",
            venv_dir / "checkov",
        ]

    for path in search_paths:
        if not path.exists():
            continue
        wrapped = _wrap_cmd(path)
        if _probe(wrapped):
            print(f"   🔧 Checkov found at: {path}")
            return wrapped

    # ── Nothing worked – print what we tried to help the user debug ──────────
    print("   ❌ Checkov discovery failed. Locations tried:")
    print(f"      • {sys.executable} -m checkov")
    if checkov_on_path:
        print(f"      • {checkov_on_path} (PATH, but probe failed)")
    for p in search_paths:
        if p.exists():
            print(f"      • {p} (exists but probe failed)")
    return None


# Cache the result so we only probe once per process.
_CHECKOV_CMD: list[str] | None | bool = False   # False = not yet resolved


def _get_checkov_cmd() -> list[str] | None:
    global _CHECKOV_CMD
    if _CHECKOV_CMD is False:
        _CHECKOV_CMD = _find_checkov_cmd()
    return _CHECKOV_CMD  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan_with_checkov(tf_code: str) -> tuple[int, int, list[str]]:
    """
    Runs a Checkov scan against *tf_code* and returns
    ``(passed: int, failed: int, violation_messages: list[str])``.

    Uses a temporary DIRECTORY (not --file) because Checkov 3.x has a
    known internal runner crash when scanning a single .tf file that
    references external modules without a prior ``terraform init``.
    Directory mode + ``--download-external-modules false`` avoids this.

    Safe to call even when Checkov is not installed – returns
    ``(0, 0, [descriptive error])`` so the caller can decide what to do.
    """
    checkov_cmd = _get_checkov_cmd()
    if checkov_cmd is None:
        return 0, 0, [
            "❌ Checkov is not available on this system.\n"
            "   Fix options (choose one):\n"
            "     • pip install checkov                    # install into current venv\n"
            "     • pipx reinstall checkov                 # repair broken pipx install\n"
            "     • pipx install checkov --force           # force-reinstall via pipx\n"
            "   Then restart this script."
        ]

    import shutil as _shutil
    # Use a temp DIRECTORY so Checkov uses directory-scan mode.
    # --file mode crashes in Checkov 3.x when the .tf uses external modules.
    tmp_dir = tempfile.mkdtemp(prefix="checkov_scan_")
    tmp_tf  = os.path.join(tmp_dir, "main.tf")
    try:
        with open(tmp_tf, "w") as fh:
            fh.write(tf_code)

            result = subprocess.run(
                checkov_cmd + [
                    "--directory", tmp_dir,
                    "--framework", "terraform",
                    "--output", "json",
                    "--skip-check", "CKV_AWS_18,CKV_AWS_144",
                    # Prevent Checkov from trying to download / init external
                    # modules (terraform-aws-modules/*).  That download attempt
                    # is exactly what triggers the runner crash in 3.x.
                    "--download-external-modules", "false",
                    "--compact",
                ],
                capture_output=True,
                text=True,
                shell=False,
                timeout=120,
            )
    except subprocess.TimeoutExpired:
        return 0, 0, ["Checkov timed out after 120 s."]
    except OSError as exc:
        return 0, 0, [f"Failed to launch Checkov process: {exc}"]
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)

    return _parse_checkov_output(result.stdout, result.stderr)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_checkov_output(
    stdout: str, stderr: str
) -> tuple[int, int, list[str]]:
    """Parses raw Checkov stdout/stderr into (passed, failed, messages)."""

    raw = stdout.strip()

    # ── Detect broken entry-point before trying to parse JSON ──────────────
    combined_err = (stdout or "") + (stderr or "")
    if "ModuleNotFoundError" in combined_err:
        return 0, 0, [
            "❌ Checkov entry-point is broken (ModuleNotFoundError).\n"
            "   The checkov command exists on PATH but its Python module is missing.\n"
            "   Fix: pipx reinstall checkov   OR   pip install checkov"
        ]

    if not raw:
        fallback = stderr.strip()
        return 0, 0, [
            f"Checkov produced no JSON output.\n"
            f"stderr: {fallback[:300] if fallback else '(empty)'}"
        ]

    # ── Try to parse JSON (Checkov may prefix human-readable text) ──────────
    scan_result = _extract_json(raw)
    if scan_result is None:
        return 0, 0, [
            f"Could not parse Checkov output as JSON.\n"
            f"Raw (first 500 chars):\n{raw[:500]}"
        ]

    # ── Aggregate results across frameworks ─────────────────────────────────
    passed, failed, error_messages = 0, 0, []
    reports = scan_result if isinstance(scan_result, list) else [scan_result]

    for report in reports:
        if not isinstance(report, dict):
            continue

        summary = report.get("summary") or {}
        passed += summary.get("passed", 0)
        failed += summary.get("failed", 0)

        failed_checks = (report.get("results") or {}).get("failed_checks") or []
        for check in failed_checks:
            check_id   = check.get("check_id", "UNKNOWN")
            check_name = check.get("check_name", "Unknown check")
            resource   = check.get("resource", "")
            lines      = check.get("file_line_range", ["?", "?"])
            error_messages.append(
                f"- [{check_id}] {check_name} "
                f"| resource: {resource} "
                f"| lines: {lines[0]}-{lines[1]}"
            )

    return passed, failed, error_messages


def _extract_json(raw: str) -> dict | list | None:
    """
    Tries to extract a JSON value from *raw*, handling the case where
    Checkov prints human-readable text before the JSON block.
    """
    # Fast path: the whole string is valid JSON.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Slow path: scan for the first '[' or '{' and try from there.
    for start_char in ("[", "{"):
        idx = raw.find(start_char)
        if idx != -1:
            try:
                return json.loads(raw[idx:])
            except json.JSONDecodeError:
                continue

    return None