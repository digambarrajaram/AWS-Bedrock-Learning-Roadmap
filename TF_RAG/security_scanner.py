import json
import os
import platform
import subprocess
import sys
import tempfile
import re
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
    """
    try:
        result = subprocess.run(
            argv + ["--version"],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        combined = (result.stdout or "") + (result.stderr or "")

        if "ModuleNotFoundError" in combined or "ImportError" in combined:
            return False

        return bool(re.search(r"\d+\.\d+", combined))

    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _wrap_cmd(path: Path) -> list[str]:
    """
    Returns the correct argv prefix to invoke *path*.
    """
    if _IS_WINDOWS and path.suffix.lower() in (".cmd", ".bat"):
        return ["cmd.exe", "/c", str(path)]
    return [str(path)]


def _find_checkov_cmd() -> list[str] | None:
    """
    Returns the argv prefix needed to invoke Checkov, trying strategies
    in order of reliability.
    """
    # ── Strategy 1: current interpreter ─────────────────────────────────────
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

    venv_scripts = Path(sys.executable).parent   # …\Scripts or …/bin
    for name in ("checkov.cmd", "checkov.exe", "checkov"):
        search_paths.append(venv_scripts / name)

    local_bin = Path.home() / ".local" / "bin"
    search_paths += [
        local_bin / "checkov.cmd",
        local_bin / "checkov.exe",
        local_bin / "checkov",
    ]

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

    print("   ❌ Checkov discovery failed. Locations tried:")
    print(f"      • {sys.executable} -m checkov")
    if checkov_on_path:
        print(f"      • {checkov_on_path} (PATH, but probe failed)")
    for p in search_paths:
        if p.exists():
            print(f"      • {p} (exists but probe failed)")
    return None


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
    tmp_dir = tempfile.mkdtemp(prefix="checkov_scan_")
    tmp_tf  = os.path.join(tmp_dir, "main.tf")
    try:
        with open(tmp_tf, "w") as fh:
            fh.write(tf_code)

        # Build dynamic execution parameters using discovered path engine
        cmd_args = checkov_cmd + [
            "-d", tmp_dir,                     # Scan the clean directory mode
            "--output", "json", 
            "--no-guide",                      # Suppresses the text banner headers
            "--external-modules-download", "false", # Skips registry lock constraints
            "--skip-check", "CKV_AWS_18,CKV_AWS_144"
        ]

        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,
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

def _parse_checkov_output(stdout: str, stderr: str) -> tuple[int, int, list[str]]:
    """Parses raw Checkov stdout/stderr into (passed, failed, messages)."""
    raw_output = (stdout or "").strip()
    combined_err = raw_output + (stderr or "")

    if "ModuleNotFoundError" in combined_err or "ImportError" in combined_err:
        return 0, 0, ["❌ Checkov environment is broken. Reinstall via pip/pipx."]

    if not raw_output:
        return 0, 0, ["❌ Checkov returned an empty string response."]

    # Resiliency fix: Find where the actual JSON payload starts to avoid ASCII banner parsing
    if "{" in raw_output:
        raw_output = raw_output[raw_output.find("{"):]

    try:
        data = json.loads(raw_output)
        
        # Checkov wraps single directory metrics inside a dictionary or a list of dictionaries
        if isinstance(data, list):
            summary = data[0].get("summary", {})
            results = data[0].get("results", {}).get("failed_checks", [])
        else:
            summary = data.get("summary", {})
            results = data.get("results", {}).get("failed_checks", [])

        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        
        violation_messages = []
        for check in results:
            check_id = check.get("check_id")
            check_name = check.get("check_name")
            file_path = check.get("file_path")
            lines = check.get("file_line_range")
            violation_messages.append(f"[{check_id}] {check_name} in {file_path} lines {lines}")

        return passed, failed, violation_messages

    except json.JSONDecodeError:
        # Fallback if text data could not be parsed as pure JSON structures
        return 0, 0, [f"Could not parse Checkov output as JSON. Raw data: {stdout[:200]}"]
