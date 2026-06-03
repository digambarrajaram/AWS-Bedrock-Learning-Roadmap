import json
import os
import platform
import subprocess
import sys
import tempfile
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Checkov discovery
# ---------------------------------------------------------------------------

def _probe(argv: list[str]) -> bool:
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
    if _IS_WINDOWS and path.suffix.lower() in (".cmd", ".bat"):
        return ["cmd.exe", "/c", str(path)]
    return [str(path)]

def _find_checkov_cmd() -> list[str] | None:
    candidate = [sys.executable, "-m", "checkov"]
    if _probe(candidate):
        print(f"   🔧 Checkov found via: {' '.join(candidate)}")
        return candidate

    import shutil
    checkov_on_path = shutil.which("checkov")
    if checkov_on_path:
        wrapped = _wrap_cmd(Path(checkov_on_path))
        if _probe(wrapped):
            print(f"   🔧 Checkov found on PATH: {checkov_on_path}")
            return wrapped

    search_paths: list[Path] = []
    venv_scripts = Path(sys.executable).parent
    for name in ("checkov.cmd", "checkov.exe", "checkov"):
        search_paths.append(venv_scripts / name)

    local_bin = Path.home() / ".local" / "bin"
    search_paths += [local_bin / "checkov.cmd", local_bin / "checkov.exe", local_bin / "checkov"]

    pipx_home = Path(os.environ.get("PIPX_HOME", Path.home() / ".local" / "pipx"))
    for sub in ("Scripts", "bin"):
        venv_dir = pipx_home / "venvs" / "checkov" / sub
        search_paths += [venv_dir / "checkov.cmd", venv_dir / "checkov.exe", venv_dir / "checkov"]

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

_CHECKOV_CMD: list[str] | None | bool = False

def _get_checkov_cmd() -> list[str] | None:
    global _CHECKOV_CMD
    if _CHECKOV_CMD is False:
        _CHECKOV_CMD = _find_checkov_cmd()
    return _CHECKOV_CMD

# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

def scan_with_checkov(tf_code: str) -> tuple[int, int, list[str]]:
    checkov_cmd = _get_checkov_cmd()
    if checkov_cmd is None:
        error_msg = (
            "❌ Checkov is not available on this system.\n"
            "   Fix options (choose one):\n"
            "     • pip install checkov\n"
            "     • pipx reinstall checkov\n"
            "   Then restart this script."
        )
        logger.error("Checkov not found")
        return 0, 0, [error_msg]

    import shutil as _shutil
    tmp_dir = tempfile.mkdtemp(prefix="checkov_scan_")
    tmp_tf  = os.path.join(tmp_dir, "main.tf")
    logger.debug(f"Checkov temp directory: {tmp_dir}")
    
    try:
        with open(tmp_tf, "w") as fh:
            fh.write(tf_code)
        logger.debug(f"Wrote {len(tf_code)} chars to temp Terraform file")

        cmd_args = checkov_cmd + [
            "-d", tmp_dir,
            "--output", "json",
            "--skip-check", "CKV_AWS_18,CKV_AWS_144"
        ]
        
        logger.info(f"Running Checkov: {' '.join(cmd_args[:4])}...")

        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,
        )
        
        logger.debug(f"Checkov exit code: {result.returncode}")

    except subprocess.TimeoutExpired:
        logger.error("Checkov scan timed out after 120 seconds")
        return 0, 0, ["Checkov timed out after 120 s. Terraform code may be too complex."]
    except OSError as exc:
        logger.error(f"Failed to launch Checkov process: {exc}")
        return 0, 0, [f"Failed to launch Checkov process: {exc}"]
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temp directory: {tmp_dir}")

    return _parse_checkov_output(result.stdout, result.stderr)

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_checkov_output(stdout: str, stderr: str) -> tuple[int, int, list[str]]:
    raw_output = (stdout or "").strip()
    combined_err = raw_output + (stderr or "")

    if "ModuleNotFoundError" in combined_err or "ImportError" in combined_err:
        error_msg = "❌ Checkov environment is broken. Reinstall via pip/pipx."
        logger.error(error_msg)
        return 0, 0, [error_msg]

    if not raw_output and stderr:
        error_msg = f"❌ Checkov stderr output:\n{stderr}"
        logger.error(error_msg)
        return 0, 0, [error_msg]
    elif not raw_output:
        error_msg = "❌ Checkov returned an empty string response. Check Checkov configuration."
        logger.error(error_msg)
        return 0, 0, [error_msg]

    if "{" in raw_output:
        raw_output = raw_output[raw_output.find("{"):]

    try:
        data = json.loads(raw_output)

        if isinstance(data, list):
            scan_profile = data[0] if len(data) > 0 else {}
        else:
            scan_profile = data

        summary = scan_profile.get("summary", {})
        results = scan_profile.get("results", {})
        failed_checks = results.get("failed_checks", []) if results else []

        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)

        violation_messages = []
        if failed_checks:
            for check in failed_checks:
                check_id = check.get("check_id", "UNKNOWN_CHECK")
                check_name = check.get("check_name", "No Description")
                file_path = check.get("file_path", "main.tf")
                lines = check.get("file_line_range", [0, 0])
                line_info = f"lines {lines[0]}-{lines[1]}" if isinstance(lines, list) else f"lines {lines}"
                violation_messages.append(f"[{check_id}] {check_name} in {file_path} ({line_info})")
                logger.debug(f"Checkov violation: {check_id} - {check_name}")

        if passed == 0 and failed == 0:
            passed = 1

        logger.info(f"Checkov results: {passed} passed, {failed} failed")
        return passed, failed, violation_messages

    except json.JSONDecodeError as e:
        error_msg = (
            f"Could not parse Checkov JSON output.\n"
            f"Error: {e}\n"
            f"Raw output (first 300 chars):\n{stdout[:300]}"
        )
        logger.error(error_msg)
        return 0, 0, [error_msg]
