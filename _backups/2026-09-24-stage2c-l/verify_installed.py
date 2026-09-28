"""Verify the installed Stage 2C-L daemon candidate without changing it."""

import argparse
import hashlib
import json
import stat
import subprocess
from pathlib import Path

import reachy_mini
from reachy_mini.io.protocol import DaemonStatus


def main() -> None:
    """Run the installed-file and authoritative protocol pre-start gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    site_packages = Path(reachy_mini.__file__).resolve().parent.parent
    mismatches: list[str] = []
    checked: list[dict[str, str]] = []

    for expected in manifest["production_files"]:
        relative_path = expected["file"]
        installed_path = site_packages / relative_path
        if not installed_path.is_file():
            mismatches.append(f"missing:{relative_path}")
            continue
        actual_hash = hashlib.sha256(installed_path.read_bytes()).hexdigest()
        actual_mode = f"{stat.S_IMODE(installed_path.stat().st_mode):04o}"
        checked.append({"file": relative_path, "sha256": actual_hash, "mode": actual_mode})
        if actual_hash != expected["sha256"]:
            mismatches.append(f"hash:{relative_path}:{actual_hash}")
        if actual_mode != expected["expected_mode"]:
            mismatches.append(f"mode:{relative_path}:{actual_mode}")

    launcher = site_packages / "reachy_mini/daemon/app/services/wireless/launcher.sh"
    launcher_bytes = launcher.read_bytes()
    launcher_stat = launcher.stat()
    launcher_mode = stat.S_IMODE(launcher_stat.st_mode)
    syntax = subprocess.run(["/bin/bash", "-n", str(launcher)], check=False, capture_output=True, text=True)

    base_status = {
        "type": "daemon_status",
        "robot_name": "reachy_mini",
        "state": "running",
        "wireless_version": True,
        "desktop_app_daemon": False,
        "simulation_enabled": False,
        "mockup_sim_enabled": False,
        "backend_status": None,
    }
    missing = DaemonStatus.model_validate(base_status)
    enabled = DaemonStatus.model_validate(base_status | {"head_tracking_enabled": True})
    disabled = DaemonStatus.model_validate(base_status | {"head_tracking_enabled": False})

    abstract_source = (site_packages / "reachy_mini/daemon/backend/abstract.py").read_text(encoding="utf-8")
    daemon_source = (site_packages / "reachy_mini/daemon/daemon.py").read_text(encoding="utf-8")
    result = {
        "package_version": reachy_mini.__version__,
        "package_path": str(Path(reachy_mini.__file__).resolve()),
        "checked_files": checked,
        "mismatches": mismatches,
        "launcher_path": str(launcher),
        "launcher_regular": stat.S_ISREG(launcher_stat.st_mode),
        "launcher_mode": f"{launcher_mode:04o}",
        "owner_executable": bool(launcher_mode & stat.S_IXUSR),
        "group_executable": bool(launcher_mode & stat.S_IXGRP),
        "other_executable": bool(launcher_mode & stat.S_IXOTH),
        "launcher_shebang": launcher_bytes.splitlines()[0].decode("utf-8", "replace"),
        "launcher_lf_only": b"\r" not in launcher_bytes,
        "bash_syntax_exit_code": syntax.returncode,
        "bash_syntax_stderr": syntax.stderr,
        "protocol_field_present": "head_tracking_enabled" in DaemonStatus.model_fields,
        "protocol_missing_is_none": missing.head_tracking_enabled is None,
        "protocol_true": enabled.head_tracking_enabled is True,
        "protocol_false": disabled.head_tracking_enabled is False,
        "backend_accessor_present": "def is_head_tracking_enabled(self) -> bool:" in abstract_source,
        "backend_accessor_locked": "with self._tracking_lock:\n            return self._tracking_enabled" in abstract_source,
        "daemon_publication_present": (
            "self._status.head_tracking_enabled = self.backend.is_head_tracking_enabled()" in daemon_source
        ),
        "daemon_none_fallback_present": "self._status.head_tracking_enabled = None" in daemon_source,
    }
    print(json.dumps(result, indent=2))

    required = (
        not mismatches
        and result["launcher_regular"]
        and result["launcher_mode"] == "0755"
        and result["owner_executable"]
        and result["group_executable"]
        and result["other_executable"]
        and result["launcher_shebang"] == "#!/bin/bash"
        and result["launcher_lf_only"]
        and result["bash_syntax_exit_code"] == 0
        and result["protocol_field_present"]
        and result["protocol_missing_is_none"]
        and result["protocol_true"]
        and result["protocol_false"]
        and result["backend_accessor_present"]
        and result["backend_accessor_locked"]
        and result["daemon_publication_present"]
        and result["daemon_none_fallback_present"]
    )
    if not required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
