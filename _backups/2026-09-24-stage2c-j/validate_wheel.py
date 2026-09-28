"""Validate Stage 2C-J wheel contents and archive metadata."""

import io
import csv
import json
import stat
import base64
import hashlib
import zipfile
import argparse
from pathlib import Path


PATCHED_FILES = {
    "reachy_mini/io/protocol.py": "src/reachy_mini/io/protocol.py",
    "reachy_mini/daemon/backend/abstract.py": "src/reachy_mini/daemon/backend/abstract.py",
    "reachy_mini/daemon/daemon.py": "src/reachy_mini/daemon/daemon.py",
}


def _digest(content: bytes) -> str:
    value = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return "sha256=" + value.rstrip(b"=").decode("ascii")


def _mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0o7777


def main() -> None:
    """Validate a candidate wheel and write its deployment manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_wheel) as base, zipfile.ZipFile(args.candidate_wheel) as candidate:
        base_infos = {info.filename: info for info in base.infolist()}
        candidate_infos = {info.filename: info for info in candidate.infolist()}
        if base_infos.keys() != candidate_infos.keys():
            raise ValueError("Candidate archive paths differ from the official wheel")

        record_paths = [name for name in candidate_infos if name.endswith(".dist-info/RECORD")]
        if len(record_paths) != 1:
            raise ValueError(f"Expected one RECORD entry, found {record_paths}")
        record_path = record_paths[0]

        content_differences: list[str] = []
        metadata_differences: list[str] = []
        executable_manifest: list[dict[str, object]] = []
        symlinks: list[str] = []

        for name, base_info in base_infos.items():
            candidate_info = candidate_infos[name]
            if base.read(name) != candidate.read(name):
                content_differences.append(name)
            metadata_fields = (
                "create_system",
                "external_attr",
                "internal_attr",
                "flag_bits",
                "date_time",
                "compress_type",
            )
            if any(getattr(base_info, field) != getattr(candidate_info, field) for field in metadata_fields):
                metadata_differences.append(name)
            mode = _mode(candidate_info)
            if mode & 0o111 and not candidate_info.is_dir():
                content = candidate.read(name)
                executable_manifest.append(
                    {
                        "file": name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "mode": f"{mode:04o}",
                        "shebang": content.splitlines()[0].decode("utf-8", "replace") if content else "",
                        "lf_only": b"\r" not in content,
                    }
                )
            file_type = (candidate_info.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                symlinks.append(name)

        allowed_differences = set(PATCHED_FILES) | {record_path}
        if set(content_differences) != allowed_differences:
            raise ValueError(f"Unexpected content differences: {content_differences}")
        if metadata_differences:
            raise ValueError(f"Archive metadata changed: {metadata_differences}")

        for wheel_path, source_path in PATCHED_FILES.items():
            if candidate.read(wheel_path) != (args.source_root / source_path).read_bytes():
                raise ValueError(f"Candidate content does not match source: {wheel_path}")

        rows = list(csv.reader(io.StringIO(candidate.read(record_path).decode("utf-8"))))
        records = {row[0]: row[1:] for row in rows}
        if records.keys() != candidate_infos.keys():
            raise ValueError("RECORD paths do not match archive paths")
        for name in candidate_infos:
            if name == record_path:
                if records[name] != ["", ""]:
                    raise ValueError("RECORD must not hash itself")
                continue
            content = candidate.read(name)
            if records[name] != [_digest(content), str(len(content))]:
                raise ValueError(f"Invalid RECORD entry: {name}")

        production_manifest: list[dict[str, str]] = []
        manifest_paths = set(PATCHED_FILES)
        manifest_paths.update(item["file"] for item in executable_manifest)
        for name in sorted(manifest_paths):
            content = candidate.read(name)
            production_manifest.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "expected_mode": f"{_mode(candidate_infos[name]):04o}",
                }
            )

    manifest = {
        "base_wheel": str(args.base_wheel.resolve()),
        "candidate_wheel": str(args.candidate_wheel.resolve()),
        "candidate_sha256": hashlib.sha256(args.candidate_wheel.read_bytes()).hexdigest(),
        "content_differences": content_differences,
        "metadata_differences": metadata_differences,
        "symlinks": symlinks,
        "executable_files": executable_manifest,
        "production_files": production_manifest,
        "record_valid": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
