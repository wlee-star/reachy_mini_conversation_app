"""Build a patched wheel while preserving base-wheel archive metadata."""

import io
import csv
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


def _record_digest(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def main() -> None:
    """Build the requested metadata-preserving candidate wheel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_wheel, "r") as source:
        infos = source.infolist()
        contents = {info.filename: source.read(info.filename) for info in infos}

    for wheel_path, source_path in PATCHED_FILES.items():
        if wheel_path not in contents:
            raise ValueError(f"Base wheel is missing {wheel_path}")
        contents[wheel_path] = (args.source_root / source_path).read_bytes()

    record_paths = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        raise ValueError(f"Expected one RECORD entry, found {record_paths}")
    record_path = record_paths[0]

    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for info in infos:
        if info.filename == record_path:
            continue
        content = contents[info.filename]
        writer.writerow((info.filename, _record_digest(content), len(content)))
    writer.writerow((record_path, "", ""))
    contents[record_path] = record_buffer.getvalue().encode("utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as destination:
        for info in infos:
            destination.writestr(info, contents[info.filename])


if __name__ == "__main__":
    main()
