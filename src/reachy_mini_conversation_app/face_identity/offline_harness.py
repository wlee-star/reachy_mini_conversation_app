"""Offline face-identity harness for local JPEG/PNG enrolment and recognition tests."""

from __future__ import annotations
import sys
import json
import logging
import argparse
from pathlib import Path


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for offline face-memory checks."""
    parser = argparse.ArgumentParser(description="Offline YuNet + SFace face-memory harness")
    parser.add_argument("--data-dir", type=Path, required=True, help="Writable directory for identity/profile stores")
    parser.add_argument("--download-models", action="store_true", help="Allow one-time model download into the cache")
    sub = parser.add_subparsers(dest="command", required=True)

    enrol = sub.add_parser("enrol", help="Enrol a person from several images")
    enrol.add_argument("--name", required=True)
    enrol.add_argument("--details", default=None)
    enrol.add_argument("images", nargs="+", type=Path)

    recognize = sub.add_parser("recognize", help="Recognize one image against the gallery")
    recognize.add_argument("image", type=Path)

    sub.add_parser("list", help="List remembered people")

    forget = sub.add_parser("forget", help="Forget a person by name")
    forget.add_argument("--name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the offline harness."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    from reachy_mini_conversation_app.face_identity.models import ensure_models
    from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline

    try:
        ensure_models(allow_download=bool(args.download_models))
    except Exception as exc:
        logger.error("Model setup failed: %s", exc)
        return 2

    pipeline = FaceIdentityPipeline(args.data_dir, allow_download=False)
    if args.command == "enrol":
        result = pipeline.enrol_from_images(args.images, name=args.name, details=args.details)
    elif args.command == "recognize":
        result = pipeline.recognize_image(args.image)
    elif args.command == "list":
        result = {"people": pipeline.list_remembered()}
    elif args.command == "forget":
        result = pipeline.forget_person(name=args.name)
    else:
        logger.error("Unknown command: %s", args.command)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"rejected"} else 1


if __name__ == "__main__":
    sys.exit(main())
