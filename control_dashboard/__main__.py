"""Run the Reachy Mini AI control dashboard."""

from __future__ import annotations
import logging

from control_dashboard.server import serve


def main() -> None:
    """Start the local control dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    serve()


if __name__ == "__main__":
    main()
