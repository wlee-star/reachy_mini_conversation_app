"""Start the installed conversation app through the physical robot daemon."""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--daemon-port", type=int, default=8000)
    parser.add_argument("--app-port", type=int, default=7860)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    url = (
        f"http://{args.host}:{args.daemon_port}"
        "/api/apps/start-app/reachy_mini_conversation_app"
    )
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            print(response.read().decode("utf-8", errors="replace"), flush=True)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Conversation app start request failed: {exc}", file=sys.stderr, flush=True)
        return 1

    status_url = f"http://{args.host}:{args.daemon_port}/api/apps/current-app-status"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=3.0) as response:
                body = response.read().decode("utf-8", errors="replace")
                if '"state":"running"' in body and '"name":"reachy_mini_conversation_app"' in body:
                    print("Conversation app is running under the Reachy daemon", flush=True)
                    return 0
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
            continue
        time.sleep(1.0)
    print("Conversation app did not reach running state before timeout", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
