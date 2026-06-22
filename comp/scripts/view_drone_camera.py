"""Live viewer for the simulator UDP camera feed (port 5600 by default).

Use this when FlightSim is running but the autonomy stack is NOT — only one
process can bind to the vision UDP port at a time.

While the stack is running, prefer:
  python -m src.main --show-vision
or:
  .\\run_sim_stack.ps1 -ShowVision
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.vision.jpeg_reassembler import JpegReassembler


def load_sim_vision_port(sim_config: Path) -> tuple[str, int]:
    cfg = yaml.safe_load(sim_config.read_text(encoding="utf-8")) or {}
    return str(cfg.get("vision_host", "0.0.0.0")), int(cfg.get("vision_port", 5600))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="View FlightSim drone camera UDP stream")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--sim-config",
        default=str(root / "config" / "sim_comp.yaml"),
        help="Used for vision_host/vision_port when --host/--port not set",
    )
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host:
        host = args.host
        port = args.port or 5600
    else:
        host, port = load_sim_vision_port(Path(args.sim_config))
        if args.port:
            port = args.port

    reassembler = JpegReassembler()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(
            f"[VIEW] Cannot bind {host}:{port} ({exc}). "
            "Stop src.main if it is running, or pick another port."
        )
        return 1
    sock.settimeout(0.2)

    window = "Drone camera (UDP)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = max(0.25, float(args.scale))
    frame_count = 0
    last_report = time.monotonic()
    print(f"[VIEW] Listening on {host}:{port}. Press q or Esc to quit.", flush=True)

    running = True
    while running:
        try:
            packet, _addr = sock.recvfrom(65536)
        except socket.timeout:
            key = cv2.waitKey(1) & 0xFF
            running = key not in (27, ord("q"))
            continue

        frame = reassembler.add_packet(packet)
        if frame is None:
            continue
        image = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue

        frame_count += 1
        now = time.monotonic()
        if now - last_report >= 1.0:
            fps = frame_count / max(now - last_report, 1e-6)
            print(f"[VIEW] frame_id={frame.frame_id} fps={fps:.1f}", flush=True)
            frame_count = 0
            last_report = now

        cv2.putText(
            image,
            f"id={frame.frame_id}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window, image)
        key = cv2.waitKey(1) & 0xFF
        running = key not in (27, ord("q"))

    sock.close()
    cv2.destroyAllWindows()
    print("[VIEW] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
