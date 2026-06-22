from __future__ import annotations

import socket
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.types import SharedState, VisionFrame
from src.vision.jpeg_reassembler import JpegReassembler


class UdpVisionReceiver:
    def __init__(self, shared_state: SharedState, host: str, port: int, logger=None) -> None:
        self.shared_state = shared_state
        self.host = host
        self.port = port
        self.logger = logger
        self.reassembler = JpegReassembler()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._packet_count = 0
        self._frame_count = 0
        self._decode_failures = 0
        self._last_reported_stale_drops = 0
        self._last_published_key: tuple[int, int] | None = None

    def start(self) -> None:
        self._preflight_bind_check()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)
        while self._running:
            try:
                packet, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            self._packet_count += 1
            frame = self.reassembler.add_packet(packet)
            if self.reassembler.stale_drop_count > self._last_reported_stale_drops:
                dropped = self.reassembler.stale_drop_count - self._last_reported_stale_drops
                self._last_reported_stale_drops = self.reassembler.stale_drop_count
                if self.logger is not None:
                    self.logger.log("vision_drop", {"count": dropped, "reason": "stale_reassembly"})
            if frame is None:
                continue
            image = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self._decode_failures += 1
                if self.logger is not None:
                    self.logger.log("vision_drop", {"count": 1, "reason": "jpeg_decode_failed"})
                continue
            frame_key = (frame.frame_id, frame.sim_time_ns)
            if frame_key == self._last_published_key:
                continue
            self._last_published_key = frame_key
            self._frame_count += 1
            vision_frame = VisionFrame(
                frame_id=frame.frame_id,
                sim_time_ns=frame.sim_time_ns,
                wall_time_s=time.monotonic(),
                image_shape=(image.shape[0], image.shape[1]),
                image_bgr=image,
            )
            self.shared_state.update_frame(vision_frame)
            if self.logger is not None:
                self.logger.log(
                    "vision_frame",
                    {
                        "frame_id": vision_frame.frame_id,
                        "sim_time_ns": vision_frame.sim_time_ns,
                        "shape": list(vision_frame.image_shape),
                    },
                )

    def get_stats(self) -> dict[str, int]:
        return {
            "packets_rx": self._packet_count,
            "frames_rx": self._frame_count,
            "frames_dropped": self.reassembler.stale_drop_count + self._decode_failures,
            "decode_failures": self._decode_failures,
        }

    def _preflight_bind_check(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind((self.host, self.port))
        except OSError as exc:
            raise RuntimeError(
                f"Cannot bind vision UDP {self.host}:{self.port}; the port is already in use. "
                f"Close the stale pilot process and retry."
            ) from exc
        finally:
            probe.close()
