from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional


HEADER_FORMAT = "<IHHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class ReassembledFrame:
    frame_id: int
    sim_time_ns: int
    jpeg_bytes: bytes


class JpegReassembler:
    def __init__(self, max_age_s: float = 0.15) -> None:
        self.max_age_s = max_age_s
        self._frames: dict[int, dict] = {}
        self.stale_drop_count = 0

    def add_packet(self, packet: bytes, now_s: Optional[float] = None) -> Optional[ReassembledFrame]:
        now_s = time.time() if now_s is None else now_s
        self._drop_stale(now_s)
        if len(packet) < HEADER_SIZE:
            return None

        frame_id, chunk_id, total_chunks, jpeg_size, _payload_size, sim_time_ns = struct.unpack(
            HEADER_FORMAT, packet[:HEADER_SIZE]
        )
        payload = packet[HEADER_SIZE:]
        frame = self._frames.setdefault(
            frame_id,
            {
                "created_s": now_s,
                "total_chunks": total_chunks,
                "jpeg_size": jpeg_size,
                "sim_time_ns": sim_time_ns,
                "chunks": {},
            },
        )
        frame["created_s"] = now_s
        frame["chunks"][chunk_id] = payload
        if len(frame["chunks"]) != total_chunks:
            return None

        jpeg_bytes = bytearray()
        for idx in range(total_chunks):
            chunk = frame["chunks"].get(idx)
            if chunk is None:
                return None
            jpeg_bytes.extend(chunk)
        del self._frames[frame_id]
        return ReassembledFrame(frame_id=frame_id, sim_time_ns=sim_time_ns, jpeg_bytes=bytes(jpeg_bytes[:jpeg_size]))

    def _drop_stale(self, now_s: float) -> None:
        stale_ids = [
            frame_id
            for frame_id, meta in self._frames.items()
            if now_s - float(meta["created_s"]) > self.max_age_s
        ]
        for frame_id in stale_ids:
            del self._frames[frame_id]
        self.stale_drop_count += len(stale_ids)
