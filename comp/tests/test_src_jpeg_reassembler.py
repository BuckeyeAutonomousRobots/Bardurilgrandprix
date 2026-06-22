import struct

from src.vision.jpeg_reassembler import HEADER_FORMAT, JpegReassembler


def test_jpeg_reassembler_returns_complete_frame():
    r = JpegReassembler()
    payload = b"abcdefgh"
    pkt0 = struct.pack(HEADER_FORMAT, 5, 0, 2, len(payload), 4, 123) + payload[:4]
    pkt1 = struct.pack(HEADER_FORMAT, 5, 1, 2, len(payload), 4, 123) + payload[4:]
    assert r.add_packet(pkt0, now_s=1.0) is None
    frame = r.add_packet(pkt1, now_s=1.01)
    assert frame is not None
    assert frame.frame_id == 5
    assert frame.jpeg_bytes == payload
