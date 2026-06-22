from src.estimation.vision_depth import estimate_gate_depth_m
from src.types import GateTrack


def test_estimate_gate_depth_from_bbox():
    track = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 180.0),
        bbox=(200, 100, 100, 100),
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )
    camera = {"fx": 320.0, "fy": 320.0, "gate_inner_size_m": 1.5, "width": 640, "height": 360}
    depth_m, rx, ry, conf = estimate_gate_depth_m(track, camera)
    assert 3.0 < depth_m < 10.0
    assert rx > 0.0 and ry > 0.0
    assert conf > 0.2
