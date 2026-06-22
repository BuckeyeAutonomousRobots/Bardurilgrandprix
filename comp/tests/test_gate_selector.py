from src.perception.gate_selector import score_gate_detection, select_gate_detection
from src.types import GateDetection


def _det(center_x: float, area: float, conf: float = 0.9) -> GateDetection:
    bw = int(640 * (area**0.5))
    bh = bw
    return GateDetection(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(center_x, 180.0),
        bbox=(int(center_x - bw / 2), 90, bw, bh),
        corners_px=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        confidence=conf,
        area_fraction=area,
    )


def test_select_prefers_centered_gate_for_first_gate():
    near = _det(320.0, 0.18)
    off = _det(120.0, 0.22)
    chosen = select_gate_detection(
        [off, near],
        image_w=640,
        image_h=360,
        aim_cx=320.0,
        aim_cy=180.0,
        target_gate_index=0,
    )
    assert chosen is near


def test_select_prefers_smaller_gate_when_seeking_next():
    passed = _det(320.0, 0.50)
    nxt = _det(300.0, 0.03)
    chosen = select_gate_detection(
        [passed, nxt],
        image_w=640,
        image_h=360,
        aim_cx=320.0,
        aim_cy=180.0,
        target_gate_index=1,
        seeking_next_gate=True,
        max_acquire_area=0.42,
    )
    assert chosen is nxt


def test_score_penalizes_off_center():
    centered = score_gate_detection(
        _det(320.0, 0.10),
        image_w=640,
        image_h=360,
        aim_cx=320.0,
        aim_cy=180.0,
        seeking_next_gate=False,
        max_acquire_area=0.45,
        min_next_gate_area=0.006,
    )
    off = score_gate_detection(
        _det(80.0, 0.10),
        image_w=640,
        image_h=360,
        aim_cx=320.0,
        aim_cy=180.0,
        seeking_next_gate=False,
        max_acquire_area=0.45,
        min_next_gate_area=0.006,
    )
    assert centered > off
