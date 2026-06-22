import cv2
import numpy as np

from src.perception.sim_gate_detector import SimGateDetector


def test_orange_refinement_tightens_loose_prior_box():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    orange = (35, 120, 230)  # BGR
    cv2.rectangle(img, (130, 80), (190, 140), orange, 6)

    detector = SimGateDetector(backend="color", min_confidence=0.05)
    loose = detector._pack_detection(
        320,
        240,
        95,
        45,
        130,
        130,
        0.8,
        "gatenet",
    )

    refined = detector._refine_orange_box(img, loose)

    assert refined is not None
    x, y, w, h = refined["bbox"]
    assert 120 <= x <= 135
    assert 70 <= y <= 90
    assert 55 <= w <= 75
    assert 55 <= h <= 75
    assert refined["source"] == "gatenet+orange"
