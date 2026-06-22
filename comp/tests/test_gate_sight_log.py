from src.types import EstimatedState, GateDetection, GateTrack, RacePlan, VehicleState
from src.vision.gate_sight_log import gate_sight_record


def test_gate_sight_record_pixel_error():
    camera = {"cx": 320.0, "desired_v_px": 180.0, "width": 640, "height": 360}
    det = GateDetection(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(280.0, 220.0),
        bbox=(200, 150, 120, 120),
        corners_px=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        confidence=0.9,
        area_fraction=0.08,
    )
    track = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=det.center_px,
        bbox=det.bbox,
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(velocity_ned_mps=(1.0, 0.2, 0.0), position_ned_m=(0.0, 0.0, -1.5)),
        gate_track=track,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=-0.12,
        gate_bearing_y_rad=0.08,
        gate_range_m=6.5,
        gate_confidence=0.9,
        gate_depth_confidence=0.8,
    )
    row = gate_sight_record(
        frame_id=42,
        plan=RacePlan("APPROACH_GATE", 2.0),
        est=est,
        camera=camera,
        detection=det,
        track=track,
        target_gate_index=0,
        detections_count=2,
        all_centers_px=[(280.0, 220.0), (500.0, 200.0)],
    )
    assert row["pixel_error_px"] == [-40.0, 40.0]
    assert row["gate_center_px"] == [280.0, 220.0]
    assert row["detections_count"] == 2
