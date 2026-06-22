from src.planning.map_guidance import (
    gate_commit_info,
    gate_plane_metrics,
    gate_target_ned,
    path_target_ned,
)


TRACK = [
    {"gate_id": 0, "position_ned": [20.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    {"gate_id": 1, "position_ned": [40.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
]


def test_gate_plane_metrics_negative_before_gate():
    track = [
        {
            "gate_id": 0,
            "position_ned": [20.0, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "width": 2.7,
            "height": 2.7,
        },
        {"gate_id": 1, "position_ned": [40.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]
    plane = gate_plane_metrics(track, 0, (10.0, 0.0, 0.0))
    assert plane is not None
    assert plane.signed_dist_m < -5.0
    assert plane.within_bounds is True


def test_gate_commit_active_near_gate():
    commit = gate_commit_info(TRACK, 0, (18.5, 0.0, 0.0))
    assert commit is not None
    assert commit.active is True
    assert commit.strength >= 0.40


def test_path_target_uses_approach_standoff_when_far():
    target = path_target_ned(TRACK, 0, (0.0, 0.0, 0.0))
    center = gate_target_ned(TRACK, 0)
    dist_target = ((target[0] - 0.0) ** 2 + (target[1] - 0.0) ** 2) ** 0.5
    dist_center = ((center[0] - 0.0) ** 2 + (center[1] - 0.0) ** 2) ** 0.5
    assert dist_target < dist_center
