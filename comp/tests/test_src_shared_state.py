import numpy as np

from src.types import SharedState, VisionFrame


def test_vision_ready_times_out_when_frames_go_stale():
    shared = SharedState()

    assert shared.vision_is_ready() is False

    shared.update_frame(
        VisionFrame(
            frame_id=1,
            sim_time_ns=123,
            wall_time_s=10.0,
            image_shape=(2, 2),
            image_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        )
    )

    assert shared.vision_is_ready() is True
    assert shared.vision_is_ready(now_s=10.5, timeout_s=1.0) is True
    assert shared.vision_is_ready(now_s=11.1, timeout_s=1.0) is False
