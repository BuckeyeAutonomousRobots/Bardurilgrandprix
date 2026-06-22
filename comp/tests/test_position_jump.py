from src.main import _position_jump_is_spawn_rebase, _should_ignore_position_jump


def test_spawn_rebase_detects_sim_teleport():
    from_pos = (-579.63, 0.18, -3.65)
    to_pos = (-0.0004, -0.0, 0.02)
    jump_m = 579.63
    assert _position_jump_is_spawn_rebase(from_pos, to_pos, jump_m) is True


def test_ignore_position_jump_during_takeoff():
    assert _should_ignore_position_jump(
        "TAKEOFF",
        now_s=100.0,
        reset_sent_mono=90.0,
        from_pos=(-100.0, 0.0, -3.0),
        to_pos=(0.0, 0.0, 0.0),
        jump_m=100.0,
        sim_config={},
    )


def test_real_jump_not_ignored_in_approach():
    assert not _should_ignore_position_jump(
        "APPROACH_GATE",
        now_s=100.0,
        reset_sent_mono=10.0,
        from_pos=(10.0, 0.0, -2.0),
        to_pos=(40.0, 0.0, -2.0),
        jump_m=30.0,
        sim_config={"position_jump_ignore_after_reset_s": 5.0},
    )
