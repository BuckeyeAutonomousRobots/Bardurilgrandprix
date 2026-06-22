from src.main import _should_auto_arm


def test_should_auto_arm_only_in_startup_states_after_deadline():
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=9.0,
        last_arm_attempt_s=0.0,
        link_ready=True,
        armed=False,
        plan_state="WAIT_START",
    ) is True
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=9.0,
        last_arm_attempt_s=0.0,
        link_ready=True,
        armed=False,
        plan_state="SEARCH_GATE",
    ) is False


def test_should_auto_arm_respects_link_arm_and_retry_guard():
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=9.0,
        last_arm_attempt_s=9.5,
        link_ready=True,
        armed=False,
        plan_state="WAIT_START",
    ) is False
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=11.0,
        last_arm_attempt_s=0.0,
        link_ready=True,
        armed=False,
        plan_state="WAIT_START",
    ) is False
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=9.0,
        last_arm_attempt_s=0.0,
        link_ready=False,
        armed=False,
        plan_state="WAIT_START",
    ) is False
    assert _should_auto_arm(
        now_s=10.0,
        arm_deadline_s=9.0,
        last_arm_attempt_s=0.0,
        link_ready=True,
        armed=True,
        plan_state="WAIT_START",
    ) is False
