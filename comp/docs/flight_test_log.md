# Flight Test Log

## Logging Format

For each run, record:

- Date/time
- Objective
- Code/config state
- Launch command
- Result
- What worked
- What did not work
- Next tweak

## 2026-06-13

### Reference Import

- Change: imported `C:\Users\notor\Downloads\Bardurilgrandprix-main.zip` into `external/Bardurilgrandprix-main`
- Objective: inspect whether this external baseline provides a better ACRO stability architecture than the current workspace
- What worked:
- The package extracted cleanly and is now available in-tree for direct code comparison
- What did not work:
- The imported sample is not a stronger controller; its sample `controller.py` sends fixed body-rate commands with `thrust = 1.0` and no closed-loop takeoff or hover stabilization
- Next tweak:
- Treat the import as a reference baseline only and continue stabilizing the existing ACRO controller rather than swapping to this sample

### Setup Baseline

- Current baseline pinned in `docs/current_setup.md`.
- First live verification target: straight-flight control with `run_straight_acro.ps1`.
- Goal: confirm arm, takeoff, heading lock, and forward motion before touching race-target logic.

### Run 1

- Date/time: 2026-06-13 evening local
- Objective: baseline straight-flight verification
- Code/config state: initial straight verifier baseline
- Launch command: `powershell -ExecutionPolicy Bypass -File .\run_straight_acro.ps1 -MaxSeconds 45`
- Result: failed to take off
- What worked:
- Simulator launched
- MAVLink heartbeat connected
- Controller armed the vehicle
- Straight verifier produced a telemetry CSV at `logs/straight_verifier/straight_20260613_214617.csv`
- What did not work:
- Auto flight-mode set was not calibrated on this machine
- The controller entered `TAKEOFF` but never received fresh position telemetry
- Logged telemetry stayed pinned at zero / missing position for the full run
- The drone did not climb or move forward
- Next tweak:
- Keep requesting telemetry streams after arming instead of only during `INIT`
- Emit a clear warning when a run arms without fresh position telemetry
- Re-run after this tweak and verify the sim is in an active flight scene with `ACRO` set manually if needed

### Run 1 Follow-up Tweak

- Change: straight verifier now re-requests telemetry after arming and warns explicitly when position telemetry never becomes fresh
- Files:
- `pilot/control/straight_controller.py`
- `docs/current_setup.md`

### Run 2

- Date/time: 2026-06-13 evening local
- Objective: confirm telemetry recovery after stream-request tweak
- Code/config state: straight verifier re-requests telemetry after arming
- Launch command: `powershell -ExecutionPolicy Bypass -File .\run_straight_acro.ps1 -NoLaunch -MaxSeconds 20`
- Result: telemetry recovered, flight unstable
- What worked:
- Fresh position, attitude, velocity, and actuator telemetry came alive
- Ground reference was captured correctly at the start of the run
- The drone did respond to control input and left the ground
- What did not work:
- Takeoff immediately over-climbed far past the `2.0 m` target
- Roll angle diverged into repeated loss-of-control behavior
- The verifier never stabilized enough to transition into clean straight flight
- Auto ACRO switching still was not calibrated; manual ACRO remains a setup dependency
- Next tweak:
- Tune thrust through the wrapper instead of bypassing the normal launch path
- Prefer a full simulator relaunch between unstable runs

### Run 3

- Date/time: 2026-06-13 evening local
- Objective: parameter-only thrust reduction test
- Code/config state: same straight verifier, lower thrust via CLI only
- Launch command: `python -m pilot.straight_main --sim-mode ACRO --reset --max-seconds 15 --hover-thrust 0.30 --takeoff-thrust 0.36 --forward-pitch-deg 2.0 --pitch-ramp-seconds 3.0`
- Result: inconclusive
- What worked:
- The lower-thrust configuration can be launched directly
- What did not work:
- The run did not start from a clean ground state
- Ground reference was captured while the vehicle was already in a bad residual state
- This run is not usable as a control-quality comparison against Run 2
- Next tweak:
- Expose thrust tuning through `run_straight_acro.ps1`
- Fully relaunch the simulator before the next lower-thrust comparison run

### Run 3 Follow-up Tweak

- Change: `run_straight_acro.ps1` now exposes `-HoverThrust` and `-TakeoffThrust`
- Files:
- `run_straight_acro.ps1`
- `pilot/scripts/run_straight_acro.ps1`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 2

- Change: straight takeoff now switches to upright recovery when tipped, and only applies takeoff thrust while sufficiently level
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 3

- Change: tipped takeoff recovery now preserves some extra lift near the ground instead of immediately falling back to pure hover thrust
- Reason: latest observation was that the vehicle barely had enough thrust to break ground before falling off-axis
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 4

- Change: straight verifier now inserts a hover-stabilize phase after takeoff and weakens tilt compensation before straight-flight handoff
- Reason: latest analysis points to a mix of high collective and unstable attitude handoff, not just bad forward guidance
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 5

- Change: stability-first mode is now the default; the straight verifier holds hover after takeoff unless `--forward-after-hover` is explicitly enabled
- Reason: the current objective is stability, not forward progress
- Files:
- `pilot/control/straight_controller.py`
- `pilot/straight_main.py`
- `run_straight_acro.ps1`
- `pilot/scripts/run_straight_acro.ps1`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 6

- Change: ACRO stability mode now targets a separate low hover altitude instead of continuing the full climb to race takeoff height
- Reason: latest ACRO logs showed the vehicle could stay relatively composed near the floor, but remained in `TAKEOFF` too long and kept accumulating climb energy
- Files:
- `pilot/control/straight_controller.py`
- `pilot/straight_main.py`
- `run_straight_acro.ps1`
- `pilot/scripts/run_straight_acro.ps1`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 7

- Change: ACRO stability mode now transitions into `HOVER` earlier and prevents above-target thrust from dropping too far below hover
- Reason: latest run climbed cleanly but then stayed in `TAKEOFF` too long and appeared to fall after collective was cut too aggressively
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 8

- Change: replaced the straight verifier's hover/takeoff inner loop with PID-style roll, pitch, yaw, and altitude hold, and reset the integrators on phase changes
- Reason: repeated ACRO failures showed the previous P plus damping loop was not cancelling persistent bias or recovering cleanly once hover started to drift
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 9

- Change: integrated the new PID logic into a shared helper module so the straight verifier and `TRPYController` use the same PID primitive
- Reason: avoid maintaining two separate PID implementations while continuing ACRO stability tuning
- Files:
- `pilot/control/pid.py`
- `pilot/control/straight_controller.py`
- `pilot/control/trpy_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 10

- Change: takeoff now captures the craft's resting launch attitude and fades that trim out over the first part of the climb instead of treating the initial on-ground pitch bias as an immediate hover-level error
- Reason: latest ACRO logs showed the drone consistently starting at about `+0.31 rad` pitch on the ground and the controller immediately commanding strong nose-down correction, which is an architectural mismatch rather than a small gain error
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 11

- Change: disarm now resets the straight verifier back to `INIT` from `HOVER` as well as other active phases
- Reason: latest post-failure log tail showed the verifier remaining in `HOVER` with stale telemetry after a loss-of-control event, which obscured the actual failure mode
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 12

- Change: added an offline straight-log replay tool to compare current controller outputs against recorded bad-flight telemetry before requesting another live run
- Reason: code-path tests alone do not prove a change will matter on the actual failure trace; replay gives a direct impact check on the same telemetry sequence
- Files:
- `scripts/replay_straight_log.py`
- `docs/current_setup.md`

### Run 3 Follow-up Tweak 13

- Change: ACRO takeoff now estimates projected vertical apex from current climb rate and brakes earlier when the low-hover target would be overshot; `TAKEOFF -> HOVER` handoff is also blocked while that projected overshoot remains too large
- Reason: offline replay showed the launch-trim fix changed pitch behavior, but vertical thrust was still effectively unchanged near `0.55-0.62 m` while the craft was climbing at roughly `1.6 m/s`
- Offline impact check:
- On the recorded bad log near the low-hover target, old thrust was about `0.300` while the current controller now commands about `0.220`
- On the first row the old log labeled `HOVER` (`alt=1.2903`, `vz=-1.0818`), the current controller would now remain in `TAKEOFF` and continue braking instead of handing off
- Files:
- `pilot/control/straight_controller.py`
- `tests/test_straight_controller.py`
- `docs/current_setup.md`

### Run 4

- Date/time: 2026-06-13 23:51 local
- Objective: first live ACRO hover-only run after launch-trim and projected-apex braking changes
- Code/config state: shared PID, launch-trim fade, projected-apex takeoff braking, hover-only stability mode
- Launch command: `powershell -ExecutionPolicy Bypass -File .\run_straight_acro.ps1 -MaxSeconds 20 -HoverThrust 0.34 -TakeoffThrust 0.40 -StabilityHoverAlt 0.6`
- Result: invalid setup-state run, not usable as a controller-quality comparison
- What worked:
- Controller launched and produced a fresh log at `logs/straight_verifier/straight_20260613_235129.csv`
- The new control architecture executed without crashing
- What did not work:
- The sim did not start from a valid ground-rest state
- Ground reference was captured at about `156.36` with initial attitude about `roll=-1.81 rad`
- Initial telemetry already showed large non-rest velocities before any meaningful takeoff evaluation
- The run never produced a valid clean takeoff window for judging the new architecture
- Sim flight-mode automation remains uncalibrated on this machine
- Next tweak:
- Treat this as a simulator/state reset failure, not as evidence for or against the new controller
- Get a clean ground-rest start before the next live comparison
