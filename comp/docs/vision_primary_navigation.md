# Vision-Primary Navigation

Competition-style gate racing using **camera + MAVLink telemetry only** for steering. Sim track-map positions are not used for navigation; `active_gate_index` from race status still confirms gate passes.

**Config:** `config/sim_comp.yaml`  
**Entry:** `.\run_sim_stack.ps1 -VisionPrimary`  
**Stack:** `python -m src.main --sim-config config/sim_comp.yaml`

---

## Architecture

```
FlightSim.exe
  ├─ MAVLink UDP :14550  → src/comms/mavlink_client.py
  │                         pose, velocity, race status (active_gate_index)
  └─ Vision UDP :5600    → src/vision/udp_receiver.py + jpeg_reassembler.py
                              → src/perception/gate_detector.py
                                   → sim_gate_detector.py (GateNet + orange CV)
                              → src/tracking/gate_tracker.py
                                    → src/estimation/state_estimator.py
                                          → vision_depth.py (range from bbox/corners)
                                    → src/planning/race_fsm.py
                                    → src/control/attitude_controller.py
                                          → MAVLink attitude target
```

### Data flow (one control tick)

1. **MAVLink** updates `VehicleState` (NED position, velocity, attitude).
2. **Vision thread** decodes JPEG chunks, runs detector at `perception_hz`, updates `GateTrack`.
3. **State estimator** smooths bearing/range/confidence; map blend is off when `vision_primary_navigation: true`.
4. **Race FSM** (`planner_hz`) chooses state + target speed from gates, bearings, race status.
5. **Attitude controller** (`control_hz`) outputs roll/pitch/yaw setpoints + thrust.
6. **Logger** writes `logs/sim_stack_<timestamp>.jsonl` (FSM, control, gate_sight, jumps).

---

## FSM states (vision-primary path)

Typical happy path:

```
WAIT_LINK → WAIT_VISION → WAIT_START → TAKEOFF → SEARCH_GATE
  → ALIGN_GATE → APPROACH_GATE → COMMIT_GATE → PASS_GATE → (next gate)
```

| State | Purpose |
|-------|---------|
| `TAKEOFF` | Climb to low `vision_takeoff_altitude_m`; may skip `STABILIZE` if gate visible |
| `SEARCH_GATE` | Yaw scan + light damping; steers toward gate when confidence ≥ threshold |
| `ALIGN_GATE` | Center on gate (roll/yaw/pitch); snap hover to `vision_align_hover_altitude_m` |
| `APPROACH_GATE` | Forward motion + center fixation; dwell then commit |
| `COMMIT_GATE` | Push through gate plane; timeout → `PASS_GATE` |
| `PASS_GATE` | Wait for sim `active_gate_index` advance or re-approach |
| `RECOVER` | Slow + re-acquire after lost gate / saturation (not while gate locked) |

### Sim-authoritative pass

Local `PASS_GATE` is **not** enough. Success requires MAVLink race status:

- `active_gate_index` increments (e.g. 0 → 1), or
- `last_gate_race_time` updates after crossing

Verifier: `python scripts/run_until_gate_pass.py --sim-config config/sim_comp.yaml`

---

## Pitch convention (critical)

In this FlightSim build, **positive pitch setpoint = nose down / forward** in the attitude rate loop.

| Signal | Meaning |
|--------|---------|
| `gate_bearing_y_rad > 0` | Gate below image aim point → need more nose-down pitch |
| `gate_bearing_y_rad < 0` | Gate above aim → reduce nose-down |
| `_vertical_pitch_trim` | `kp * theta_y` — positive trim when gate below |
| `_clamp_gate_pitch_sp` | `hi = hover_pitch + down_limit` allows nose-down |

Launch pitch trim (`capture_launch_pitch_trim`) captures sim spawn pitch (~+0.31 rad) as hover reference so gate tracking is relative to trim, not absolute zero.

---

## Altitude control (vision-primary)

Three coupled mechanisms:

### 1. Hover altitude target (`_vision_hover_altitude_target`)

- **ALIGN:** absolute `vision_align_hover_altitude_m` (default 0.95 m), minus gate-center offset and lower-gate extra drop.
- **APPROACH/COMMIT:** base altitude minus `vision_approach_alt_drop_m` scaled by proximity + gate-center offset.
- Floor: `vision_approach_min_alt_m` (default 0.35 m).

On `ALIGN_GATE` entry, hover snaps to align altitude and integral resets.

### 2. Gate-below descent thrust (`_vision_altitude_hold_thrust`)

When gate is below crosshair and drone is above hover target:

- Disables **sink-boost** (was adding thrust while descending).
- Caps thrust below `hover_thrust` by `vision_gate_descent_thrust_cut` (+ extra cut vs bearing and altitude error).
- Clears positive altitude integral so climb wind-up from takeoff does not fight descent.

**Symptom fixed:** saw gate → pitched down but **thrust spiked upward** (altitude PID fighting pitch).

### 3. Collective limits (`config/gains.yaml`)

| Parameter | Role |
|-----------|------|
| `hover_thrust` | Trim collective (~0.50) |
| `thrust_min` | Lower bound (0.12) — must allow descent |
| `thrust_max` | Upper bound (0.72) |
| `alt_hold_kp` / `alt_hold_ki` | Altitude PI on top of vz-damped baseline |

---

## Gate lock & RECOVER logic

`gate_strong` requires fresh track, visible, not predicted, area ≥ threshold, confidence ≥ `detect_confidence`.

`vision_gate_locked` is looser: usable track (includes short predicted dropout) + smoothed confidence high. Used to **avoid false RECOVER** while visually locked.

| RECOVER trigger | Vision-primary guard |
|-----------------|----------------------|
| `vision_overspeed` | Only when **not** `vision_gate_locked` |
| `gate_timeout_align` | Skipped while confidence ≥ 55% of detect threshold |
| `position_discontinuity` | Ignored in ALIGN/APPROACH/COMMIT |
| `persistent_control_saturation` | Skipped when gate locked in gate phases |

---

## Key source files

| File | Responsibility |
|------|----------------|
| `src/main.py` | Threads, config merge, position-jump handling, saturation watchdog, CLI |
| `src/planning/race_fsm.py` | State machine, vision acquire/align/commit transitions |
| `src/control/attitude_controller.py` | Pitch/roll/yaw/thrust, vision search/align/commit commands |
| `src/estimation/state_estimator.py` | Bearing/range smoothing; map blend off in vision-primary |
| `src/tracking/gate_tracker.py` | Alpha-beta track, predicted frames on dropout |
| `src/perception/sim_gate_detector.py` | GateNet + orange CV pipeline |
| `src/vision/gate_sight_log.py` | Per-frame gate-in-view records for analysis |
| `config/sim_comp.yaml` | FSM thresholds, vision altitudes, pitch limits |
| `config/gains.yaml` | PID, thrust, vertical pitch gains |
| `config/camera.yaml` | Intrinsics, aim point, gate geometry |

---

## Configuration reference (`sim_comp.yaml`)

### Takeoff

| Key | Default | Notes |
|-----|---------|-------|
| `vision_takeoff_altitude_m` | 0.55 | Target climb height |
| `vision_takeoff_min_alt_m` | 0.35 | Minimum to exit takeoff with gate visible |
| `vision_takeoff_skip_stabilize` | true | Go SEARCH/ALIGN if gate acquired |
| `vision_takeoff_max_time_s` | 1.8 | Always exit TAKEOFF by timeout |
| `takeoff_thrust` | 0.56 | Floor thrust while below target |

### Altitude (gate phases)

| Key | Default | Notes |
|-----|---------|-------|
| `vision_align_hover_altitude_m` | 0.95 | Absolute ALIGN hover |
| `vision_approach_min_alt_m` | 0.35 | Hard floor |
| `vision_approach_alt_drop_m` | 0.85 | Max drop from base on approach |
| `vision_gate_center_alt_kp_m` | 2.6 | Lower hover when gate below aim |
| `vision_lower_gate_extra_drop_per_rad_m` | 2.2 | Extra drop per rad gate-below |
| `vision_align_alt_slew_rate_mps` | 4.0 | Fast downward slew in ALIGN |

### Pitch (nose-down for lower gates)

| Key | Default | Notes |
|-----|---------|-------|
| `vision_align_pitch_limit_rad` | 0.40 | Max nose-down in ALIGN |
| `vision_approach_pitch_limit_rad` | 0.58 | Max nose-down in approach/commit |
| `vision_align_vertical_pitch_scale` | 1.15 | Scales vertical trim in ALIGN |
| `vision_approach_vertical_pitch_scale` | 1.15 | Scales vertical trim in approach |
| `vision_min_nose_down_pitch_delta_rad` | 0.14 | Minimum nose-down floor when gate below |
| `vision_gate_below_bearing_rad` | 0.08 | Threshold for “gate below” logic |

### Descent thrust caps (gate below)

| Key | Default | Notes |
|-----|---------|-------|
| `vision_gate_descent_thrust_cut` | 0.10 | Base cut from hover_thrust |
| `vision_gate_descent_extra_thrust_cut_per_rad` | 0.22 | More cut when gate further below |
| `vision_gate_descent_thrust_cut_max` | 0.20 | Total cut cap |

### FSM / recover

| Key | Default | Notes |
|-----|---------|-------|
| `vision_recover_speed_mps` | 3.0 | Overspeed threshold (only if not locked) |
| `vision_align_recover_timeout_s` | 4.5 | Gate-lost timeout before RECOVER |
| `vision_align_speed_brake_mps` | 1.2 | Pitch brake when faster than this |

---

## Running & debugging

### Launch

```powershell
cd comp
.\run_sim_stack.ps1 -VisionPrimary -ShowVision -WaitSeconds 5
```

Flags:

| Flag | Effect |
|------|--------|
| `-VisionPrimary` | Uses `config/sim_comp.yaml` |
| `-ShowVision` | Live OpenCV window with gate overlay |
| `-KeepPrevious` | Do not kill prior stack process |
| `-MaxSeconds N` | Auto-stop after N seconds |
| `-NoLaunch` | Assume FlightSim already running |

### Verify gate pass

```powershell
python scripts/run_until_gate_pass.py --sim-config config/sim_comp.yaml --max-attempts 3 --max-seconds 90
```

### Log analysis

```powershell
python scripts/diag_latest_log.py          # FSM transitions
python scripts/show_gate_sight.py          # Gate-in-view timeline + altitude
python scripts/analyze_latest_log.py       # Full stats
python scripts/analyze_collisions.py       # Impacts + misalignment
```

### Useful log events

| Event | What to check |
|-------|----------------|
| `fsm_transition` | `reason` — e.g. `vision_overspeed`, `vision_gate_strong_acquire` |
| `gate_sight` | `altitude_m`, `gate_bearing_rad`, `pixel_error_px` |
| `control` | `thrust`, `saturation.thrust_min`, `vehicle_attitude_rad` |
| `position_jump` | `ignored` vs recover trigger |
| `recover_request` | `reason` |

### Stuck lock file

```powershell
Remove-Item logs\sim_stack.active.json -ErrorAction SilentlyContinue
```

---

## Tests

```powershell
python -m pytest tests/test_src_race_fsm.py tests/test_src_attitude_controller.py tests/test_src_state_estimator.py tests/test_src_main.py -q
```

Coverage highlights:

- Vision takeoff skip stabilize / timeout exit
- `vision_gate_locked` prevents overspeed RECOVER
- Align absolute hover altitude
- Vertical pitch trim sign (gate below → nose down)
- Position jump ignore rules

---

## Known issues & tuning notes

1. **Sim telemetry jumps** — LOCAL_POSITION_NED can spike; jumps ignored in gate phases and after reset.
2. **thrust_min saturation** — If altitude stays high, check `thrust_min` and gate-below thrust cap in logs.
3. **Flying over gate** — Lower `vision_align_hover_altitude_m`, increase pitch limits / `vertical_gate_pitch_kp`.
4. **RECOVER after ALIGN** — Check transition `reason`; should not fire on `vision_overspeed` when confidence high.
5. **Pass unconfirmed** — FSM may reach PASS locally while `active_gate_index` stays 0; tune COMMIT speed/alignment.

---

## Vision model

- **GateNet** U-Net: `models/gate_net.pth` (384×384 input)
- **Orange CV fallback:** `src/perception/gatenet/monorace_gate_detector.py`
- Train: `.\scripts\train_gatenet.ps1`

---

## Change history (session summary)

| Area | Change |
|------|--------|
| Vision-primary mode | `sim_comp.yaml` + FSM/estimator map bypass |
| Takeoff | Low altitude targets, skip stabilize when gate visible |
| ALIGN altitude | Snap to `vision_align_hover_altitude_m`, lower-gate extra drop |
| Pitch | Increased nose-down limits; gate-below floor in ALIGN/approach |
| Altitude vs pitch | Gate-below descent thrust cap; sink-boost disabled |
| RECOVER | `vision_gate_locked`; ignore jumps in gate phases |
| Logging | `gate_sight` events + `show_gate_sight.py` |
| Launcher | Auto-stop previous stack; `-VisionPrimary` flag |
