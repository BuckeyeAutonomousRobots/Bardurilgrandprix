# Module Reference — `src/`

Quick file-by-file map for code review and branch upload.

---

## `src/main.py`

Orchestrator. Loads `sim.yaml` or `sim_comp.yaml` + `gains.yaml` + `camera.yaml`, merges vision keys into gains dict.

- Spawns perception thread (detector + tracker at `perception_hz`)
- Control loop at `control_hz`, planner at `planner_hz`
- Position discontinuity detection → recover or force pass near gate
- Control saturation watchdog → recover (guarded when gate locked)
- CLI: `--sim-config`, `--show-vision`, `--max-seconds`, `--log-path`

---

## `src/planning/race_fsm.py`

`RaceFSM` state machine. Inputs: `EstimatedState`, optional `recover_requested`.

Key vision-primary helpers:

- `_gate_track_usable()` — fresh detection or short predicted dropout
- `_vision_gate_locked()` — confidence + usable track (looser than `gate_strong`)
- Vision transitions: acquire → align → approach → commit → pass

---

## `src/control/attitude_controller.py`

`AttitudeController.update()` → `AttitudeCommand` (roll/pitch/yaw setpoints + thrust).

Key methods:

| Method | Role |
|--------|------|
| `_vertical_pitch_trim` | Aim pitch at gate height |
| `_clamp_gate_pitch_sp` | Nose-down/up limits per FSM state |
| `_vision_hover_altitude_target` | ALIGN/APPROACH hover targets |
| `_vision_altitude_hold_thrust` | Gate-below descent thrust caps |
| `_vision_search_command` | SEARCH scan + steer |
| `_takeoff_command` | Low-altitude takeoff with thrust floor/cut |
| `_gate_pitch_sp` | Forward + vertical + brake pitch blend |

---

## `src/estimation/state_estimator.py`

Fuses gate track into `EstimatedState`:

- Smoothed `gate_bearing_x/y_rad`, `gate_range_m`, `gate_confidence`
- Map path only when `vision_primary_navigation` is false
- `prefer_visual` disables map bearing when vision fresh

---

## `src/estimation/vision_depth.py`

Range from bbox area / QuAdGate corners + camera intrinsics.

---

## `src/tracking/gate_tracker.py`

Maintains `GateTrack` with exponential smoothing; emits predicted tracks when detections drop.

---

## `src/perception/gate_detector.py`

Facade: selects sim detector, applies `gate_selector` for multi-detection frames.

---

## `src/perception/sim_gate_detector.py`

GateNet inference + orange CV; produces `GateDetection` with bbox, corners, confidence.

---

## `src/perception/gatenet/`

| File | Role |
|------|------|
| `gate_net.py` | U-Net model definition |
| `monorace_gate_detector.py` | Orange HSV + inference wrapper |
| `monorace_perception.py` | Pre/post processing |
| `quad_gate.py` | Corner geometry |

---

## `src/comms/mavlink_client.py`

UDP MAVLink: heartbeat, attitude, LOCAL_POSITION_NED, race status, track gates.

---

## `src/vision/`

| File | Role |
|------|------|
| `udp_receiver.py` | JPEG chunk receiver |
| `jpeg_reassembler.py` | Frame assembly |
| `frame_preview.py` | `--show-vision` overlay |
| `gate_sight_log.py` | Structured gate-in-view log records |

---

## `src/planning/map_guidance.py`

Map-based bearings, plane distance, gate volume (used in map-first mode; race status still used in vision-primary).

---

## `src/infra/`

| File | Role |
|------|------|
| `logger.py` | JSONL flight log |
| `drone_action_logger.py` | Action journal for before/after forensics |

---

## `src/types.py`

Dataclasses: `VehicleState`, `GateTrack`, `GateDetection`, `EstimatedState`, `RacePlan`, `AttitudeCommand`, `FSMTransition`.

---

## Tests (`tests/`)

| File | Covers |
|------|--------|
| `test_src_race_fsm.py` | FSM transitions, vision takeoff, gate lock recover |
| `test_src_attitude_controller.py` | Pitch trim, altitude, align hover |
| `test_src_state_estimator.py` | Bearing blend, map/vision |
| `test_src_main.py` | Position jump ignore |
| `test_src_jpeg_reassembler.py` | Vision frame assembly |
| `test_map_guidance.py` | Map geometry helpers |
