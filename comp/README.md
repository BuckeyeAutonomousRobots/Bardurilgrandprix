# AI-GP Sim Gate Stack

Modular gate-racing stack for the AI-GP FlightSim qualifier.

**Two navigation modes:**

| Mode | Config | Use |
|------|--------|-----|
| Map-first (dev/tuning) | `config/sim.yaml` | Fast iteration using sim track data |
| Vision-primary (comp) | `config/sim_comp.yaml` | Camera + telemetry only — no track-map steering |

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/vision_primary_navigation.md](docs/vision_primary_navigation.md) | Full vision-primary reference: FSM, altitude, pitch, config, debugging |
| [docs/module_reference.md](docs/module_reference.md) | File-by-file `src/` map |
| [docs/BRANCH_UPLOAD.md](docs/BRANCH_UPLOAD.md) | Git branch upload checklist |
| [docs/current_setup.md](docs/current_setup.md) | Architecture + scripts overview |

## Quick start

```powershell
# Map-first (default)
.\run_sim_stack.ps1

# Competition-style vision-primary (recommended for qualifier)
.\run_sim_stack.ps1 -VisionPrimary -ShowVision -WaitSeconds 5
```

Validate first gate pass:

```powershell
python scripts/run_until_gate_pass.py --max-attempts 3 --max-seconds 75
python scripts/run_until_gate_pass.py --sim-config config/sim_comp.yaml --max-attempts 3
python scripts/diag_latest_log.py
python scripts/analyze_collisions.py
```

## Layout

```
comp/
├── run_sim_stack.ps1          # Launch sim + python -m src.main
├── config/
│   ├── sim.yaml               # FSM thresholds, ports, map-first gate
│   ├── gains.yaml             # Attitude / hover / approach gains
│   └── camera.yaml            # Intrinsics, gate size
├── models/
│   └── gate_net.pth           # GateNet U-Net weights (~837k params)
├── src/                       # Entire flight stack (self-contained)
│   ├── main.py                # Entry point
│   ├── comms/mavlink_client.py
│   ├── control/attitude_controller.py
│   ├── estimation/state_estimator.py
│   ├── planning/race_fsm.py + map_guidance.py
│   ├── perception/gate_detector.py + sim_gate_detector.py
│   ├── perception/gatenet/    # GateNet U-Net, QuAdGate, orange CV fallback
│   ├── tracking/gate_tracker.py
│   └── vision/udp_receiver.py
├── scripts/                   # Run helpers + log analysis
├── tests/test_src_*.py        # Unit tests for gate-pass logic
└── logs/                      # sim_stack_*.jsonl flight logs (kept)
```

## Vision model

GateNet lives in `src/perception/gatenet/` — no parent-repo dependency:

- **GateNet** U-Net at `models/gate_net.pth` (384×384 input, QuAdGate corners)
- **aigp_orange** HSV color profile fallback in `monorace_gate_detector.py`
- Train: `scripts/train_gatenet.ps1` → `python -m src.tools.train_gate_net`

## Vision-primary highlights

Competition mode (`config/sim_comp.yaml`) implements:

- Low takeoff (`vision_takeoff_altitude_m` ~0.55 m) with optional skip of STABILIZE when gate visible
- ALIGN hover snap to `vision_align_hover_altitude_m` with lower-gate altitude drop
- Nose-down pitch toward gates below crosshair (positive pitch = forward/down in this sim)
- Gate-below descent thrust caps so altitude PID does not climb while pitching down
- `vision_gate_locked` prevents false RECOVER when track is briefly predicted but confidence is high

See [docs/vision_primary_navigation.md](docs/vision_primary_navigation.md) for full detail.

## What made the first gate pass work

1. **Map homing** — `state_estimator` targets full gate center `(X,Y,Z)`, not horizontal-only
2. **Altitude-first control** — `attitude_controller` slews to gate Z, limits forward until aligned
3. **Sim-authoritative FSM** — `race_fsm` trusts `active_gate_index` from MAVLink race status
4. **Gate volume checks** — commit/pass require `map_within_gate_bounds` + plane crossed
5. **Vision** — GateNet + orange CV for bearing/confidence (map drives navigation)

## Config

| File | Role |
|------|------|
| `config/sim.yaml` | Map-first dev mode (`vision_primary_navigation: false`) |
| `config/sim_comp.yaml` | Vision-primary comp mode (no track-map steering) |
| `config/gains.yaml` | Pitch/roll/yaw, altitude PID, collision slow-down near gate |
| `config/camera.yaml` | `fx/fy/cx/cy`, gate inner size for range estimate |

## Tests

```powershell
python -m pytest tests/test_src_race_fsm.py tests/test_src_state_estimator.py tests/test_src_attitude_controller.py tests/test_src_main.py tests/test_map_guidance.py -q
```

## Uploading to a branch

See [docs/BRANCH_UPLOAD.md](docs/BRANCH_UPLOAD.md) for what to commit, `.gitignore`, and PR template.

## Logs

Each run writes:

- `logs/sim_stack_<timestamp>.jsonl` — FSM, control, telemetry, collisions
- `logs/sim_stack_<timestamp>_actions.jsonl` — per-action before/after state

These are **not** deleted during cleanup — they are the tuning record.

## Requirements

```powershell
pip install -r requirements.txt
```

Needs `AIGP_3364/FlightSim.exe` (or update path in `run_sim_stack.ps1`).
