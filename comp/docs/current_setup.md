# Sim Gate Stack — Current Setup

**Entry:** `.\run_sim_stack.ps1` → `python -m src.main`

## Architecture

```
FlightSim.exe
  ├─ MAVLink UDP :14550 → src/comms/mavlink_client.py (pose, race status, track map)
  └─ Vision UDP :5600   → src/vision/udp_receiver.py
                              → src/perception/gate_detector.py
                                   → src/perception/sim_gate_detector.py (GateNet + orange CV)
                              → src/tracking/gate_tracker.py
                                    → src/estimation/state_estimator.py
                                          → src/planning/map_guidance.py
                                    → src/planning/race_fsm.py
                                    → src/control/attitude_controller.py
```

## Gate pass criteria (sim-validated)

- `active_gate_index` advances (0 → 1) in MAVLink race status, **or**
- `last_gate_race_time > 0` after crossing

Local FSM `PASS_GATE` alone is not sufficient — the sim must agree.

## Key tuning files

| Symptom | Look at |
|---------|---------|
| Wrong altitude at gate | `gains.yaml` vertical pitch, `map_gate_alt_slew_rate_mps` |
| Hits gate frame | `gains.yaml` `map_collision_*`, approach speed caps |
| Never commits | `sim.yaml` `map_gate_volume_dist_m`, lateral/vertical thresholds |
| False pass | `race_fsm.py` sim-authoritative transitions |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_until_gate_pass.py` | Retry loop until sim gate advance |
| `scripts/diag_latest_log.py` | FSM transitions + per-state stats |
| `scripts/analyze_latest_log.py` | Full log analysis |
| `scripts/analyze_collisions.py` | Speed/misalignment + collision events |
| `scripts/analyze_action_log.py` | Action journal summary |

## Vision model

- **GateNet** U-Net at `models/gate_net.pth` (384×384 input, QuAdGate corners)
- Fallback: **aigp_orange** HSV color profile via `src/perception/gatenet/monorace_gate_detector.py`
- Train: `scripts/train_gatenet.ps1`

## Removed legacy stacks

The following were removed to isolate the working path:

- `pilot/control/champion_controller.py` and vision/champion/onboard pilots
- Duplicate onboard copy of `src/`
- `external/Bardurilgrandprix-main/` reference tree

All gate-pass tuning now lives in `src/` + `config/`.

## Vision-primary (competition mode)

Full reference: **[vision_primary_navigation.md](vision_primary_navigation.md)**

Quick run:

```powershell
.\run_sim_stack.ps1 -VisionPrimary -ShowVision
python scripts/run_until_gate_pass.py --sim-config config/sim_comp.yaml
python scripts/show_gate_sight.py
```

Branch upload guide: **[BRANCH_UPLOAD.md](BRANCH_UPLOAD.md)**  
Module map: **[module_reference.md](module_reference.md)**
