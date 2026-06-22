# Uploading `comp/` to a Git Branch

This folder is the **self-contained sim gate stack** (`src/`, `config/`, `scripts/`, `tests/`). Use this guide when pushing to your remote branch.

---

## What to include

```
comp/
├── README.md
├── requirements.txt
├── run_sim_stack.ps1
├── config/
│   ├── sim.yaml              # map-first dev mode
│   ├── sim_comp.yaml         # vision-primary competition mode
│   ├── gains.yaml
│   └── camera.yaml
├── src/                      # full flight stack
├── scripts/                  # run + log analysis
├── tests/test_src_*.py
├── docs/
│   ├── vision_primary_navigation.md
│   ├── current_setup.md
│   └── BRANCH_UPLOAD.md
└── models/gate_net.pth       # if size allows (or document download)
```

## What to exclude

Add to `.gitignore` (repo root or `comp/.gitignore`):

```
logs/
*.jsonl
__pycache__/
.venv/
pilot/.venv/
AIGP_3364/                    # FlightSim binary (large)
*.pyc
logs/sim_stack.active.json
```

---

## Option A: `comp/` inside existing AI-GP repo

From repo root (`AI-GP/`):

```powershell
git checkout -b feature/vision-primary-nav
git add comp/
git status
git commit -m "Add vision-primary sim gate stack with low-altitude gate tracking"
git push -u origin feature/vision-primary-nav
```

If `comp/` was previously untracked (`?? comp/` in `git status`), the first `git add comp/` stages the whole tree.

---

## Option B: `comp/` as its own repo

```powershell
cd comp
git init
git add README.md requirements.txt run_sim_stack.ps1 config src scripts tests docs
git commit -m "Vision-primary sim gate stack"
git remote add origin <your-remote-url>
git push -u origin main
```

---

## Pre-push checklist

- [ ] `python -m pytest tests/test_src_*.py -q` passes
- [ ] `config/sim_comp.yaml` has `vision_primary_navigation: true`
- [ ] `models/gate_net.pth` present or README documents how to obtain it
- [ ] `AIGP_3364/FlightSim.exe` documented as local-only (not in git)
- [ ] No secrets in logs or config

---

## PR description template

```markdown
## Summary
- Vision-primary navigation stack for AI-GP FlightSim qualifier
- Camera + telemetry steering (no track-map navigation in comp mode)
- Low-altitude gate alignment with nose-down pitch + descent thrust caps
- RECOVER guards when gate is locked

## Run
.\run_sim_stack.ps1 -VisionPrimary -ShowVision

## Verify
python scripts/run_until_gate_pass.py --sim-config config/sim_comp.yaml

## Docs
- comp/docs/vision_primary_navigation.md — full technical reference
- comp/README.md — quick start
```

---

## Key commits to highlight

1. **Vision-primary FSM** — `race_fsm.py`, `state_estimator.py`, `sim_comp.yaml`
2. **Altitude + pitch** — `attitude_controller.py` (hover targets, gate-below thrust, pitch limits)
3. **Robustness** — position jump ignore, `vision_gate_locked`, saturation guards in `main.py`
4. **Ops** — `run_sim_stack.ps1`, gate sight logging, analysis scripts
