# IR-01 Energy-Aware Robot Grid Exploration

An exploration policy for a mobile robot that observes nearby cells, manages limited energy, and safely returns to base while maximizing coverage of reachable free space.

## Problem

- 30×30 grid with 20% obstacles
- Robot starts at base (0,0)
- Manhattan distance ≤ 2 sensing
- 400 energy budget (1 per move)
- Must return to base safely
- **Objective**: Maximize coverage of reachable free cells

## Scoring (100 Points)

| Criteria | Points | Description |
|----------|--------|-------------|
| Coverage (Ck) | 55 | % of reachable free cells observed |
| Return Rate | 20 | Successful returns to base |
| Ck × Ek | 15 | Coverage × remaining energy ratio |
| Runtime | 5 | Execution speed |
| Reproducibility | 5 | Deterministic results |

## Algorithm

**Frontier-Based Exploration with Energy-Aware Return**

1. **Sense**: Reveal cells within Manhattan distance ≤ 2
2. **Find Frontiers**: Revealed free cells adjacent to unknown cells
3. **Score**: `info_gain / distance_to_frontier`
4. **Energy Check**: If `energy ≤ distance_to_base + 15` → return
5. **Move**: Go to best frontier via BFS shortest path
6. **Repeat**: Until DONE or energy = 0

## Results

| Metric | Value |
|--------|-------|
| Mean Coverage | **97.34%** |
| Successful Return | **100%** |
| Mean Ck × Ek | **0.0419** |
| Mean Runtime | **1.38s** |
| **Estimated Score** | **84.2/100** |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Experiments

```bash
python run_experiment.py
```

### Run 2D Dashboard

```bash
python dashboard_pro.py
# Open http://localhost:5000
```

### Run 3D Dashboard

```bash
python dashboard_3d_pro.py
# Open http://localhost:5001
```

## Project Structure

```
├── grid_simulator.py          # Grid generation & simulation
├── explorer.py                # Exploration algorithm
├── controller.py              # Online decision controller
├── run_experiment.py          # Experiment runner
├── dashboard_pro.py           # 2D visualization
├── dashboard_3d_pro.py        # 3D visualization
├── templates/
│   ├── dashboard.html         # 2D UI
│   └── dashboard3d.html       # 3D UI (Three.js)
├── logs/                      # Trajectory logs
├── metrics/                   # Experiment metrics
├── visualizations/            # Coverage plots
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Features

- **Info-Gain Scoring**: Prioritizes frontiers that reveal the most new cells
- **Energy Margin Optimization**: Tested 20 values, found optimal margin = 15
- **Single-Pass BFS**: Efficient pathfinding to all reachable cells
- **Real-time Dashboard**: 2D and 3D visualization with controls
- **Reproducible**: Deterministic seed-based grid generation

## Dependencies

- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- Flask ≥ 3.0.0

## License

MIT License

## Acknowledgments

- IoT, Robotics & Automation Hackathon
- Clash of Coders 3.0
