<p align="center">
  <img src="docs/insyght-logo.png" alt="Insyght" width="300">
</p>

# Antenna Insyght

Dual-band PCB IFA antenna design and optimization tool for NB-IoT and LTE-M applications.

Antenna-Insyght automates the design of compact Inverted-F Antennas (IFA) on half-circle PCB substrates, targeting the 700--960 MHz (low band) and 1710--2170 MHz (high band) cellular bands. It combines electromagnetic simulation, Bayesian optimization, matching network synthesis, and manufacturing export into a single integrated workflow.

## Features

- **Branched IFA topology** -- single-feed, dual-band antenna with low-band and high-band arms on a half-circle FR4 substrate
- **Dual simulation backends** -- NEC2 (fast method-of-moments via `necpp`) and openEMS (full-wave FDTD for accuracy)
- **Bayesian optimization** -- multi-phase optimizer using scikit-optimize with Gaussian Process surrogate to find optimal antenna geometry
- **Matching network synthesis** -- automated LC matching network design with differential evolution, targeting worst-case S11 across both bands
- **Interactive GUI** -- PySide6 desktop application with real-time parameter editing, S11/Smith chart/impedance plots, and geometry preview
- **DXF export** -- KiCad-compatible DXF output with proper layer assignment (F.Cu, B.Cu, Edge.Cuts, F.Fab)
- **KiCad project generation** -- complete KiCad 8 project with schematic, PCB layout, and footprint library
- **PDF reports** -- comprehensive design reports with plots, parameter tables, and pass/fail performance summary

## Workflow

```
  Parameters        Simulate         Optimize         Match           Export
 +-----------+    +-----------+    +-----------+    +-----------+    +-----------+
 | Antenna   |--->| NEC2 or   |--->| Bayesian  |--->| LC match  |--->| DXF/KiCad |
 | geometry  |    | openEMS   |    | optimizer |    | network   |    | PDF report|
 +-----------+    +-----------+    +-----------+    +-----------+    +-----------+
       |               |                |                |                |
   config.py      S11, Z11,        Best params      Component       Manufacturing
   GUI params     impedance        for both bands   values (L,C)    ready files
```

## Installation

### Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `scipy`, `matplotlib`, `scikit-optimize`, `necpp`, `ezdxf`, `reportlab`, `PySide6`.

### openEMS (optional, for full-wave FDTD simulation)

openEMS and CSXCAD are not pip-installable and require system-level installation. See the [openEMS installation guide](https://docs.openems.de/install.html) for instructions.

When openEMS is not installed, the tool falls back to NEC2 simulation (faster, but less accurate for complex geometries).

## Quick Start

### GUI mode

```bash
python -m gui.main
```

This launches the interactive antenna designer where you can:
- Adjust antenna parameters and see geometry updates in real time
- Run simulations and view S11, Smith chart, and impedance plots
- Optimize the design with Bayesian optimization
- Design matching networks
- Export DXF, KiCad projects, and PDF reports

### CLI optimization

```bash
python optimize.py
```

Runs the full multi-phase optimization pipeline:
1. Coarse arm-length sweep to find approximate resonance
2. Feed offset sweep for impedance matching
3. Bayesian optimization over all parameters
4. Verification with high-resolution simulation

Results are saved to `results/`.

### Validate a design

```bash
python validate_tuned.py
```

### Generate exports

```bash
python export_dxf.py          # DXF for KiCad import
python create_kicad_project.py # Full KiCad 8 project
python report.py               # PDF design report
```

## Project Structure

```
Antenna-Insyght/
├── config.py               # Central configuration (geometry, frequencies, targets)
├── antenna_model.py        # Parametric IFA geometry model (NEC2 wire lists)
├── simulate.py             # NEC2 simulation backend
├── simulate_openems.py     # openEMS FDTD simulation backend
├── simulate_dispatch.py    # Backend dispatcher (routes to NEC2 or openEMS)
├── optimize.py             # Multi-phase Bayesian optimization loop
├── matching_network.py     # LC matching network synthesis
├── deep_match.py           # Extended matching network optimization
├── push_20db.py            # Targeted S11 improvement utility
├── validate_tuned.py       # Design validation against targets
├── utils.py                # S-parameter math, plotting helpers
├── export_dxf.py           # DXF export for KiCad
├── create_kicad_project.py # KiCad 8 project generator
├── report.py               # PDF report generator (reportlab)
├── requirements.txt        # Python dependencies
├── gui/
│   ├── main.py             # GUI entry point
│   ├── main_window.py      # Main window with tabs and toolbar
│   ├── state.py            # Central design state model (signals/slots)
│   ├── panels/
│   │   ├── parameters.py   # Parameter editing panel
│   │   ├── geometry_view.py# Antenna geometry visualization
│   │   ├── s11_view.py     # S11 / VSWR plot
│   │   ├── smith_view.py   # Smith chart plot
│   │   ├── impedance_view.py # Impedance (R+jX) plot
│   │   └── matching.py     # Matching network panel
│   ├── dialogs/
│   │   ├── freq_bands.py   # Frequency band configuration
│   │   ├── sim_settings.py # Simulation settings dialog
│   │   └── export_dialog.py# Export options dialog
│   └── workers/
│       ├── simulation.py   # Background simulation worker
│       ├── optimization.py # Background optimization worker
│       └── matching_opt.py # Background matching optimization worker
└── results/                # Generated simulation results (gitignored)
```

## How It Works

**Antenna model** -- The branched IFA is a single-feed antenna with two radiating arms sharing a common shorting stub and feed point. The low-band arm meanders within the half-circle boundary to achieve the required electrical length (~0.25 wavelength at 875 MHz). The high-band arm branches off near the feed at a configurable angle. All geometry is constrained to fit within a semicircular PCB outline.

**Simulation** -- NEC2 models the antenna as thin wires in free space with a substrate correction factor (sqrt of effective permittivity). openEMS uses full-wave FDTD with the actual FR4 substrate, copper traces, and PML boundary conditions for higher accuracy.

**Optimization** -- The optimizer minimizes worst-case S11 across both frequency bands simultaneously. It uses a multi-phase strategy: coarse sweep to find the resonance region, then Bayesian optimization with a Gaussian Process surrogate model to efficiently explore the parameter space.

**Matching** -- After optimization, an LC matching network is synthesized to further improve S11. The matching network uses differential evolution to find optimal component values that minimize worst-case S11 across both bands.

**Export** -- The final design is exported as DXF (with KiCad layer assignments), a complete KiCad 8 project (schematic + PCB + footprint library), and a PDF report with all performance metrics.

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BOARD_RADIUS` | 31.0 mm | Half-circle PCB radius |
| `SUBSTRATE_ER` | 4.4 | FR4 relative permittivity |
| `SUBSTRATE_THICKNESS` | 1.6 mm | PCB thickness |
| `LB_FREQ_MIN/MAX` | 791--960 MHz | Low band target range |
| `HB_FREQ_MIN/MAX` | 1710--1990 MHz | High band target range |
| `SIM_BACKEND` | `openems` | Simulation backend (`nec2`, `openems`, `hybrid`) |
| `ANTENNA_TOPOLOGY` | `branched` | Antenna type (`branched` or `dual_ifa`) |

Performance targets, optimization bounds, and openEMS FDTD settings are also configurable in `config.py`.

## License

Copyright (C) 2026 Insyght B.V.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for the full text.
