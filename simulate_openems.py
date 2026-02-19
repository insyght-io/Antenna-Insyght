# Antenna-Insyght — Dual-Band PCB IFA Antenna Design Tool
# Copyright (C) 2026 Insyght B.V.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
openEMS FDTD simulation backend for the dual-band IFA antenna system.

Uses the openEMS FDTD solver to model the full PCB stackup:
- FR4 substrate with correct permittivity and loss tangent
- Top and bottom copper ground planes
- Antenna traces on top copper
- Lumped feed port through substrate (z-directed)
- Shorting via connecting ground layers to antenna trace

No frequency scaling hacks needed — openEMS models the substrate directly.
"""

import os
import sys
import shutil
import json
import tempfile
import multiprocessing
import numpy as np

from config import (
    BOARD_RADIUS, GND_EXTENSION, GND_HALF_WIDTH,
    SUBSTRATE_THICKNESS, SUBSTRATE_ER, SUBSTRATE_TAND,
    COPPER_THICKNESS, Z0, C0,
    SIM_FREQ_MIN, SIM_FREQ_MAX, SIM_NUM_FREQ, SIM_FREQUENCIES,
    TARGETS, LB_FREQ_CENTER, HB_FREQ_CENTER,
    OPENEMS_END_CRITERIA, OPENEMS_MAX_TIMESTEPS, OPENEMS_MESH_RES,
    OPENEMS_BC, OPENEMS_FAST_TIMESTEPS, OPENEMS_EXPLORE_TIMESTEPS,
    OPENEMS_GND_CLEARANCE,
)
from antenna_model import DualBandIFA, BranchedIFA, MM_TO_M

# Ensure openEMS binary is on PATH (needed for FDTD engine)
# Override with OPENEMS_BIN_DIR environment variable if set
_openems_bin = os.environ.get('OPENEMS_BIN_DIR', os.path.expanduser('~/opt/openEMS/bin'))
if os.path.isdir(_openems_bin) and _openems_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _openems_bin + ':' + os.environ.get('PATH', '')

# openEMS imports (deferred to allow module import even if not installed)
_openems_available = None


def _check_openems():
    """Check if openEMS is available. Caches result."""
    global _openems_available
    if _openems_available is None:
        try:
            import CSXCAD
            import openEMS
            _openems_available = True
        except ImportError:
            _openems_available = False
    return _openems_available


def _require_openems():
    """Import and return (CSXCAD, openEMS) modules, or raise."""
    if not _check_openems():
        raise ImportError(
            "openEMS is not installed. Install via:\n"
            "  brew tap thliebig/openems https://github.com/thliebig/openEMS-Project.git\n"
            "  brew install --HEAD openems\n"
            "Then install Python bindings from the Homebrew cache."
        )
    import CSXCAD
    from CSXCAD import ContinuousStructure
    import openEMS
    from openEMS import openEMS as OpenEMS_Class
    return CSXCAD, ContinuousStructure, openEMS, OpenEMS_Class


def build_openems_model(antenna, band, sim_path, fast=False, explore=False,
                        pad=None):
    """Build the FDTD model for one band's IFA on the PCB.

    Coordinate system (matches antenna_model.py and DXF export):
      - x: along flat edge of semicircle (horizontal)
      - y: distance from ground edge into semicircle clearance area
      - z: through substrate thickness (z=0 bottom copper, z=SUB_T top copper)

    Args:
        antenna: DualBandIFA instance (geometry already generated)
        band: 'lowband' or 'highband'
        sim_path: directory for FDTD simulation files
        fast: if True, use fewer timesteps for faster optimization
        explore: if True, use minimal timesteps for quick parameter exploration

    Returns:
        (fdtd, port, nf2ff_box) tuple
    """
    CSXCAD_mod, ContinuousStructure, openEMS_mod, OpenEMS_Class = _require_openems()

    ifa = antenna.lowband if band == 'lowband' else antenna.highband
    targets = TARGETS[band]

    # Frequency parameters
    f_center = targets['freq_center']
    f_min = SIM_FREQ_MIN
    f_max = SIM_FREQ_MAX

    # Substrate dimensions in mm
    sub_t = SUBSTRATE_THICKNESS  # 1.6 mm
    cu_t = COPPER_THICKNESS      # 0.035 mm

    # FDTD setup — speed hierarchy: explore < fast < full
    if explore:
        max_ts = OPENEMS_EXPLORE_TIMESTEPS
    elif fast:
        max_ts = OPENEMS_FAST_TIMESTEPS
    else:
        max_ts = OPENEMS_MAX_TIMESTEPS
    fdtd = OpenEMS_Class(
        EndCriteria=OPENEMS_END_CRITERIA,
        NrTS=max_ts,
    )
    fdtd.SetBoundaryCond(OPENEMS_BC)

    # Gaussian excitation covering full band
    fdtd.SetGaussExcite((f_min + f_max) / 2, (f_max - f_min) / 2)

    CSX = ContinuousStructure()
    fdtd.SetCSX(CSX)

    # === Materials ===

    # FR4 substrate
    fr4 = CSX.AddMaterial('FR4')
    fr4.SetMaterialProperty(epsilon=SUBSTRATE_ER, kappa=0)
    # Add dielectric loss via conductivity: kappa = 2*pi*f*eps0*er*tan_d
    # Use center frequency for loss estimate
    eps0 = 8.854187817e-12
    kappa_fr4 = 2 * np.pi * f_center * eps0 * SUBSTRATE_ER * SUBSTRATE_TAND
    fr4.SetMaterialProperty(epsilon=SUBSTRATE_ER, kappa=kappa_fr4)

    # PEC for copper
    copper = CSX.AddMetal('Copper')

    # === Geometry ===

    # PCB footprint extents (mm)
    pcb_x_min = -GND_HALF_WIDTH  # -31
    pcb_x_max = GND_HALF_WIDTH   # +31
    pcb_y_min = -GND_EXTENSION   # -88 (ground extends below flat edge)
    pcb_y_max = BOARD_RADIUS     # +30 (semicircle area)

    # Clearance gap between top ground pour and antenna traces.
    # In a real PCB, the top copper ground pour stops at the clearance
    # area boundary. The antenna traces are separate copper islands
    # connected to ground ONLY through the shorting via (through substrate
    # to bottom ground). Without this gap, the top ground and trace bases
    # share the y=0 boundary at z=sub_t, shorting the entire antenna.
    # Larger clearance (3-10mm) reduces Q-factor and widens bandwidth.
    gnd_clearance = OPENEMS_GND_CLEARANCE

    # 1. FR4 substrate block: full PCB footprint
    fr4_start = [pcb_x_min, pcb_y_min, 0]
    fr4_stop = [pcb_x_max, pcb_y_max, sub_t]
    fr4.AddBox(fr4_start, fr4_stop, priority=1)

    # 2. Bottom ground plane (PEC) at z=0
    # Only covers rectangular body (y<0). No ground under antenna clearance
    # area for better radiation resistance and impedance.
    # Many commercial IFA designs leave both layers clear in the antenna area.
    copper.AddBox(
        [pcb_x_min, pcb_y_min, 0],
        [pcb_x_max, 0, 0],
        priority=10,
    )

    # 3. Top ground plane (PEC) at z=sub_t
    # Pulled back from y=0 by gnd_clearance to avoid shorting antenna traces.
    # In a real PCB, this is the top copper pour with clearance cutout.
    copper.AddBox(
        [pcb_x_min, pcb_y_min, sub_t],
        [pcb_x_max, -gnd_clearance, sub_t],
        priority=10,
    )

    # 4. Shorting via: connects bottom ground (z=0) to antenna trace base (z=sub_t)
    # This is the only galvanic connection between ground and antenna trace.
    via_size = 0.4  # mm half-width for via pad
    short_x = ifa.short_x
    copper.AddBox(
        [short_x - via_size, -via_size, 0],
        [short_x + via_size, via_size, sub_t],
        priority=15,
    )

    # 5. Antenna traces on top copper (z = sub_t)
    # Use zero-thickness PEC sheets (not 3D boxes with cu_t thickness)
    # to avoid tiny mesh cells that force extremely small timesteps.
    trace_half_w = ifa.trace_width / 2.0
    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.05:
            continue

        # For each segment, create a copper sheet with trace width at z=sub_t
        if abs(x2 - x1) < 0.01:
            # Vertical segment (constant x)
            copper.AddBox(
                [x1 - trace_half_w, min(y1, y2), sub_t],
                [x1 + trace_half_w, max(y1, y2), sub_t],
                priority=15,
            )
        elif abs(y2 - y1) < 0.01:
            # Horizontal segment (constant y)
            copper.AddBox(
                [min(x1, x2), y1 - trace_half_w, sub_t],
                [max(x1, x2), y1 + trace_half_w, sub_t],
                priority=15,
            )
        else:
            # Angled segment — approximate with a thin polygon via AddLinPoly
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            seg_len = np.sqrt(dx**2 + dy**2)
            # Normal vector for trace width
            nx = -dy / seg_len * trace_half_w
            ny = dx / seg_len * trace_half_w

            # Four corners of the trace strip
            points_x = [x1 + nx, x2 + nx, x2 - nx, x1 - nx]
            points_y = [y1 + ny, y2 + ny, y2 - ny, y1 - ny]
            points = np.array([points_x, points_y])

            # Zero-thickness PEC polygon at z=sub_t
            copper.AddLinPoly(
                points, 'z', sub_t, 0,
                priority=15,
            )

    # 6. Feed via + port: box port through substrate at feed position.
    #    The PEC via (same coords as the port) ensures metal connection AND
    #    overrides the port's lumped R (priority 15 > 5), making the port
    #    effectively an ideal voltage source. This gives correct S11 and
    #    correct P_acc for efficiency computation (P_rad/P_acc ≈ 90% for
    #    PEC on FR4, confirmed by NF2FF validation).
    feed_x = ifa.feed_x
    copper.AddBox(
        [feed_x - via_size, -via_size, 0],
        [feed_x + via_size, via_size, sub_t],
        priority=15,
    )
    port = fdtd.AddLumpedPort(
        port_nr=1,
        R=Z0,
        start=[feed_x - via_size, -via_size, 0],
        stop=[feed_x + via_size, via_size, sub_t],
        p_dir='z',
        excite=1.0,
        priority=5,
    )

    # === Mesh ===
    _generate_mesh(CSX, fdtd, ifa, sub_t, cu_t, f_max,
                   pcb_x_min, pcb_x_max, pcb_y_min, pcb_y_max,
                   pad_override=pad)

    # === NF2FF box for radiation pattern ===
    nf2ff = fdtd.CreateNF2FFBox()

    # Write model
    os.makedirs(sim_path, exist_ok=True)
    CSX.Write2XML(os.path.join(sim_path, 'model.xml'))

    return fdtd, port, nf2ff


def _generate_mesh(CSX, fdtd, ifa, sub_t, cu_t, f_max,
                   pcb_x_min, pcb_x_max, pcb_y_min, pcb_y_max,
                   pad_override=None):
    """Generate adaptive FDTD mesh.

    Strategy:
    - Base resolution: lambda/OPENEMS_MESH_RES at f_max
    - Fine mesh near feed point and substrate interfaces
    - Substrate gets at least 4 z-cells
    - Simulation domain = PCB + padding + absorbing BCs
    """
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # all coordinates in mm

    # Wavelength at highest frequency
    lambda_min_mm = C0 / f_max * 1000.0
    base_res = lambda_min_mm / OPENEMS_MESH_RES  # ~6mm at 2.5 GHz

    # Padding around PCB for boundary conditions.
    # Default 15mm is sufficient for S11 (PML absorbs outgoing waves).
    # For NF2FF/radiation patterns, use pad_override=40+ so the NF2FF box
    # encloses the full antenna structure.
    pad = pad_override if pad_override is not None else 15.0

    # Z mesh: substrate layers need fine resolution
    z_lines = [0]
    # Substrate interior cells
    n_sub_z = max(4, int(np.ceil(sub_t / (base_res / 4))))
    z_sub = np.linspace(0, sub_t, n_sub_z + 1)
    z_lines = list(z_sub)
    # Air above substrate — use arange that extends past target so the
    # boundary lands on a natural grid point (avoids tiny boundary cells
    # after SmoothMeshLines).
    z_top_target = sub_t + pad + 20
    z_above = np.arange(sub_t + base_res, z_top_target + base_res, base_res)
    z_lines.extend(z_above.tolist())
    # Air below substrate (mostly ground plane)
    # NOTE: Don't add -cu_t here — copper is modeled as zero-thickness PEC
    # sheets. Adding a 0.035mm gap creates tiny cells that force extremely
    # small timesteps (dt ~ 6e-14 s instead of ~7e-13 s).
    z_bot_target = -pad - 5
    z_below = np.arange(-base_res, z_bot_target - base_res, -base_res)
    z_lines.extend(z_below.tolist())

    # X mesh
    x_lines = []
    # PCB region with finer mesh
    pcb_res = min(base_res, 3.0)
    x_pcb = np.arange(pcb_x_min, pcb_x_max + pcb_res / 2, pcb_res)
    x_lines.extend(x_pcb.tolist())
    # Ensure exact PCB edges
    x_lines.extend([pcb_x_min, pcb_x_max])
    # Padding — extend arange past target to avoid tiny boundary cells
    x_pad_left = np.arange(pcb_x_min - base_res, pcb_x_min - pad - 5 - base_res, -base_res)
    x_pad_right = np.arange(pcb_x_max + base_res, pcb_x_max + pad + 5 + base_res, base_res)
    x_lines.extend(x_pad_left.tolist())
    x_lines.extend(x_pad_right.tolist())

    # Fine mesh near feed point and short position
    feed_x = ifa.feed_x
    short_x = ifa.short_x
    feed_fine = np.arange(feed_x - 3, feed_x + 3.1, 0.5)
    x_lines.extend(feed_fine.tolist())
    short_fine = np.arange(short_x - 2, short_x + 2.1, 0.5)
    x_lines.extend(short_fine.tolist())

    # Y mesh
    y_lines = []
    # Ground region
    y_gnd = np.arange(pcb_y_min, 0, pcb_res)
    y_lines.extend(y_gnd.tolist())
    # Antenna region (finer)
    antenna_res = min(base_res, 2.0)
    y_ant = np.arange(0, BOARD_RADIUS + 1, antenna_res)
    y_lines.extend(y_ant.tolist())
    # Ensure exact boundaries
    y_lines.extend([pcb_y_min, 0, BOARD_RADIUS])
    # Padding — extend arange past target to avoid tiny boundary cells
    y_pad_bottom = np.arange(pcb_y_min - base_res, pcb_y_min - pad - 5 - base_res, -base_res)
    y_pad_top = np.arange(BOARD_RADIUS + base_res, BOARD_RADIUS + pad + 5 + base_res, base_res)
    y_lines.extend(y_pad_bottom.tolist())
    y_lines.extend(y_pad_top.tolist())

    # Fine mesh near feed point and ground clearance (y direction)
    # Include clearance gap boundary and short/feed positions
    feed_fine_y = np.arange(-2, 3.1, 0.5)
    y_lines.extend(feed_fine_y.tolist())
    # Ensure the clearance gap boundary is meshed
    y_lines.extend([-0.5, -0.25, 0.0, 0.25])

    # Sort and deduplicate, then apply SmoothMeshLines
    from CSXCAD import CSRectGrid
    x_lines = sorted(set(x_lines))
    y_lines = sorted(set(y_lines))
    z_lines = sorted(set(z_lines))

    mesh.AddLine('x', x_lines)
    mesh.AddLine('y', y_lines)
    mesh.AddLine('z', z_lines)

    # Smooth mesh to improve quality
    mesh.SmoothMeshLines('x', base_res, ratio=1.4)
    mesh.SmoothMeshLines('y', base_res, ratio=1.4)
    mesh.SmoothMeshLines('z', base_res / 2, ratio=1.3)


def simulate_s11(antenna, band, freqs=None, verbose=True, fast=False, explore=False):
    """Broadband S11 via single FDTD run.

    Unlike NEC2 which sweeps each frequency, FDTD runs once and extracts
    broadband results via Fourier transform of the time-domain port signals.

    Args:
        antenna: DualBandIFA (geometry generated)
        band: 'lowband' or 'highband'
        freqs: array of frequencies in Hz for output (default: SIM_FREQUENCIES)
        verbose: print progress info
        fast: use fewer timesteps for optimization speed
        explore: use minimal timesteps for quick parameter exploration

    Returns:
        dict with keys: freqs, z_complex, s11_complex, s11_db, vswr
    """
    from utils import impedance_to_s11, s11_db as calc_s11_db, s11_to_vswr

    if freqs is None:
        freqs = SIM_FREQUENCIES

    if verbose:
        mode = "explore" if explore else ("fast" if fast else "full")
        print(f"\nSimulating {band} S11 sweep (openEMS FDTD, {mode} mode)...")
        print(f"  Frequency range: {freqs[0]/1e6:.0f} - {freqs[-1]/1e6:.0f} MHz")
        print(f"  {len(freqs)} output frequency points")

    # Create temp directory for simulation files
    sim_path = tempfile.mkdtemp(prefix=f'openems_{band}_')
    saved_cwd = os.getcwd()

    try:
        fdtd, port, nf2ff = build_openems_model(antenna, band, sim_path,
                                                  fast=fast, explore=explore)

        # Run FDTD simulation
        if verbose:
            print(f"  Running FDTD simulation...")
            fdtd.Run(sim_path, verbose=1)
        else:
            fdtd.Run(sim_path, verbose=0)

        # Extract port data
        port.CalcPort(sim_path, freqs)

        # Impedance from port reflection
        z_complex = port.uf_tot / port.if_tot
        s11_complex = port.uf_ref / port.uf_inc
        s11_vals_db = calc_s11_db(s11_complex)
        vswr_vals = s11_to_vswr(s11_complex)

        if verbose:
            # Print a few sample points
            n_freq = len(freqs)
            for idx in [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]:
                z = z_complex[idx]
                print(f"  f={freqs[idx]/1e6:.0f} MHz: Z={np.real(z):.1f}+j{np.imag(z):.1f}")

        return {
            'freqs': freqs,
            'z_complex': z_complex,
            's11_complex': s11_complex,
            's11_db': s11_vals_db,
            'vswr': vswr_vals,
        }
    finally:
        os.chdir(saved_cwd)
        try:
            shutil.rmtree(sim_path)
        except Exception:
            pass


def simulate_radiation(antenna, band, freq_hz=None):
    """Radiation pattern via NF2FF post-processing.

    Args:
        antenna: DualBandIFA (geometry generated)
        band: 'lowband' or 'highband'
        freq_hz: frequency in Hz (default: band center)

    Returns:
        dict with keys: theta, phi, e_plane_gain, h_plane_gain,
                        gain_max, gain_mean, efficiency, freq_hz
    """
    if freq_hz is None:
        freq_hz = TARGETS[band]['freq_center']

    print(f"\nSimulating {band} radiation pattern at {freq_hz/1e6:.0f} MHz (openEMS FDTD)...")

    sim_path = tempfile.mkdtemp(prefix=f'openems_rad_{band}_')
    saved_cwd = os.getcwd()

    theta_vals = np.arange(0, 181, 5)
    phi_vals = np.arange(0, 361, 5)

    # Default fallback result for when NF2FF fails
    _fallback = {
        'theta': theta_vals, 'phi': phi_vals,
        'e_plane_gain': np.full(len(theta_vals), -30.0),
        'h_plane_gain': np.full(len(phi_vals), -30.0),
        'gain_max': -30.0, 'gain_mean': -30.0,
        'efficiency': 0.0, 'freq_hz': freq_hz,
    }

    try:
        fdtd, port, nf2ff = build_openems_model(antenna, band, sim_path,
                                                pad=40)

        # Run FDTD
        fdtd.Run(sim_path, verbose=0)

        # NF2FF post-processing
        nf2ff_result = nf2ff.CalcNF2FF(
            sim_path, freq_hz,
            theta=theta_vals,
            phi=phi_vals,
        )

        # Extract directivity
        gain_total = nf2ff_result.Dmax  # directivity array
        dmax_val = float(gain_total[0]) if hasattr(gain_total, '__len__') else float(gain_total)

        if dmax_val <= 0 or not np.isfinite(dmax_val):
            print(f"  WARNING: Dmax={dmax_val}, NF2FF returned no valid directivity")
            print(f"  This may indicate very low radiation at {freq_hz/1e6:.0f} MHz")
            return _fallback

        e_norm = nf2ff_result.E_norm[0]  # first frequency
        e_norm_max = np.max(np.abs(e_norm)**2)

        if e_norm_max <= 0 or not np.isfinite(e_norm_max):
            print(f"  WARNING: E_norm is zero/invalid, no valid radiation pattern")
            return _fallback

        # E-plane: phi=0 index
        e_plane_gain = 10 * np.log10(np.maximum(
            np.abs(e_norm[:, 0])**2 / e_norm_max, 1e-12
        )) + 10 * np.log10(dmax_val)

        # H-plane: theta=90 index (index 18 for 5-degree steps)
        theta_90_idx = 18
        if theta_90_idx < e_norm.shape[0]:
            h_plane_gain = 10 * np.log10(np.maximum(
                np.abs(e_norm[theta_90_idx, :])**2 / e_norm_max, 1e-12
            )) + 10 * np.log10(dmax_val)
        else:
            h_plane_gain = np.full(len(phi_vals), -30.0)

        gain_max = 10 * np.log10(dmax_val)
        # Mean gain estimate from E_norm
        gain_linear_total = np.mean(np.abs(e_norm)**2) / e_norm_max * dmax_val
        gain_mean = 10 * np.log10(max(gain_linear_total, 1e-12))

        # Replace any remaining NaN/Inf with -30
        e_plane_gain = np.where(np.isfinite(e_plane_gain), e_plane_gain, -30.0)
        h_plane_gain = np.where(np.isfinite(h_plane_gain), h_plane_gain, -30.0)
        if not np.isfinite(gain_max):
            gain_max = -30.0
        if not np.isfinite(gain_mean):
            gain_mean = -30.0

        # Efficiency from radiated power / input power
        p_rad = nf2ff_result.Prad[0]
        # Get accepted power from port
        freqs_port = np.array([freq_hz])
        port.CalcPort(sim_path, freqs_port)
        p_in = np.real(0.5 * port.uf_tot * np.conj(port.if_tot))[0]
        if abs(p_in) > 0:
            efficiency = abs(p_rad / p_in)
            efficiency = min(efficiency, 1.0)
        else:
            efficiency = 0.0

        print(f"  Peak gain (directivity): {gain_max:.1f} dBi")
        print(f"  Mean gain: {gain_mean:.1f} dBi")
        print(f"  Estimated efficiency: {efficiency*100:.0f}%")

        return {
            'theta': theta_vals,
            'phi': phi_vals,
            'e_plane_gain': e_plane_gain,
            'h_plane_gain': h_plane_gain,
            'gain_max': gain_max,
            'gain_mean': gain_mean,
            'efficiency': efficiency,
            'freq_hz': freq_hz,
        }
    except Exception as e:
        print(f"  WARNING: Radiation pattern failed: {e}")
        return _fallback
    finally:
        os.chdir(saved_cwd)
        try:
            shutil.rmtree(sim_path)
        except Exception:
            pass


def run_full_simulation(params=None, output_dir='results', quick=False):
    """Run complete simulation for both bands using openEMS.

    Same interface as simulate.py version.

    Args:
        params: optional dict of parameter overrides
        output_dir: directory for output files
        quick: if True, use fewer frequency points

    Returns:
        dict with results for each band
    """
    from utils import (
        plot_s11, plot_vswr, plot_smith, plot_radiation_pattern,
        plot_antenna_geometry, print_summary,
    )

    # Build geometry
    dual = DualBandIFA(params)
    geom = dual.generate()

    # Plot geometry
    plot_antenna_geometry(
        dual.lowband.trace_2d,
        dual.highband.trace_2d,
        save_path=os.path.join(output_dir, 'antenna_layout.png')
    )

    if quick:
        freqs = np.linspace(SIM_FREQ_MIN, SIM_FREQ_MAX, 101)
    else:
        freqs = SIM_FREQUENCIES

    results = {}

    for band in ['lowband', 'highband']:
        band_dir = os.path.join(output_dir, band)
        os.makedirs(band_dir, exist_ok=True)

        targets = TARGETS[band]
        band_label = f"{'Low' if band == 'lowband' else 'High'} Band"

        # S11 sweep
        s11_result = simulate_s11(dual, band, freqs)

        # Find resonance frequency for radiation pattern
        # Use actual resonance (best S11) instead of band center for more
        # meaningful radiation data, especially for LB which resonates
        # far from band center.
        res_idx = np.argmin(s11_result['s11_db'])
        res_freq = s11_result['freqs'][res_idx]
        print(f"\n  Resonance at {res_freq/1e6:.0f} MHz (S11={s11_result['s11_db'][res_idx]:.1f} dB)")
        print(f"  Using resonance frequency for radiation pattern")

        # Radiation pattern at resonance frequency
        rad_result = simulate_radiation(dual, band, freq_hz=res_freq)

        # Generate plots
        plot_s11(s11_result['freqs'], s11_result['s11_complex'],
                 band_label, targets,
                 save_path=os.path.join(band_dir, 's11.png'))

        plot_vswr(s11_result['freqs'], s11_result['s11_complex'],
                  band_label, targets,
                  save_path=os.path.join(band_dir, 'vswr.png'))

        plot_smith(s11_result['s11_complex'], band_label,
                   freqs=s11_result['freqs'],
                   save_path=os.path.join(band_dir, 'smith.png'))

        # Only plot radiation patterns if data is valid (not all -30 dB)
        if rad_result['gain_max'] > -25:
            plot_radiation_pattern(
                rad_result['theta'], rad_result['e_plane_gain'],
                band_label, rad_result['freq_hz'] / 1e9, plane='E',
                save_path=os.path.join(band_dir, 'pattern_e.png'))

            plot_radiation_pattern(
                rad_result['phi'], rad_result['h_plane_gain'],
                band_label, rad_result['freq_hz'] / 1e9, plane='H',
                save_path=os.path.join(band_dir, 'pattern_h.png'))
        else:
            print(f"  Skipping radiation plots (gain too low: {rad_result['gain_max']:.1f} dBi)")

        # Print summary
        print_summary(band, s11_result['freqs'], s11_result['s11_complex'],
                      gain_dbi=rad_result['gain_max'],
                      efficiency=rad_result['efficiency'])

        results[band] = {
            's11': s11_result,
            'radiation': rad_result,
            'geometry': geom[band],
        }

        # Save numerical data
        np.savez(os.path.join(band_dir, 'data.npz'),
                 freqs=s11_result['freqs'],
                 s11_db=s11_result['s11_db'],
                 vswr=s11_result['vswr'],
                 z_real=np.real(s11_result['z_complex']),
                 z_imag=np.imag(s11_result['z_complex']),
                 gain_max=rad_result['gain_max'],
                 efficiency=rad_result['efficiency'])

    results['antenna'] = dual

    # Generate PDF report
    try:
        from report import generate_report
        generate_report(results_dir=output_dir)
    except Exception as e:
        print(f"WARNING: Report generation failed: {e}")

    return results


def build_openems_branched(antenna, sim_path, fast=False, explore=False,
                           gnd_clearance=None, pad=None):
    """Build FDTD model for a BranchedIFA — single feed, both arms in one model.

    Args:
        antenna: BranchedIFA instance (geometry already generated)
        sim_path: directory for FDTD simulation files
        fast: fewer timesteps
        explore: minimal timesteps
        gnd_clearance: override ground clearance (mm)

    Returns:
        (fdtd, port, nf2ff_box) tuple
    """
    CSXCAD_mod, ContinuousStructure, openEMS_mod, OpenEMS_Class = _require_openems()

    f_min = SIM_FREQ_MIN
    f_max = SIM_FREQ_MAX

    sub_t = SUBSTRATE_THICKNESS
    cu_t = COPPER_THICKNESS

    if explore:
        max_ts = OPENEMS_EXPLORE_TIMESTEPS
    elif fast:
        max_ts = OPENEMS_FAST_TIMESTEPS
    else:
        max_ts = OPENEMS_MAX_TIMESTEPS

    fdtd = OpenEMS_Class(
        EndCriteria=OPENEMS_END_CRITERIA,
        NrTS=max_ts,
    )
    fdtd.SetBoundaryCond(OPENEMS_BC)
    fdtd.SetGaussExcite((f_min + f_max) / 2, (f_max - f_min) / 2)

    CSX = ContinuousStructure()
    fdtd.SetCSX(CSX)

    # Materials
    f_center = (f_min + f_max) / 2
    eps0 = 8.854187817e-12
    kappa_fr4 = 2 * np.pi * f_center * eps0 * SUBSTRATE_ER * SUBSTRATE_TAND
    fr4 = CSX.AddMaterial('FR4')
    fr4.SetMaterialProperty(epsilon=SUBSTRATE_ER, kappa=kappa_fr4)
    copper = CSX.AddMetal('Copper')

    # PCB footprint
    pcb_x_min = -GND_HALF_WIDTH
    pcb_x_max = GND_HALF_WIDTH
    pcb_y_min = -GND_EXTENSION
    pcb_y_max = BOARD_RADIUS

    gc = gnd_clearance if gnd_clearance is not None else antenna.gnd_clearance

    # 1. FR4 substrate
    fr4.AddBox([pcb_x_min, pcb_y_min, 0], [pcb_x_max, pcb_y_max, sub_t], priority=1)

    # 2. Bottom ground plane at z=0
    copper.AddBox(
        [pcb_x_min, pcb_y_min, 0],
        [pcb_x_max, 0, 0],
        priority=10,
    )

    # 3. Top ground plane at z=sub_t (pulled back by clearance)
    copper.AddBox(
        [pcb_x_min, pcb_y_min, sub_t],
        [pcb_x_max, -gc, sub_t],
        priority=10,
    )

    # 4. Shorting via
    via_size = 0.4
    short_x = antenna.short_x
    copper.AddBox(
        [short_x - via_size, -via_size, 0],
        [short_x + via_size, via_size, sub_t],
        priority=15,
    )

    # 5. ALL antenna traces (LB arm + HB branch + stubs) at z=sub_t
    trace_half_w = antenna.trace_width / 2.0
    for seg in antenna._segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.05:
            continue

        if abs(x2 - x1) < 0.01:
            copper.AddBox(
                [x1 - trace_half_w, min(y1, y2), sub_t],
                [x1 + trace_half_w, max(y1, y2), sub_t],
                priority=15,
            )
        elif abs(y2 - y1) < 0.01:
            copper.AddBox(
                [min(x1, x2), y1 - trace_half_w, sub_t],
                [max(x1, x2), y1 + trace_half_w, sub_t],
                priority=15,
            )
        else:
            dx = x2 - x1
            dy = y2 - y1
            seg_len = np.sqrt(dx**2 + dy**2)
            nx = -dy / seg_len * trace_half_w
            ny = dx / seg_len * trace_half_w
            points_x = [x1 + nx, x2 + nx, x2 - nx, x1 - nx]
            points_y = [y1 + ny, y2 + ny, y2 - ny, y1 - ny]
            points = np.array([points_x, points_y])
            copper.AddLinPoly(points, 'z', sub_t, 0, priority=15)

    # 6. Capacitive load pad (PEC box at z=sub_t)
    if antenna.cap_load_rect is not None:
        cx, cy, w, h = antenna.cap_load_rect
        copper.AddBox(
            [cx - w / 2, cy - h / 2, sub_t],
            [cx + w / 2, cy + h / 2, sub_t],
            priority=15,
        )

    # 7. Feed port — lumped port through substrate, sized to match trace width
    feed_x = antenna.feed_x
    port_start = [feed_x - trace_half_w, -trace_half_w, 0]
    port_stop = [feed_x + trace_half_w, trace_half_w, sub_t]
    port = fdtd.AddLumpedPort(
        port_nr=1, R=Z0,
        start=port_start, stop=port_stop,
        p_dir='z', excite=1.0,
    )

    # Mesh
    _generate_mesh_branched(CSX, fdtd, antenna, sub_t, cu_t, f_max,
                             pcb_x_min, pcb_x_max, pcb_y_min, pcb_y_max,
                             pad_override=pad)

    # NF2FF box
    nf2ff = fdtd.CreateNF2FFBox()

    os.makedirs(sim_path, exist_ok=True)
    CSX.Write2XML(os.path.join(sim_path, 'model.xml'))

    return fdtd, port, nf2ff


def _generate_mesh_branched(CSX, fdtd, antenna, sub_t, cu_t, f_max,
                             pcb_x_min, pcb_x_max, pcb_y_min, pcb_y_max,
                             pad_override=None):
    """Generate adaptive FDTD mesh for branched IFA."""
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)

    lambda_min_mm = C0 / f_max * 1000.0
    base_res = lambda_min_mm / OPENEMS_MESH_RES
    # Default 15mm for S11; use pad_override=40+ for NF2FF enclosure
    pad = pad_override if pad_override is not None else 15.0

    # Z mesh — extend arange past target to avoid tiny boundary cells
    # after SmoothMeshLines (matching _generate_mesh approach).
    z_lines = []
    n_sub_z = max(4, int(np.ceil(sub_t / (base_res / 4))))
    z_sub = np.linspace(0, sub_t, n_sub_z + 1)
    z_lines = list(z_sub)
    z_top = sub_t + pad + 20
    z_above = np.arange(sub_t + base_res, z_top + base_res, base_res)
    z_lines.extend(z_above.tolist())
    z_bot = -pad - 5
    z_below = np.arange(-base_res, z_bot - base_res, -base_res)
    z_lines.extend(z_below.tolist())

    # X mesh — extend arange past target to avoid tiny boundary cells
    x_lines = []
    pcb_res = min(base_res, 3.0)
    x_pcb = np.arange(pcb_x_min, pcb_x_max + pcb_res / 2, pcb_res)
    x_lines.extend(x_pcb.tolist())
    x_lines.extend([pcb_x_min, pcb_x_max])
    x_pad_left = np.arange(pcb_x_min - base_res, pcb_x_min - pad - 5 - base_res, -base_res)
    x_pad_right = np.arange(pcb_x_max + base_res, pcb_x_max + pad + 5 + base_res, base_res)
    x_lines.extend(x_pad_left.tolist())
    x_lines.extend(x_pad_right.tolist())

    # Fine mesh near feed and short
    feed_x = antenna.feed_x
    short_x = antenna.short_x
    feed_fine = np.arange(feed_x - 3, feed_x + 3.1, 0.5)
    x_lines.extend(feed_fine.tolist())
    short_fine = np.arange(short_x - 2, short_x + 2.1, 0.5)
    x_lines.extend(short_fine.tolist())

    # Fine mesh around HB branch endpoint (if exists)
    if antenna._hb_segments:
        hb_end = antenna._hb_segments[-1]
        hb_fine_x = np.arange(hb_end[2] - 2, hb_end[2] + 2.1, 0.5)
        x_lines.extend(hb_fine_x.tolist())

    # Y mesh
    y_lines = []
    y_gnd = np.arange(pcb_y_min, 0, pcb_res)
    y_lines.extend(y_gnd.tolist())
    antenna_res = min(base_res, 2.0)
    y_ant = np.arange(0, BOARD_RADIUS + 1, antenna_res)
    y_lines.extend(y_ant.tolist())
    y_lines.extend([pcb_y_min, 0, BOARD_RADIUS])
    y_pad_bottom = np.arange(pcb_y_min - base_res, pcb_y_min - pad - 5 - base_res, -base_res)
    y_pad_top = np.arange(BOARD_RADIUS + base_res, BOARD_RADIUS + pad + 5 + base_res, base_res)
    y_lines.extend(y_pad_bottom.tolist())
    y_lines.extend(y_pad_top.tolist())

    feed_fine_y = np.arange(-2, 3.1, 0.5)
    y_lines.extend(feed_fine_y.tolist())
    y_lines.extend([-0.5, -0.25, 0.0, 0.25])

    # Fine mesh near cap load
    if antenna.cap_load_rect is not None:
        cx, cy, w, h = antenna.cap_load_rect
        cap_fine_x = np.arange(cx - w / 2 - 1, cx + w / 2 + 1.1, 0.5)
        cap_fine_y = np.arange(cy - h / 2 - 1, cy + h / 2 + 1.1, 0.5)
        x_lines.extend(cap_fine_x.tolist())
        y_lines.extend(cap_fine_y.tolist())

    from CSXCAD import CSRectGrid
    x_lines = sorted(set(x_lines))
    y_lines = sorted(set(y_lines))
    z_lines = sorted(set(z_lines))

    mesh.AddLine('x', x_lines)
    mesh.AddLine('y', y_lines)
    mesh.AddLine('z', z_lines)

    mesh.SmoothMeshLines('x', base_res, ratio=1.4)
    mesh.SmoothMeshLines('y', base_res, ratio=1.4)
    mesh.SmoothMeshLines('z', base_res / 2, ratio=1.3)

    # Fix tiny cells at domain boundaries caused by smoothing misalignment.
    # SmoothMeshLines can leave a sub-0.1mm sliver between the last smoothed
    # line and a forced boundary, which shrinks the CFL timestep by 5-10x.
    _min_cell = 0.15  # mm — well below smallest intentional cell (0.25 mm)
    for ax in range(3):
        lines = np.array(mesh.GetLines(ax))
        trimmed = False
        while len(lines) > 2 and (lines[1] - lines[0]) < _min_cell:
            lines = lines[1:]
            trimmed = True
        while len(lines) > 2 and (lines[-1] - lines[-2]) < _min_cell:
            lines = lines[:-1]
            trimmed = True
        if trimmed:
            mesh.ClearLines(ax)
            mesh.AddLine(['x', 'y', 'z'][ax], lines.tolist())


def simulate_s11_branched(antenna, freqs=None, verbose=True, fast=False,
                           explore=False, gnd_clearance=None):
    """Broadband S11 for branched IFA — one FDTD run covers both bands.

    Args:
        antenna: BranchedIFA (geometry generated)
        freqs: output frequency array (default: SIM_FREQUENCIES)
        verbose: print progress
        fast: fewer timesteps
        explore: minimal timesteps
        gnd_clearance: override ground clearance

    Returns:
        dict: freqs, z_complex, s11_complex, s11_db, vswr
    """
    from utils import s11_db as calc_s11_db, s11_to_vswr

    if freqs is None:
        freqs = SIM_FREQUENCIES

    if verbose:
        mode = "explore" if explore else ("fast" if fast else "full")
        print(f"\nSimulating branched IFA S11 (openEMS FDTD, {mode} mode)...")
        print(f"  Frequency range: {freqs[0]/1e6:.0f} - {freqs[-1]/1e6:.0f} MHz")

    sim_path = tempfile.mkdtemp(prefix='openems_branched_')
    saved_cwd = os.getcwd()

    try:
        fdtd, port, nf2ff = build_openems_branched(
            antenna, sim_path, fast=fast, explore=explore,
            gnd_clearance=gnd_clearance)

        if verbose:
            print(f"  Running FDTD simulation...")
            fdtd.Run(sim_path, verbose=1)
        else:
            fdtd.Run(sim_path, verbose=0)

        port.CalcPort(sim_path, freqs)

        z_complex = port.uf_tot / port.if_tot
        s11_complex = port.uf_ref / port.uf_inc
        s11_vals_db = calc_s11_db(s11_complex)
        vswr_vals = s11_to_vswr(s11_complex)

        if verbose:
            n_freq = len(freqs)
            for idx in [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]:
                z = z_complex[idx]
                print(f"  f={freqs[idx]/1e6:.0f} MHz: Z={np.real(z):.1f}+j{np.imag(z):.1f}, "
                      f"S11={s11_vals_db[idx]:.1f} dB")

        return {
            'freqs': freqs,
            'z_complex': z_complex,
            's11_complex': s11_complex,
            's11_db': s11_vals_db,
            'vswr': vswr_vals,
        }
    finally:
        os.chdir(saved_cwd)
        try:
            shutil.rmtree(sim_path)
        except Exception:
            pass


def simulate_radiation_branched(antenna, freq_hz, verbose=True):
    """Radiation pattern for branched IFA at a single frequency.

    Args:
        antenna: BranchedIFA (geometry generated)
        freq_hz: frequency in Hz

    Returns:
        dict: theta, phi, e_plane_gain, h_plane_gain, gain_max, gain_mean,
              efficiency, freq_hz
    """
    if verbose:
        print(f"\nSimulating branched IFA radiation at {freq_hz/1e6:.0f} MHz...")

    sim_path = tempfile.mkdtemp(prefix='openems_rad_branched_')
    saved_cwd = os.getcwd()

    theta_vals = np.arange(0, 181, 5)
    phi_vals = np.arange(0, 361, 5)

    _fallback = {
        'theta': theta_vals, 'phi': phi_vals,
        'e_plane_gain': np.full(len(theta_vals), -30.0),
        'h_plane_gain': np.full(len(phi_vals), -30.0),
        'gain_max': -30.0, 'gain_mean': -30.0,
        'efficiency': 0.0, 'freq_hz': freq_hz,
    }

    try:
        fdtd, port, nf2ff = build_openems_branched(antenna, sim_path, pad=40)

        fdtd.Run(sim_path, verbose=0)

        nf2ff_result = nf2ff.CalcNF2FF(
            sim_path, freq_hz, theta=theta_vals, phi=phi_vals)

        gain_total = nf2ff_result.Dmax
        dmax_val = float(gain_total[0]) if hasattr(gain_total, '__len__') else float(gain_total)

        if dmax_val <= 0 or not np.isfinite(dmax_val):
            if verbose:
                print(f"  WARNING: Dmax={dmax_val}, NF2FF returned no valid directivity")
            return _fallback

        e_norm = nf2ff_result.E_norm[0]
        e_norm_max = np.max(np.abs(e_norm)**2)

        if e_norm_max <= 0 or not np.isfinite(e_norm_max):
            if verbose:
                print(f"  WARNING: E_norm is zero/invalid")
            return _fallback

        e_plane_gain = 10 * np.log10(np.maximum(
            np.abs(e_norm[:, 0])**2 / e_norm_max, 1e-12
        )) + 10 * np.log10(dmax_val)

        theta_90_idx = 18
        if theta_90_idx < e_norm.shape[0]:
            h_plane_gain = 10 * np.log10(np.maximum(
                np.abs(e_norm[theta_90_idx, :])**2 / e_norm_max, 1e-12
            )) + 10 * np.log10(dmax_val)
        else:
            h_plane_gain = np.full(len(phi_vals), -30.0)

        gain_max = 10 * np.log10(dmax_val)
        gain_linear_total = np.mean(np.abs(e_norm)**2) / e_norm_max * dmax_val
        gain_mean = 10 * np.log10(max(gain_linear_total, 1e-12))

        e_plane_gain = np.where(np.isfinite(e_plane_gain), e_plane_gain, -30.0)
        h_plane_gain = np.where(np.isfinite(h_plane_gain), h_plane_gain, -30.0)
        if not np.isfinite(gain_max):
            gain_max = -30.0
        if not np.isfinite(gain_mean):
            gain_mean = -30.0

        p_rad = nf2ff_result.Prad[0]
        freqs_port = np.array([freq_hz])
        port.CalcPort(sim_path, freqs_port)
        p_in = np.real(0.5 * port.uf_tot * np.conj(port.if_tot))[0]
        if abs(p_in) > 0:
            efficiency = min(abs(p_rad / p_in), 1.0)
        else:
            efficiency = 0.0

        if verbose:
            print(f"  Peak gain: {gain_max:.1f} dBi, efficiency: {efficiency*100:.0f}%")

        return {
            'theta': theta_vals, 'phi': phi_vals,
            'e_plane_gain': e_plane_gain, 'h_plane_gain': h_plane_gain,
            'gain_max': gain_max, 'gain_mean': gain_mean,
            'efficiency': efficiency, 'freq_hz': freq_hz,
        }
    except Exception as e:
        print(f"  WARNING: Radiation pattern failed: {e}")
        return _fallback
    finally:
        os.chdir(saved_cwd)
        try:
            shutil.rmtree(sim_path)
        except Exception:
            pass


def run_full_simulation_branched(params=None, output_dir='results', quick=False):
    """Run complete simulation for branched IFA.

    One FDTD run gives S11 across both bands. Radiation patterns at each
    resonance frequency.

    Returns:
        dict with results
    """
    from utils import (
        plot_s11, plot_vswr, plot_smith, plot_radiation_pattern,
        plot_antenna_geometry_branched, plot_combined_s11_branched,
        print_summary_branched,
    )
    from config import LB_FREQ_MIN, LB_FREQ_MAX, HB_FREQ_MIN, HB_FREQ_MAX

    # Build geometry
    antenna = BranchedIFA(params)
    antenna.generate_geometry()

    violations = antenna.validate_geometry()
    if violations:
        print(f"WARNING: {len(violations)} boundary violations")
    gap = antenna.check_gap()
    print(f"LB arm: {antenna.actual_lb_length:.1f}mm, HB branch: {antenna.actual_hb_length:.1f}mm")
    if gap < 3.0:
        print(f"WARNING: Min LB-HB gap is {gap:.1f}mm (target: >= 3mm)")

    os.makedirs(output_dir, exist_ok=True)

    # Plot geometry
    plot_antenna_geometry_branched(
        antenna.lb_trace_2d, antenna.hb_trace_2d,
        cap_rect=antenna.cap_load_rect,
        save_path=os.path.join(output_dir, 'antenna_layout.png')
    )

    if quick:
        freqs = np.linspace(SIM_FREQ_MIN, SIM_FREQ_MAX, 101)
    else:
        freqs = SIM_FREQUENCIES

    # Single S11 sweep
    s11_result = simulate_s11_branched(antenna, freqs)

    # Combined S11 plot
    plot_combined_s11_branched(
        s11_result,
        save_path=os.path.join(output_dir, 's11_combined.png')
    )

    # Also plot per-band views
    for band, label, fmin, fmax in [
        ('lowband', 'Low Band (Branched LB)', LB_FREQ_MIN, LB_FREQ_MAX),
        ('highband', 'High Band (Branched HB)', HB_FREQ_MIN, HB_FREQ_MAX),
    ]:
        band_dir = os.path.join(output_dir, band)
        os.makedirs(band_dir, exist_ok=True)

        targets = TARGETS[band]

        plot_s11(s11_result['freqs'], s11_result['s11_complex'],
                 label, targets,
                 save_path=os.path.join(band_dir, 's11.png'))

        plot_vswr(s11_result['freqs'], s11_result['s11_complex'],
                  label, targets,
                  save_path=os.path.join(band_dir, 'vswr.png'))

        plot_smith(s11_result['s11_complex'], label,
                   freqs=s11_result['freqs'],
                   save_path=os.path.join(band_dir, 'smith.png'))

    # Find resonances for radiation patterns
    results = {'s11': s11_result, 'antenna': antenna}

    for band, label, fmin, fmax in [
        ('lowband', 'Low Band', LB_FREQ_MIN, LB_FREQ_MAX),
        ('highband', 'High Band', HB_FREQ_MIN, HB_FREQ_MAX),
    ]:
        band_dir = os.path.join(output_dir, band)
        band_mask = (s11_result['freqs'] >= fmin) & (s11_result['freqs'] <= fmax)

        if np.any(band_mask):
            band_s11 = s11_result['s11_db'][band_mask]
            band_freqs = s11_result['freqs'][band_mask]
            res_idx = np.argmin(band_s11)
            res_freq = band_freqs[res_idx]
            print(f"\n  {label} resonance at {res_freq/1e6:.0f} MHz "
                  f"(S11={band_s11[res_idx]:.1f} dB)")
        else:
            # Use overall best
            res_idx = np.argmin(s11_result['s11_db'])
            res_freq = s11_result['freqs'][res_idx]

        rad_result = simulate_radiation_branched(antenna, freq_hz=res_freq)

        if rad_result['gain_max'] > -25:
            plot_radiation_pattern(
                rad_result['theta'], rad_result['e_plane_gain'],
                label, rad_result['freq_hz'] / 1e9, plane='E',
                save_path=os.path.join(band_dir, 'pattern_e.png'))
            plot_radiation_pattern(
                rad_result['phi'], rad_result['h_plane_gain'],
                label, rad_result['freq_hz'] / 1e9, plane='H',
                save_path=os.path.join(band_dir, 'pattern_h.png'))

        results[band] = {
            's11': s11_result,
            'radiation': rad_result,
        }

        # Save numerical data
        np.savez(os.path.join(band_dir, 'data.npz'),
                 freqs=s11_result['freqs'],
                 s11_db=s11_result['s11_db'],
                 vswr=s11_result['vswr'],
                 z_real=np.real(s11_result['z_complex']),
                 z_imag=np.imag(s11_result['z_complex']),
                 gain_max=rad_result['gain_max'],
                 efficiency=rad_result['efficiency'])

    # Print summary
    print_summary_branched(s11_result)

    # Generate report
    try:
        from report import generate_report_branched
        generate_report_branched(results_dir=output_dir, params=params)
    except Exception as e:
        print(f"WARNING: Report generation failed: {e}")

    return results


def validate_geometry_for_branched(antenna):
    """Validate branched IFA geometry for openEMS simulation.

    Returns (ok, reason) tuple.
    """
    if not antenna._segments:
        return False, "No segments generated"

    # Check feed stub exists
    feed_found = False
    for seg in antenna._segments:
        x1, y1, x2, y2 = seg
        if (abs(x1 - antenna.feed_x) < 0.01 and abs(y1) < 0.01 and
                abs(x2 - antenna.feed_x) < 0.01 and y2 > 0):
            feed_found = True
            break

    if not feed_found:
        return False, "Feed wire not found in geometry"

    # Check short stub exists
    short_found = False
    for seg in antenna._segments:
        x1, y1, x2, y2 = seg
        if (abs(x1 - antenna.short_x) < 0.01 and abs(y1) < 0.01 and
                abs(x2 - antenna.short_x) < 0.01 and y2 > 0):
            short_found = True
            break

    if not short_found:
        return False, "Short stub not found in geometry"

    # Check segments have reasonable lengths
    for seg in antenna._segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.01:
            return False, f"Degenerate segment (length={length:.4f}mm)"

    return True, "OK"


def validate_geometry_for_openems(antenna, band):
    """Validate that the antenna geometry is suitable for openEMS simulation.

    Returns (ok, reason) tuple.
    """
    ifa = antenna.lowband if band == 'lowband' else antenna.highband

    if not ifa._segments:
        return False, "No segments generated"

    # Check that feed point exists in the segments
    feed_found = False
    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        if (abs(x1 - ifa.feed_x) < 0.01 and abs(y1) < 0.01 and
                abs(x2 - ifa.feed_x) < 0.01 and y2 > 0):
            feed_found = True
            break

    if not feed_found:
        return False, "Feed wire not found in geometry"

    # Check that short point exists
    short_found = False
    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        if (abs(x1 - ifa.short_x) < 0.01 and abs(y1) < 0.01 and
                abs(x2 - ifa.short_x) < 0.01 and y2 > 0):
            short_found = True
            break

    if not short_found:
        return False, "Short stub not found in geometry"

    # Check that segments have reasonable lengths
    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.01:
            return False, f"Degenerate segment found (length={length:.4f}mm)"

    return True, "OK"


def _subprocess_s11_worker(params_dict, band, freqs_list, result_dict):
    """Worker function for subprocess-safe S11 simulation."""
    try:
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

        freqs = np.array(freqs_list)
        dual = DualBandIFA(params_dict)
        dual.generate()

        ok, reason = validate_geometry_for_openems(dual, band)
        if not ok:
            result_dict['error'] = reason
            return

        result = simulate_s11(dual, band, freqs, verbose=False)

        result_dict['s11_db'] = result['s11_db'].tolist()
        result_dict['vswr'] = result['vswr'].tolist()
        result_dict['z_real'] = np.real(result['z_complex']).tolist()
        result_dict['z_imag'] = np.imag(result['z_complex']).tolist()
        result_dict['ok'] = True
    except Exception as e:
        result_dict['error'] = str(e)
    finally:
        sys.stdout = old_stdout
        devnull.close()


def simulate_s11_safe(params_dict, band, freqs, timeout=120):
    """Subprocess-safe S11 simulation. Returns result dict or None on failure.

    Longer default timeout than NEC2 version (120s vs 30s) since FDTD takes longer.
    """
    from utils import impedance_to_s11, s11_db as calc_s11_db, s11_to_vswr

    mgr = multiprocessing.Manager()
    result_dict = mgr.dict()
    result_dict['ok'] = False

    p = multiprocessing.Process(
        target=_subprocess_s11_worker,
        args=(params_dict, band, freqs.tolist(), result_dict)
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        return None

    if p.exitcode != 0:
        return None

    if not result_dict.get('ok', False):
        return None

    return {
        'freqs': freqs,
        's11_db': np.array(result_dict['s11_db']),
        'vswr': np.array(result_dict['vswr']),
        'z_real': np.array(result_dict['z_real']),
        'z_imag': np.array(result_dict['z_imag']),
    }


if __name__ == '__main__':
    import argparse
    from config import ANTENNA_TOPOLOGY

    parser = argparse.ArgumentParser(description='IFA antenna simulation (openEMS FDTD)')
    parser.add_argument('--quick', action='store_true',
                        help='Use fewer frequency points for quick iteration')
    parser.add_argument('--band', choices=['lowband', 'highband', 'both'],
                        default='both', help='Which band to simulate (dual_ifa only)')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--topology', choices=['dual_ifa', 'branched'],
                        default=None, help='Antenna topology (default: from config)')
    args = parser.parse_args()

    if not _check_openems():
        print("ERROR: openEMS is not installed.")
        sys.exit(1)

    topology = args.topology or ANTENNA_TOPOLOGY

    print("=" * 60)
    if topology == 'branched':
        print("  Branched IFA Antenna Simulation")
    else:
        print("  Dual-Band IFA Antenna Simulation")
    print("  Using openEMS (FDTD with FR4 substrate)")
    print("=" * 60)

    if topology == 'branched':
        results = run_full_simulation_branched(output_dir=args.output, quick=args.quick)
    else:
        results = run_full_simulation(output_dir=args.output, quick=args.quick)

    print("\nSimulation complete! Check the results/ directory for plots.")
