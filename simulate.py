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
NEC2-based simulation for the dual-band IFA antenna system.

Simulates each antenna independently using necpp (NEC2 engine):
- Frequency sweep for S11, VSWR
- Radiation pattern at center frequency
- Gain and efficiency estimation

NEC2 is a Method of Moments (MoM) solver, well-suited for wire antennas.
For PCB traces, we approximate traces as wires with equivalent radius.
"""

import os
import sys
import json
import multiprocessing
import numpy as np
import necpp

from config import (
    BOARD_RADIUS, GND_EXTENSION, GND_HALF_WIDTH, GND_GRID_WIRE_RADIUS,
    SUBSTRATE_ER, SUBSTRATE_THICKNESS,
    COPPER_THICKNESS, Z0, C0, NEC_SEGMENTS_PER_LAMBDA, NEC_GND_SEGMENTS,
    SIM_FREQ_MIN, SIM_FREQ_MAX, SIM_NUM_FREQ, SIM_FREQUENCIES, TARGETS,
    LB_FREQ_CENTER, HB_FREQ_CENTER,
)
from antenna_model import DualBandIFA, MM_TO_M, trace_width_to_wire_radius, segments_for_length
from utils import (
    impedance_to_s11, s11_db, s11_to_vswr,
    plot_s11, plot_vswr, plot_smith, plot_radiation_pattern,
    plot_antenna_geometry, print_summary,
)


def _build_ground_grid(nec, start_tag, feed_x_mm, short_x_mm):
    """Build a proper wire-grid ground plane for the rectangular PCB body.

    The grid uses the real PCB dimensions (62mm wide x 88mm deep) with
    cell-by-cell construction so that all wires at grid intersections
    share exact endpoints (galvanic connection required by NEC2).

    The top wire at z=0 is additionally split at the antenna junction
    points (feed_x, short_x) so the IFA stubs share endpoints with ground.

    Vertical ground wires are placed at x-positions with >=3mm clearance
    from junction points to avoid NEC2 singularity from coincident wires.

    Args:
        nec: NEC2 context
        start_tag: first available tag ID
        feed_x_mm: feed stub x position in mm
        short_x_mm: short stub x position in mm

    Returns:
        next available tag ID
    """
    tag_id = start_tag
    gnd_wr_m = GND_GRID_WIRE_RADIUS * MM_TO_M
    min_seg_len = 8 * GND_GRID_WIRE_RADIUS  # mm, NEC2 thin-wire constraint

    # X positions for vertical grid wires (mm), chosen to:
    # - span the full PCB width (-31 to +31)
    # - keep max spacing < lambda/10 at 2170 MHz (13.8mm)
    # - avoid junction x positions (feed_x, short_x) by >= 3mm
    x_grid = [-31.0, -22.0, -14.0, -0.5, 10.0, 22.0, 31.0]

    # Remove any grid x that is within 3mm of a junction point
    junction_xs_mm = sorted(set([feed_x_mm, short_x_mm]))
    x_grid_safe = []
    for xg in x_grid:
        too_close = False
        for jx in junction_xs_mm:
            if abs(xg - jx) < 3.0:
                too_close = True
                break
        if not too_close:
            x_grid_safe.append(xg)
    x_grid = x_grid_safe

    # Z positions for horizontal wires (mm), 0 to -88 in ~11mm steps
    z_levels = [0.0, -11.0, -22.0, -33.0, -44.0, -55.0, -66.0, -77.0, -88.0]

    # --- Top wire at z=0: split at BOTH grid x positions AND junction points ---
    top_breakpoints = sorted(set(x_grid + junction_xs_mm))
    # Ensure full extent
    if top_breakpoints[0] > x_grid[0]:
        top_breakpoints.insert(0, x_grid[0])
    if top_breakpoints[-1] < x_grid[-1]:
        top_breakpoints.append(x_grid[-1])
    top_breakpoints = sorted(set(top_breakpoints))

    for i in range(len(top_breakpoints) - 1):
        x1_mm = top_breakpoints[i]
        x2_mm = top_breakpoints[i + 1]
        length_mm = abs(x2_mm - x1_mm)
        if length_mm < 0.1:
            continue
        # Multi-segment for longer pieces, respect thin-wire constraint
        n_seg = max(1, min(
            int(length_mm / 5.0),        # target ~5mm segments
            int(length_mm / min_seg_len)  # respect NEC2 constraint
        ))
        necpp.nec_wire(nec, tag_id, n_seg,
                       x1_mm * MM_TO_M, 0, 0,
                       x2_mm * MM_TO_M, 0, 0,
                       gnd_wr_m, 1.0, 1.0)
        tag_id += 1

    # --- Horizontal wires below z=0 (between consecutive grid x positions) ---
    for z_mm in z_levels[1:]:  # skip z=0, already done above
        for i in range(len(x_grid) - 1):
            x1_mm = x_grid[i]
            x2_mm = x_grid[i + 1]
            necpp.nec_wire(nec, tag_id, 1,
                           x1_mm * MM_TO_M, 0, z_mm * MM_TO_M,
                           x2_mm * MM_TO_M, 0, z_mm * MM_TO_M,
                           gnd_wr_m, 1.0, 1.0)
            tag_id += 1

    # --- Vertical wires (at each grid x, between consecutive z levels) ---
    for xg_mm in x_grid:
        for i in range(len(z_levels) - 1):
            z1_mm = z_levels[i]
            z2_mm = z_levels[i + 1]
            necpp.nec_wire(nec, tag_id, 1,
                           xg_mm * MM_TO_M, 0, z1_mm * MM_TO_M,
                           xg_mm * MM_TO_M, 0, z2_mm * MM_TO_M,
                           gnd_wr_m, 1.0, 1.0)
            tag_id += 1

    return tag_id


def build_nec_for_band(antenna, band, freq_hz):
    """Build a NEC2 context for simulating one band's IFA.

    Args:
        antenna: DualBandIFA instance (geometry already generated)
        band: 'lowband' or 'highband'
        freq_hz: reference frequency for segment sizing

    Returns:
        nec: NEC2 context
        feed_tag: tag ID for the feed wire
        feed_seg: segment index for excitation (1-based)
    """
    nec = necpp.nec_create()

    ifa = antenna.lowband if band == 'lowband' else antenna.highband
    wire_radius_m = ifa.wire_radius * MM_TO_M

    tag_id = 1
    feed_tag = None
    feed_seg = None

    # --- Antenna wire segments ---
    for seg in ifa._segments:
        x1_mm, y1_mm, x2_mm, y2_mm = seg
        length_mm = np.sqrt((x2_mm - x1_mm)**2 + (y2_mm - y1_mm)**2)
        if length_mm < 0.1:
            continue

        n_seg = segments_for_length(length_mm, freq_hz, min_seg=3,
                                    wire_radius_mm=ifa.wire_radius)

        # Map 2D antenna coords to NEC2 3D: (x, y_antenna) → (x, 0, z=y_antenna)
        necpp.nec_wire(nec, tag_id, n_seg,
                       x1_mm * MM_TO_M, 0, y1_mm * MM_TO_M,
                       x2_mm * MM_TO_M, 0, y2_mm * MM_TO_M,
                       wire_radius_m, 1.0, 1.0)

        # Identify feed stub: vertical wire from ground (y=0) upward at feed_x
        if (abs(x1_mm - ifa.feed_x) < 0.01 and abs(y1_mm) < 0.01 and
                abs(x2_mm - ifa.feed_x) < 0.01 and y2_mm > 0):
            feed_tag = tag_id
            feed_seg = max(1, n_seg // 2)  # middle segment for best excitation

        tag_id += 1

    # --- Ground plane wire grid ---
    tag_id = _build_ground_grid(nec, tag_id, ifa.feed_x, ifa.short_x)

    if feed_tag is None:
        print(f"WARNING [{band}]: Feed wire not identified! Using tag 2, segment 1")
        feed_tag = 2
        feed_seg = 1

    # Geometry complete — no symmetry
    necpp.nec_geometry_complete(nec, 0)

    return nec, feed_tag, feed_seg


def simulate_s11(antenna, band, freqs=None, verbose=True):
    """Run frequency sweep and calculate S11 for one band.

    Uses NEC2's built-in frequency sweep (nec_fr_card with multiple steps)
    to avoid creating/destroying contexts per frequency point.

    Args:
        antenna: DualBandIFA (geometry generated)
        band: 'lowband' or 'highband'
        freqs: array of frequencies in Hz (default: SIM_FREQUENCIES)
        verbose: print progress info

    Returns:
        dict with keys: freqs, z_complex, s11_complex, s11_db, vswr
    """
    if freqs is None:
        freqs = SIM_FREQUENCIES

    targets = TARGETS[band]
    ref_freq = targets['freq_center']
    n_freq = len(freqs)

    if verbose:
        print(f"\nSimulating {band} S11 sweep ({n_freq} points)...")
        print(f"  Frequency range: {freqs[0]/1e6:.0f} - {freqs[-1]/1e6:.0f} MHz")

    # Build model once
    nec, feed_tag, feed_seg = build_nec_for_band(antenna, band, ref_freq)

    # Voltage excitation
    necpp.nec_ex_card(nec, 0, feed_tag, feed_seg, 0,
                      1.0, 0, 0, 0, 0, 0)

    # Frequency sweep: linear interpolation from f_start to f_end
    f_start_mhz = freqs[0] / 1e6
    f_step_mhz = (freqs[-1] - freqs[0]) / (n_freq - 1) / 1e6 if n_freq > 1 else 0
    necpp.nec_fr_card(nec, 0, n_freq, f_start_mhz, f_step_mhz)

    # Minimal radiation pattern request (needed to trigger solution)
    necpp.nec_rp_card(nec, 0, 1, 1, 0, 0, 0, 0, 0, 90, 90, 0, 0, 0)

    # Extract impedance at each frequency
    z_real = np.zeros(n_freq)
    z_imag = np.zeros(n_freq)
    for i in range(n_freq):
        z_real[i] = necpp.nec_impedance_real(nec, i)
        z_imag[i] = necpp.nec_impedance_imag(nec, i)

    necpp.nec_delete(nec)

    # Print a few sample points
    if verbose:
        for idx in [0, n_freq // 4, n_freq // 2, 3 * n_freq // 4, n_freq - 1]:
            print(f"  f={freqs[idx]/1e6:.0f} MHz: Z={z_real[idx]:.1f}+j{z_imag[idx]:.1f}")

    z_complex = z_real + 1j * z_imag
    s11_complex = impedance_to_s11(z_complex)
    s11_vals_db = s11_db(s11_complex)
    vswr_vals = s11_to_vswr(s11_complex)

    return {
        'freqs': freqs,
        'z_complex': z_complex,
        's11_complex': s11_complex,
        's11_db': s11_vals_db,
        'vswr': vswr_vals,
    }


def simulate_radiation(antenna, band, freq_hz=None):
    """Calculate radiation pattern at a single frequency.

    Args:
        antenna: DualBandIFA (geometry generated)
        band: 'lowband' or 'highband'
        freq_hz: frequency in Hz (default: band center)

    Returns:
        dict with keys: theta, phi, gain_total, gain_max, efficiency
    """
    if freq_hz is None:
        freq_hz = TARGETS[band]['freq_center']

    print(f"\nSimulating {band} radiation pattern at {freq_hz/1e6:.0f} MHz...")

    nec, feed_tag, feed_seg = build_nec_for_band(antenna, band, freq_hz)

    # Voltage excitation
    necpp.nec_ex_card(nec, 0, feed_tag, feed_seg, 0,
                      1.0, 0, 0, 0, 0, 0)

    # Set frequency
    necpp.nec_fr_card(nec, 0, 1, freq_hz / 1e6, 0)

    # Radiation pattern: full sphere
    n_theta = 37   # 0 to 180 in 5° steps
    n_phi = 73     # 0 to 360 in 5° steps
    necpp.nec_rp_card(nec, 0, n_theta, n_phi, 0, 0, 0, 0, 0,
                      5.0, 5.0, 0, 0, 0)

    # Extract gain data
    gain_max = necpp.nec_gain_max(nec, 0)
    gain_mean = necpp.nec_gain_mean(nec, 0)
    gain_min = necpp.nec_gain_min(nec, 0)

    # Extract pattern data for E-plane and H-plane
    theta_vals = np.arange(0, 181, 5)
    phi_vals = np.arange(0, 361, 5)

    # E-plane (phi=0, vary theta)
    e_plane_gain = np.zeros(len(theta_vals))
    for i, theta in enumerate(theta_vals):
        idx = i * n_phi + 0  # phi=0 index
        try:
            e_plane_gain[i] = necpp.nec_gain(nec, 0, idx)
        except Exception:
            e_plane_gain[i] = -30.0

    # H-plane (theta=90, vary phi)
    h_plane_gain = np.zeros(len(phi_vals))
    theta_90_idx = 18  # 90°/5° = 18
    for i, phi in enumerate(phi_vals):
        idx = theta_90_idx * n_phi + i
        try:
            h_plane_gain[i] = necpp.nec_gain(nec, 0, idx)
        except Exception:
            h_plane_gain[i] = -30.0

    # Estimate efficiency from gain data
    # For a lossless antenna, directivity ≈ gain. With FR4 losses,
    # efficiency ≈ gain_mean / directivity_mean
    # Approximate: efficiency = 10^(gain_mean/10) / (4π/Ω_beam)
    # Simple estimate: compare to theoretical isotropic
    gain_linear = 10**(gain_mean / 10)
    # For a small antenna above ground, typical directivity ~3-5 dBi
    estimated_directivity = 4.0  # dBi, typical for IFA
    efficiency = gain_linear / (10**(estimated_directivity / 10))
    efficiency = min(efficiency, 1.0)

    # Account for FR4 dielectric losses
    # FR4 at RF frequencies has significant loss
    substrate_loss_factor = 1.0 / (1.0 + 0.5 * np.sqrt(freq_hz / 1e9))
    efficiency *= substrate_loss_factor

    z_real = necpp.nec_impedance_real(nec, 0)
    z_imag = necpp.nec_impedance_imag(nec, 0)

    necpp.nec_delete(nec)

    print(f"  Peak gain: {gain_max:.1f} dBi")
    print(f"  Mean gain: {gain_mean:.1f} dBi")
    print(f"  Estimated efficiency: {efficiency*100:.0f}%")
    print(f"  Impedance: {z_real:.1f} + j{z_imag:.1f} Ω")

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


def run_full_simulation(params=None, output_dir='results', quick=False):
    """Run complete simulation for both bands.

    Args:
        params: optional dict of parameter overrides
        output_dir: directory for output files
        quick: if True, use fewer frequency points for faster iteration

    Returns:
        dict with results for each band
    """
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

        # Radiation pattern
        rad_result = simulate_radiation(dual, band)

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

        plot_radiation_pattern(
            rad_result['theta'], rad_result['e_plane_gain'],
            band_label, rad_result['freq_hz'] / 1e9, plane='E',
            save_path=os.path.join(band_dir, 'pattern_e.png'))

        plot_radiation_pattern(
            rad_result['phi'], rad_result['h_plane_gain'],
            band_label, rad_result['freq_hz'] / 1e9, plane='H',
            save_path=os.path.join(band_dir, 'pattern_h.png'))

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


def validate_geometry_for_nec(antenna, band):
    """Check that the geometry won't crash NEC2.

    Returns (ok, reason) tuple.
    """
    ifa = antenna.lowband if band == 'lowband' else antenna.highband

    if not ifa._segments:
        return False, "No segments generated"

    wire_radius = ifa.wire_radius
    min_length = 8 * wire_radius  # NEC2 thin-wire constraint

    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.1:
            continue
        if length < min_length:
            return False, f"Segment too short: {length:.2f}mm < {min_length:.2f}mm"

    # Check that feed wire exists
    feed_found = False
    for seg in ifa._segments:
        x1, y1, x2, y2 = seg
        if (abs(x1 - ifa.feed_x) < 0.01 and abs(y1) < 0.01 and
                abs(x2 - ifa.feed_x) < 0.01 and y2 > 0):
            feed_found = True
            break

    if not feed_found:
        return False, "Feed wire not found in geometry"

    return True, "OK"


def _subprocess_s11_worker(params_dict, band, freqs_list, result_dict):
    """Worker function for subprocess-safe S11 simulation.

    Runs in a child process so NEC2 segfaults don't kill the parent.
    """
    try:
        # Suppress stdout in worker (geometry print spam)
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = devnull

        freqs = np.array(freqs_list)
        dual = DualBandIFA(params_dict)
        dual.generate()

        # Validate geometry before NEC2 call
        ok, reason = validate_geometry_for_nec(dual, band)
        if not ok:
            result_dict['error'] = reason
            return

        targets = TARGETS[band]
        ref_freq = targets['freq_center']

        nec, feed_tag, feed_seg = build_nec_for_band(dual, band, ref_freq)
        necpp.nec_ex_card(nec, 0, feed_tag, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)

        n_freq = len(freqs)
        f_start_mhz = freqs[0] / 1e6
        f_step_mhz = (freqs[-1] - freqs[0]) / (n_freq - 1) / 1e6 if n_freq > 1 else 0
        necpp.nec_fr_card(nec, 0, n_freq, f_start_mhz, f_step_mhz)
        necpp.nec_rp_card(nec, 0, 1, 1, 0, 0, 0, 0, 0, 90, 90, 0, 0, 0)

        z_real = np.zeros(n_freq)
        z_imag = np.zeros(n_freq)
        for i in range(n_freq):
            z_real[i] = necpp.nec_impedance_real(nec, i)
            z_imag[i] = necpp.nec_impedance_imag(nec, i)
        necpp.nec_delete(nec)

        # Check for NEC2 failure indicators
        if np.any(z_real < -900):
            result_dict['error'] = "NEC2 returned -999 (solver failure)"
            return

        z_complex = z_real + 1j * z_imag
        s11_complex = impedance_to_s11(z_complex)
        s11_vals_db = s11_db(s11_complex)
        vswr_vals = s11_to_vswr(s11_complex)

        result_dict['s11_db'] = s11_vals_db.tolist()
        result_dict['vswr'] = vswr_vals.tolist()
        result_dict['z_real'] = z_real.tolist()
        result_dict['z_imag'] = z_imag.tolist()
        result_dict['ok'] = True
    except Exception as e:
        result_dict['error'] = str(e)
    finally:
        sys.stdout = old_stdout
        devnull.close()


def simulate_s11_safe(params_dict, band, freqs, timeout=30):
    """Subprocess-safe S11 simulation. Returns result dict or None on crash."""
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
        return None  # segfault or other crash

    if not result_dict.get('ok', False):
        return None

    return {
        'freqs': freqs,
        's11_db': np.array(result_dict['s11_db']),
        'vswr': np.array(result_dict['vswr']),
        'z_real': np.array(result_dict['z_real']),
        'z_imag': np.array(result_dict['z_imag']),
    }


def objective_function(params):
    """Compute optimization objective: worst-case S11 across target bands.

    Returns a positive number (lower is better). Target is 0 (all S11 < threshold).
    """
    dual = DualBandIFA(params)
    geom = dual.generate()

    # Quick frequency sweep (fewer points)
    total_penalty = 0.0

    for band in ['lowband', 'highband']:
        targets = TARGETS[band]
        # Sample frequencies across the target band
        band_freqs = np.linspace(targets['freq_min'], targets['freq_max'], 21)

        s11_result = simulate_s11(dual, band, band_freqs)
        worst_s11 = np.max(s11_result['s11_db'])

        # Penalty: how much above the threshold
        if worst_s11 > targets['s11_db']:
            total_penalty += (worst_s11 - targets['s11_db'])**2

    return total_penalty


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dual-band IFA antenna simulation')
    parser.add_argument('--quick', action='store_true',
                        help='Use fewer frequency points for quick iteration')
    parser.add_argument('--band', choices=['lowband', 'highband', 'both'],
                        default='both', help='Which band to simulate')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()

    print("=" * 60)
    print("  Dual-Band PCB IFA Antenna Simulation")
    print("  Using NEC2 (Method of Moments)")
    print("=" * 60)

    results = run_full_simulation(output_dir=args.output, quick=args.quick)

    print("\nSimulation complete! Check the results/ directory for plots.")
