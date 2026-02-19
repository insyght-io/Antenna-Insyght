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
Utility functions for antenna simulation:
- S-parameter calculations
- Plotting (S11, Smith chart, VSWR, radiation pattern)
- Geometry helpers
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from config import C0, Z0, TARGETS


# ============================================================
# S-Parameter Calculations
# ============================================================

def impedance_to_s11(z, z0=Z0):
    """Convert complex impedance to S11 (reflection coefficient)."""
    return (z - z0) / (z + z0)


def s11_to_vswr(s11):
    """Convert S11 (complex) to VSWR."""
    gamma = np.abs(s11)
    gamma = np.clip(gamma, 0, 0.9999)
    return (1 + gamma) / (1 - gamma)


def s11_db(s11):
    """Convert S11 (complex) to dB."""
    return 20 * np.log10(np.maximum(np.abs(s11), 1e-12))


def return_loss_db(s11):
    """Return loss in dB (positive number)."""
    return -s11_db(s11)


def find_resonances(freqs, s11_vals_db, threshold_db=-6.0):
    """Find resonant frequencies where S11 dips below threshold.
    Returns list of (freq, s11_db) tuples.
    """
    resonances = []
    below = s11_vals_db < threshold_db
    # Find groups of consecutive points below threshold
    in_dip = False
    dip_start = 0
    for i in range(len(below)):
        if below[i] and not in_dip:
            in_dip = True
            dip_start = i
        elif not below[i] and in_dip:
            in_dip = False
            # Find minimum in this dip
            dip_region = s11_vals_db[dip_start:i]
            min_idx = dip_start + np.argmin(dip_region)
            resonances.append((freqs[min_idx], s11_vals_db[min_idx]))
    # Handle case where dip extends to end
    if in_dip:
        dip_region = s11_vals_db[dip_start:]
        min_idx = dip_start + np.argmin(dip_region)
        resonances.append((freqs[min_idx], s11_vals_db[min_idx]))
    return resonances


def find_bandwidth(freqs, s11_vals_db, center_freq, threshold_db=-6.0):
    """Find -N dB bandwidth around a center frequency.
    Returns (f_low, f_high, bandwidth) in Hz, or None if not found.
    """
    below = s11_vals_db < threshold_db
    center_idx = np.argmin(np.abs(freqs - center_freq))

    if not below[center_idx]:
        return None

    # Search left
    f_low = freqs[center_idx]
    for i in range(center_idx, -1, -1):
        if not below[i]:
            # Interpolate
            if i + 1 < len(freqs):
                frac = (threshold_db - s11_vals_db[i]) / (s11_vals_db[i + 1] - s11_vals_db[i])
                f_low = freqs[i] + frac * (freqs[i + 1] - freqs[i])
            break
    else:
        f_low = freqs[0]

    # Search right
    f_high = freqs[center_idx]
    for i in range(center_idx, len(freqs)):
        if not below[i]:
            if i - 1 >= 0:
                frac = (threshold_db - s11_vals_db[i - 1]) / (s11_vals_db[i] - s11_vals_db[i - 1])
                f_high = freqs[i - 1] + frac * (freqs[i] - freqs[i - 1])
            break
    else:
        f_high = freqs[-1]

    return (f_low, f_high, f_high - f_low)


# ============================================================
# Plotting Functions
# ============================================================

def plot_s11(freqs, s11_complex, band_name, targets=None, save_path=None):
    """Plot S11 in dB vs frequency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    s11_val = s11_db(s11_complex)
    ax.plot(freqs / 1e6, s11_val, 'b-', linewidth=2, label='S11')

    # Target threshold
    if targets:
        ax.axhline(y=targets['s11_db'], color='r', linestyle='--',
                    alpha=0.7, label=f"Target ({targets['s11_db']} dB)")
        ax.axvspan(targets['freq_min'] / 1e6, targets['freq_max'] / 1e6,
                    alpha=0.1, color='green', label='Target band')

    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('S11 (dB)', fontsize=12)
    ax.set_title(f'S11 - {band_name}', fontsize=14)
    ax.set_ylim([-35, 0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_vswr(freqs, s11_complex, band_name, targets=None, save_path=None):
    """Plot VSWR vs frequency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    vswr = s11_to_vswr(s11_complex)
    ax.plot(freqs / 1e6, vswr, 'b-', linewidth=2, label='VSWR')

    if targets:
        ax.axhline(y=targets['vswr_max'], color='r', linestyle='--',
                    alpha=0.7, label=f"Target (VSWR < {targets['vswr_max']}:1)")
        ax.axvspan(targets['freq_min'] / 1e6, targets['freq_max'] / 1e6,
                    alpha=0.1, color='green', label='Target band')

    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('VSWR', fontsize=12)
    ax.set_title(f'VSWR - {band_name}', fontsize=14)
    ax.set_ylim([1, 10])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_smith(s11_complex, band_name, freqs=None, save_path=None):
    """Plot Smith chart."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw Smith chart circles
    theta = np.linspace(0, 2 * np.pi, 200)

    # Unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)

    # Constant resistance circles
    for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx = r / (1 + r)
        cr = 1 / (1 + r)
        circle_x = cx + cr * np.cos(theta)
        circle_y = cr * np.sin(theta)
        mask = circle_x**2 + circle_y**2 <= 1.01
        ax.plot(circle_x[mask], circle_y[mask], 'k-', linewidth=0.3, alpha=0.4)

    # Constant reactance arcs
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        cx = 1.0
        cy = 1.0 / x
        cr = 1.0 / x
        arc_theta = np.linspace(-np.pi, np.pi, 400)
        arc_x = cx + cr * np.cos(arc_theta)
        arc_y = cy + cr * np.sin(arc_theta)
        mask = arc_x**2 + arc_y**2 <= 1.01
        ax.plot(arc_x[mask], arc_y[mask], 'k-', linewidth=0.3, alpha=0.4)
        # Conjugate
        ax.plot(arc_x[mask], -arc_y[mask], 'k-', linewidth=0.3, alpha=0.4)

    # Horizontal axis
    ax.plot([-1, 1], [0, 0], 'k-', linewidth=0.5)

    # Plot S11 data
    s11_r = np.real(s11_complex)
    s11_i = np.imag(s11_complex)
    scatter = ax.scatter(s11_r, s11_i, c=freqs / 1e6 if freqs is not None else range(len(s11_r)),
                         cmap='rainbow', s=3, zorder=5)
    if freqs is not None:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('Frequency (MHz)')

    # Mark start and end
    ax.plot(s11_r[0], s11_i[0], 'go', markersize=8, label='Start')
    ax.plot(s11_r[-1], s11_i[-1], 'rs', markersize=8, label='End')

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.set_title(f'Smith Chart - {band_name}', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_radiation_pattern(theta_deg, gain_db, band_name, freq_ghz, plane='E',
                            save_path=None):
    """Plot radiation pattern in polar coordinates."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    theta_rad = np.deg2rad(theta_deg)

    # Sanitize gain data
    gain_db = np.where(np.isfinite(gain_db), gain_db, -30.0)

    ax.plot(theta_rad, gain_db, 'b-', linewidth=2)
    ax.set_title(f'{plane}-plane Pattern - {band_name}\n{freq_ghz:.3f} GHz',
                 fontsize=14, pad=20)
    ax.set_rlabel_position(45)

    # Set reasonable gain range
    gain_max = np.max(gain_db)
    gain_min = max(gain_max - 40, np.min(gain_db))
    if not np.isfinite(gain_max) or not np.isfinite(gain_min):
        gain_max, gain_min = 5, -35
    ax.set_rlim([gain_min, gain_max + 3])

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_antenna_geometry(traces_lb, traces_hb, save_path=None):
    """Plot the antenna geometry showing both IFAs and the full PCB outline."""
    fig, ax = plt.subplots(figsize=(12, 14))

    from config import BOARD_RADIUS, GND_EXTENSION, GND_HALF_WIDTH

    r = BOARD_RADIUS
    hw = GND_HALF_WIDTH  # half-width of rectangular PCB body

    # Draw half-circle antenna area
    theta = np.linspace(0, np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2, label='Board edge')
    ax.plot([-hw, hw], [0, 0], 'k-', linewidth=1.5)

    # Draw rectangular PCB body (ground plane area)
    gnd_rect_x = [-hw, hw, hw, -hw, -hw]
    gnd_rect_y = [0, 0, -GND_EXTENSION, -GND_EXTENSION, 0]
    ax.fill(gnd_rect_x, gnd_rect_y, alpha=0.12, color='brown', label='Ground plane (PCB body)')
    ax.plot(gnd_rect_x, gnd_rect_y, 'k-', linewidth=2)

    # Draw low band traces
    for i, seg in enumerate(traces_lb):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        label = 'Low band IFA' if i == 0 else None
        ax.plot(xs, ys, 'b-', linewidth=2, label=label)

    # Draw high band traces
    for i, seg in enumerate(traces_hb):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        label = 'High band IFA' if i == 0 else None
        ax.plot(xs, ys, 'r-', linewidth=2, label=label)

    # Dimensions annotations
    ax.annotate('', xy=(hw, -GND_EXTENSION - 3), xytext=(-hw, -GND_EXTENSION - 3),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(0, -GND_EXTENSION - 6, f'{2*hw:.0f} mm', ha='center', fontsize=9, color='gray')
    ax.annotate('', xy=(hw + 3, 0), xytext=(hw + 3, -GND_EXTENSION),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(hw + 6, -GND_EXTENSION / 2, f'{GND_EXTENSION:.0f} mm', ha='left',
            fontsize=9, color='gray', rotation=90, va='center')

    ax.set_xlim([-40, 45])
    ax.set_ylim([-GND_EXTENSION - 10, 35])
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Dual-Band IFA Antenna Layout', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_antenna_geometry_branched(traces_lb, traces_hb, cap_rect=None,
                                    save_path=None):
    """Plot the branched IFA geometry showing LB arm, HB branch, and cap load."""
    fig, ax = plt.subplots(figsize=(12, 14))

    from config import BOARD_RADIUS, GND_EXTENSION, GND_HALF_WIDTH

    r = BOARD_RADIUS
    hw = GND_HALF_WIDTH

    # Half-circle antenna area
    theta = np.linspace(0, np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2, label='Board edge')
    ax.plot([-hw, hw], [0, 0], 'k-', linewidth=1.5)

    # Rectangular PCB body
    gnd_rect_x = [-hw, hw, hw, -hw, -hw]
    gnd_rect_y = [0, 0, -GND_EXTENSION, -GND_EXTENSION, 0]
    ax.fill(gnd_rect_x, gnd_rect_y, alpha=0.12, color='brown', label='Ground plane')
    ax.plot(gnd_rect_x, gnd_rect_y, 'k-', linewidth=2)

    # LB arm traces (blue)
    for i, seg in enumerate(traces_lb):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        label = 'LB arm' if i == 0 else None
        ax.plot(xs, ys, 'b-', linewidth=2, label=label)

    # HB branch traces (red)
    for i, seg in enumerate(traces_hb):
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        label = 'HB branch' if i == 0 else None
        ax.plot(xs, ys, 'r-', linewidth=2, label=label)

    # Capacitive load rectangle (green)
    if cap_rect is not None:
        cx, cy, w, h = cap_rect
        from matplotlib.patches import Rectangle
        rect = Rectangle((cx - w / 2, cy - h / 2), w, h,
                          linewidth=2, edgecolor='green', facecolor='green',
                          alpha=0.3, label='Cap load')
        ax.add_patch(rect)

    # Dimensions
    ax.annotate('', xy=(hw, -GND_EXTENSION - 3), xytext=(-hw, -GND_EXTENSION - 3),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(0, -GND_EXTENSION - 6, f'{2*hw:.0f} mm', ha='center', fontsize=9, color='gray')
    ax.annotate('', xy=(hw + 3, 0), xytext=(hw + 3, -GND_EXTENSION),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(hw + 6, -GND_EXTENSION / 2, f'{GND_EXTENSION:.0f} mm', ha='left',
            fontsize=9, color='gray', rotation=90, va='center')

    ax.set_xlim([-40, 45])
    ax.set_ylim([-GND_EXTENSION - 10, 35])
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Branched IFA Antenna Layout', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_combined_s11_branched(s11_result, save_path=None):
    """Plot single S11 curve with both LB and HB target bands highlighted."""
    from config import LB_FREQ_MIN, LB_FREQ_MAX, HB_FREQ_MIN, HB_FREQ_MAX

    fig, ax = plt.subplots(figsize=(12, 6))
    freqs = s11_result['freqs']
    s11_val = s11_result['s11_db']

    ax.plot(freqs / 1e6, s11_val, 'b-', linewidth=2, label='S11')

    # Target threshold
    ax.axhline(y=-6, color='r', linestyle='--', alpha=0.7, label='Target (-6 dB)')

    # LB target band
    ax.axvspan(LB_FREQ_MIN / 1e6, LB_FREQ_MAX / 1e6,
               alpha=0.1, color='blue', label='LB band (700-960 MHz)')
    # HB target band
    ax.axvspan(HB_FREQ_MIN / 1e6, HB_FREQ_MAX / 1e6,
               alpha=0.1, color='red', label='HB band (1710-2170 MHz)')

    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('S11 (dB)', fontsize=12)
    ax.set_title('Branched IFA - S11 (Both Bands)', fontsize=14)
    ax.set_ylim([-35, 0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================
# Summary and Reporting
# ============================================================

def print_summary(band_name, freqs, s11_complex, gain_dbi=None, efficiency=None):
    """Print a performance summary for one antenna band."""
    s11_val = s11_db(s11_complex)
    vswr = s11_to_vswr(s11_complex)
    targets = TARGETS.get(band_name, {})

    print(f"\n{'='*60}")
    print(f"  {band_name.upper()} ANTENNA SUMMARY")
    print(f"{'='*60}")

    # Resonances
    resonances = find_resonances(freqs, s11_val)
    if resonances:
        for i, (f, s) in enumerate(resonances):
            print(f"  Resonance {i+1}: {f/1e6:.1f} MHz (S11 = {s:.1f} dB)")
    else:
        print("  No resonances found below -6 dB")

    # Bandwidth
    if targets:
        fc = targets.get('freq_center', freqs[np.argmin(s11_val)])
        bw6 = find_bandwidth(freqs, s11_val, fc, -6.0)
        bw10 = find_bandwidth(freqs, s11_val, fc, -10.0)
        if bw6:
            print(f"  -6 dB BW: {bw6[0]/1e6:.1f} - {bw6[1]/1e6:.1f} MHz ({bw6[2]/1e6:.1f} MHz)")
        if bw10:
            print(f"  -10 dB BW: {bw10[0]/1e6:.1f} - {bw10[1]/1e6:.1f} MHz ({bw10[2]/1e6:.1f} MHz)")

    # Worst-case S11 in target band
    if targets:
        band_mask = (freqs >= targets['freq_min']) & (freqs <= targets['freq_max'])
        if np.any(band_mask):
            worst_s11 = np.max(s11_val[band_mask])
            worst_vswr = np.max(vswr[band_mask])
            print(f"  Worst S11 in band: {worst_s11:.1f} dB (target: < {targets['s11_db']} dB)")
            print(f"  Worst VSWR in band: {worst_vswr:.1f}:1 (target: < {targets['vswr_max']}:1)")

    if gain_dbi is not None:
        print(f"  Peak gain: {gain_dbi:.1f} dBi")
    if efficiency is not None:
        print(f"  Efficiency: {efficiency*100:.1f}%")

    # Pass/fail
    if targets:
        passed = True
        if np.any(band_mask):
            if worst_s11 > targets['s11_db']:
                passed = False
            if worst_vswr > targets['vswr_max']:
                passed = False
        if gain_dbi is not None and gain_dbi < targets.get('gain_min_dbi', -99):
            passed = False
        if efficiency is not None and efficiency < targets.get('efficiency_min', 0):
            passed = False
        status = "PASS" if passed else "FAIL"
        print(f"\n  Overall: [{status}]")

    print(f"{'='*60}\n")
    return resonances


def print_summary_branched(s11_result):
    """Print a performance summary for the branched IFA across both bands."""
    from config import LB_FREQ_MIN, LB_FREQ_MAX, HB_FREQ_MIN, HB_FREQ_MAX

    freqs = s11_result['freqs']
    s11_val = s11_result['s11_db']
    vswr = s11_to_vswr(s11_result['s11_complex'])

    print(f"\n{'='*60}")
    print(f"  BRANCHED IFA SUMMARY")
    print(f"{'='*60}")

    # Overall resonances
    resonances = find_resonances(freqs, s11_val)
    if resonances:
        for i, (f, s) in enumerate(resonances):
            print(f"  Resonance {i+1}: {f/1e6:.1f} MHz (S11 = {s:.1f} dB)")
    else:
        print("  No resonances found below -6 dB")

    for band_name, fmin, fmax in [('LOW BAND', LB_FREQ_MIN, LB_FREQ_MAX),
                                    ('HIGH BAND', HB_FREQ_MIN, HB_FREQ_MAX)]:
        targets = TARGETS.get('lowband' if 'LOW' in band_name else 'highband', {})
        band_mask = (freqs >= fmin) & (freqs <= fmax)

        print(f"\n  --- {band_name} ({fmin/1e6:.0f}-{fmax/1e6:.0f} MHz) ---")

        if np.any(band_mask):
            worst_s11 = np.max(s11_val[band_mask])
            best_s11 = np.min(s11_val[band_mask])
            worst_vswr = np.max(vswr[band_mask])
            print(f"  Worst S11: {worst_s11:.1f} dB (target: < {targets.get('s11_db', -6):.0f} dB)")
            print(f"  Best S11: {best_s11:.1f} dB")
            print(f"  Worst VSWR: {worst_vswr:.1f}:1")

            passed = worst_s11 <= targets.get('s11_db', -6)
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
        else:
            print("  No data in band")

    print(f"{'='*60}\n")
