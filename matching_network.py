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
Matching network design for dual-band branched IFA.
Loads Z11 from simulation, optimizes LC matching network components
to minimize worst-case S11 across both bands.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import sys
import os

# Band definitions
LB_MIN, LB_MAX = 791e6, 960e6
HB_MIN, HB_MAX = 1710e6, 1990e6
Z0 = 50.0


def load_z11(npz_path='results/full_accuracy_final.npz'):
    """Load impedance data from simulation."""
    data = np.load(npz_path, allow_pickle=True)
    return data['freqs'], data['z_complex']


def s11_from_z(z, z0=Z0):
    """Calculate S11 in dB from complex impedance."""
    gamma = (z - z0) / (z + z0)
    return 20 * np.log10(np.maximum(np.abs(gamma), 1e-15))


def apply_series(z, z_series):
    """Apply series element: Z_total = Z_ant + Z_series."""
    return z + z_series


def apply_shunt(z, z_shunt):
    """Apply shunt element: work in admittance."""
    y = 1.0 / z + 1.0 / z_shunt
    return 1.0 / y


def z_inductor(f, L):
    """Impedance of series inductor. L in nH."""
    return 1j * 2 * np.pi * f * L * 1e-9


def z_capacitor(f, C):
    """Impedance of series capacitor. C in pF."""
    return 1.0 / (1j * 2 * np.pi * f * C * 1e-12)


def z_parallel_lc(f, L, C):
    """Impedance of parallel LC (trap). L in nH, C in pF."""
    zl = z_inductor(f, L)
    zc = z_capacitor(f, C)
    return (zl * zc) / (zl + zc)


def z_series_lc(f, L, C):
    """Impedance of series LC. L in nH, C in pF."""
    return z_inductor(f, L) + z_capacitor(f, C)


# ============================================================
# Matching network topologies
# ============================================================

def match_series_L(freqs, z_ant, params):
    """Series inductor only. params: [L_nH]"""
    L = params[0]
    return apply_series(z_ant, z_inductor(freqs, L))


def match_shunt_C(freqs, z_ant, params):
    """Shunt capacitor only. params: [C_pF]"""
    C = params[0]
    return apply_shunt(z_ant, z_capacitor(freqs, C))


def match_series_L_shunt_C(freqs, z_ant, params):
    """Series L then shunt C (low-pass L-network). params: [L_nH, C_pF]"""
    L, C = params
    z = apply_series(z_ant, z_inductor(freqs, L))
    z = apply_shunt(z, z_capacitor(freqs, C))
    return z


def match_shunt_C_series_L(freqs, z_ant, params):
    """Shunt C then series L. params: [C_pF, L_nH]"""
    C, L = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C))
    z = apply_series(z, z_inductor(freqs, L))
    return z


def match_series_C_shunt_L(freqs, z_ant, params):
    """Series C then shunt L (high-pass L-network). params: [C_pF, L_nH]"""
    C, L = params
    z = apply_series(z_ant, z_capacitor(freqs, C))
    z = apply_shunt(z, z_inductor(freqs, L))
    return z


def match_shunt_L_series_C(freqs, z_ant, params):
    """Shunt L then series C. params: [L_nH, C_pF]"""
    L, C = params
    z = apply_shunt(z_ant, z_inductor(freqs, L))
    z = apply_series(z, z_capacitor(freqs, C))
    return z


def match_pi_CLC(freqs, z_ant, params):
    """Pi network: shunt C1, series L, shunt C2. params: [C1_pF, L_nH, C2_pF]"""
    C1, L, C2 = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    return z


def match_T_LCL(freqs, z_ant, params):
    """T network: series L1, shunt C, series L2. params: [L1_nH, C_pF, L2_nH]"""
    L1, C, L2 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C))
    z = apply_series(z, z_inductor(freqs, L2))
    return z


def match_series_L_shunt_C_series_C(freqs, z_ant, params):
    """Series L, shunt C, series C. params: [L_nH, C1_pF, C2_pF]"""
    L, C1, C2 = params
    z = apply_series(z_ant, z_inductor(freqs, L))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_capacitor(freqs, C2))
    return z


def match_shunt_C_series_L_shunt_C(freqs, z_ant, params):
    """Bandpass: shunt C1, series L, shunt C2. Same as pi_CLC."""
    return match_pi_CLC(freqs, z_ant, params)


def match_4elem_LCLC(freqs, z_ant, params):
    """4-element: series L1, shunt C1, series L2, shunt C2."""
    L1, C1, L2, C2 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    return z


def match_4elem_CLCL(freqs, z_ant, params):
    """4-element: shunt C1, series L1, shunt C2, series L2."""
    C1, L1, C2, L2 = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L2))
    return z


# Topology registry
TOPOLOGIES = {
    'series_L':          (match_series_L,          [(0.5, 30)]),
    'shunt_C':           (match_shunt_C,           [(0.1, 20)]),
    'series_L_shunt_C':  (match_series_L_shunt_C,  [(0.5, 30), (0.1, 20)]),
    'shunt_C_series_L':  (match_shunt_C_series_L,  [(0.1, 20), (0.5, 30)]),
    'series_C_shunt_L':  (match_series_C_shunt_L,  [(0.1, 20), (0.5, 30)]),
    'shunt_L_series_C':  (match_shunt_L_series_C,  [(0.5, 30), (0.1, 20)]),
    'pi_CLC':            (match_pi_CLC,            [(0.1, 20), (0.5, 30), (0.1, 20)]),
    'T_LCL':             (match_T_LCL,             [(0.5, 30), (0.1, 20), (0.5, 30)]),
    'series_L_shunt_C_series_C': (match_series_L_shunt_C_series_C,
                                  [(0.5, 30), (0.1, 20), (0.1, 20)]),
    '4elem_LCLC':        (match_4elem_LCLC,        [(0.5, 30), (0.1, 20), (0.5, 30), (0.1, 20)]),
    '4elem_CLCL':        (match_4elem_CLCL,        [(0.1, 20), (0.5, 30), (0.1, 20), (0.5, 30)]),
}


def cost_function(params, freqs, z_ant, match_fn, lb_weight=1.0, hb_weight=1.0):
    """
    Cost = weighted worst-case S11 across both bands.
    Lower (more negative) is better. We minimize the maximum (least negative) S11.
    """
    try:
        z_matched = match_fn(freqs, z_ant, params)
        s11 = s11_from_z(z_matched)
    except (ZeroDivisionError, FloatingPointError, RuntimeWarning):
        return 0.0  # worst possible (0 dB = total reflection)

    # Band masks
    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    lb_worst = np.max(s11[lb_mask]) if np.any(lb_mask) else 0.0
    hb_worst = np.max(s11[hb_mask]) if np.any(hb_mask) else 0.0

    # Weighted worst case (want both bands good)
    cost = lb_weight * lb_worst + hb_weight * hb_worst
    return cost


def optimize_topology(name, freqs, z_ant, lb_weight=1.0, hb_weight=1.0, n_runs=5):
    """Optimize a single matching topology. Returns best params and performance."""
    match_fn, bounds = TOPOLOGIES[name]

    best_cost = 1e9
    best_params = None

    for _ in range(n_runs):
        result = differential_evolution(
            cost_function, bounds,
            args=(freqs, z_ant, match_fn, lb_weight, hb_weight),
            maxiter=500, tol=1e-8, seed=np.random.randint(0, 100000),
            polish=True, workers=1,
        )
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x

    # Evaluate final performance
    z_matched = match_fn(freqs, z_ant, best_params)
    s11 = s11_from_z(z_matched)
    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    return {
        'name': name,
        'params': best_params,
        'bounds': bounds,
        'lb_worst_s11': np.max(s11[lb_mask]),
        'hb_worst_s11': np.max(s11[hb_mask]),
        'lb_best_s11': np.min(s11[lb_mask]),
        'hb_best_s11': np.min(s11[hb_mask]),
        'cost': best_cost,
        's11_matched': s11,
        'z_matched': z_matched,
    }


def print_result(r):
    """Pretty-print optimization result."""
    match_fn, bounds = TOPOLOGIES[r['name']]
    param_names = {
        'series_L': ['L (nH)'],
        'shunt_C': ['C (pF)'],
        'series_L_shunt_C': ['L (nH)', 'C (pF)'],
        'shunt_C_series_L': ['C (pF)', 'L (nH)'],
        'series_C_shunt_L': ['C (pF)', 'L (nH)'],
        'shunt_L_series_C': ['L (nH)', 'C (pF)'],
        'pi_CLC': ['C1 (pF)', 'L (nH)', 'C2 (pF)'],
        'T_LCL': ['L1 (nH)', 'C (pF)', 'L2 (nH)'],
        'series_L_shunt_C_series_C': ['L (nH)', 'C1 (pF)', 'C2 (pF)'],
        '4elem_LCLC': ['L1 (nH)', 'C1 (pF)', 'L2 (nH)', 'C2 (pF)'],
        '4elem_CLCL': ['C1 (pF)', 'L1 (nH)', 'C2 (pF)', 'L2 (nH)'],
    }
    names = param_names.get(r['name'], [f'p{i}' for i in range(len(r['params']))])
    parts = ', '.join(f'{n}={v:.2f}' for n, v in zip(names, r['params']))
    print(f"  {r['name']:30s} | LB worst={r['lb_worst_s11']:6.1f} dB | "
          f"HB worst={r['hb_worst_s11']:6.1f} dB | {parts}")


def plot_comparison(freqs, s11_bare, results, save_path='results/matching_comparison.png'):
    """Plot bare vs matched S11 for top results."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Bare antenna
    ax.plot(freqs / 1e6, s11_bare, 'k-', linewidth=2, label='Bare antenna', alpha=0.6)

    # Matched results
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(results), 10)))
    for r, color in zip(results, colors):
        ax.plot(freqs / 1e6, r['s11_matched'], '-', color=color, linewidth=1.5,
                label=f"{r['name']} (LB={r['lb_worst_s11']:.1f}, HB={r['hb_worst_s11']:.1f})")

    # Band regions
    ax.axvspan(LB_MIN / 1e6, LB_MAX / 1e6, alpha=0.1, color='blue', label='LB 791-960')
    ax.axvspan(HB_MIN / 1e6, HB_MAX / 1e6, alpha=0.1, color='red', label='HB 1710-1990')

    # Targets
    ax.axhline(-6, color='orange', linestyle='--', alpha=0.5, label='-6 dB target')
    ax.axhline(-10, color='green', linestyle='--', alpha=0.5, label='-10 dB')
    ax.axhline(-20, color='purple', linestyle='--', alpha=0.3, label='-20 dB target')

    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('S11 (dB)', fontsize=12)
    ax.set_title('Matching Network Comparison: Bare vs Matched', fontsize=14)
    ax.set_xlim(500, 2500)
    ax.set_ylim(-35, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved plot: {save_path}")
    plt.close()


def plot_smith(freqs, z_bare, z_matched, name, save_path):
    """Plot Smith chart: bare vs matched."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Smith chart circles
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5)
    for r in [0.2, 0.5, 1.0, 2.0, 5.0]:
        cx = r / (1 + r)
        cr = 1 / (1 + r)
        ax.plot(cx + cr * np.cos(theta), cr * np.sin(theta), 'gray', linewidth=0.3)
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        angles = np.linspace(0, np.pi / 2, 100)
        xc = 1 + np.cos(angles) / x
        yc = np.sin(angles) / x
        mask = xc**2 + yc**2 <= 1.01
        # Skip drawing x-circles for simplicity

    # Gamma from Z
    gamma_bare = (z_bare - Z0) / (z_bare + Z0)
    gamma_matched = (z_matched - Z0) / (z_matched + Z0)

    # Band masks
    lb = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    ax.plot(gamma_bare[lb].real, gamma_bare[lb].imag, 'b-', linewidth=1.5,
            alpha=0.4, label='Bare LB')
    ax.plot(gamma_bare[hb].real, gamma_bare[hb].imag, 'r-', linewidth=1.5,
            alpha=0.4, label='Bare HB')
    ax.plot(gamma_matched[lb].real, gamma_matched[lb].imag, 'b-', linewidth=2.5,
            label='Matched LB')
    ax.plot(gamma_matched[hb].real, gamma_matched[hb].imag, 'r-', linewidth=2.5,
            label='Matched HB')
    ax.plot(0, 0, 'k+', markersize=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f'Smith Chart: {name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("  MATCHING NETWORK OPTIMIZER — Dual-Band Branched IFA")
    print("=" * 70)

    # Load data
    freqs, z_ant = load_z11()
    s11_bare = s11_from_z(z_ant)

    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)
    print(f"\nBare antenna performance:")
    print(f"  LB worst S11: {np.max(s11_bare[lb_mask]):.1f} dB")
    print(f"  HB worst S11: {np.max(s11_bare[hb_mask]):.1f} dB")

    # Optimize all topologies
    print(f"\nOptimizing {len(TOPOLOGIES)} matching topologies...")
    print("-" * 90)

    results = []
    for name in TOPOLOGIES:
        print(f"  Optimizing {name}...", end='', flush=True)
        r = optimize_topology(name, freqs, z_ant, lb_weight=1.0, hb_weight=1.0, n_runs=3)
        results.append(r)
        print(f" LB={r['lb_worst_s11']:.1f} dB, HB={r['hb_worst_s11']:.1f} dB")

    # Sort by combined worst-case
    results.sort(key=lambda r: max(r['lb_worst_s11'], r['hb_worst_s11']))

    print("\n" + "=" * 90)
    print("  RESULTS (sorted by best worst-case S11)")
    print("=" * 90)
    for r in results:
        print_result(r)

    # Plot top 5
    top5 = results[:5]
    plot_comparison(freqs, s11_bare, top5)

    # Plot Smith chart for best
    best = results[0]
    plot_smith(freqs, z_ant, best['z_matched'], best['name'],
              'results/matching_smith.png')
    print(f"\n  Best topology: {best['name']}")
    print(f"  LB worst: {best['lb_worst_s11']:.1f} dB, HB worst: {best['hb_worst_s11']:.1f} dB")

    # Also try with stronger HB weight (since HB is the weak link)
    print("\n\n--- Re-optimizing best topologies with HB priority (2x weight) ---")
    top_names = [r['name'] for r in results[:5]]
    results_hb = []
    for name in top_names:
        r = optimize_topology(name, freqs, z_ant, lb_weight=1.0, hb_weight=2.0, n_runs=5)
        results_hb.append(r)
    results_hb.sort(key=lambda r: max(r['lb_worst_s11'], r['hb_worst_s11']))
    print("\nHB-weighted results:")
    for r in results_hb:
        print_result(r)

    plot_comparison(freqs, s11_bare, results_hb[:5],
                    save_path='results/matching_hb_priority.png')

    return results, results_hb


if __name__ == '__main__':
    results, results_hb = main()
