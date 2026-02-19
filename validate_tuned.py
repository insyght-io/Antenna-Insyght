#!/usr/bin/env python3

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
Full-accuracy validation of the tuned h=14 design.
Runs at 200k timesteps then deep-matches with 8-element LCLCLCLC.
"""

import sys
import os
import numpy as np
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

from antenna_model import BranchedIFA
from simulate_openems import simulate_s11_branched
from config import LB_FREQ_MIN, LB_FREQ_MAX, HB_FREQ_MIN, HB_FREQ_MAX
from matching_network import (
    s11_from_z, Z0, LB_MIN, LB_MAX, HB_MIN, HB_MAX,
    apply_series, apply_shunt, z_inductor, z_capacitor,
)
from deep_match import cost_function_balanced
from push_20db import match_6elem_LCLCLC
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def match_8elem_LCLCLCLC(freqs, z_ant, params):
    """8-element: L1, C1, L2, C2, L3, C3, L4, C4."""
    L1, C1, L2, C2, L3, C3, L4, C4 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L3))
    z = apply_shunt(z, z_capacitor(freqs, C3))
    z = apply_series(z, z_inductor(freqs, L4))
    z = apply_shunt(z, z_capacitor(freqs, C4))
    return z


def deep_optimize(freqs, z_ant, match_fn, bounds, n_runs=30, name=""):
    """Deep optimization with many runs."""
    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    best_cost = 1e9
    best_params = None

    for run in range(n_runs):
        try:
            result = differential_evolution(
                cost_function_balanced, bounds,
                args=(freqs, z_ant, match_fn),
                maxiter=2000, tol=1e-12,
                seed=np.random.randint(0, 100000),
                polish=True, workers=1,
                mutation=(0.5, 1.5), recombination=0.9,
                popsize=40,
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
                z_m = match_fn(freqs, z_ant, best_params)
                s11_m = s11_from_z(z_m)
                lb_w = np.max(s11_m[lb_mask])
                hb_w = np.max(s11_m[hb_mask])
                print(f"    run {run+1}/{n_runs}: LB={lb_w:.1f}  HB={hb_w:.1f}  "
                      f"worst={max(lb_w,hb_w):.1f}")
        except Exception:
            continue

    if best_params is None:
        return None

    z_matched = match_fn(freqs, z_ant, best_params)
    s11 = s11_from_z(z_matched)

    return {
        'params': best_params,
        'lb_worst': float(np.max(s11[lb_mask])),
        'hb_worst': float(np.max(s11[hb_mask])),
        's11_matched': s11,
    }


def main():
    print("=" * 70)
    print("  FULL-ACCURACY VALIDATION: h=14 sx=-25 fo=7 hb=30 ha=8")
    print("=" * 70)

    params = {
        'SHORT_X': -25.0,
        'FEED_OFFSET': 7.0,
        'ELEM_HEIGHT': 14.0,
        'LB_LENGTH': 200.0,
        'LB_SPACING': 10.0,
        'LB_CAP_W': 0.0,
        'LB_CAP_L': 0.0,
        'HB_LENGTH': 30.0,
        'HB_ANGLE': 8.0,
        'TRACE_WIDTH': 1.5,
        'GND_CLEARANCE': 10.0,
    }

    ant = BranchedIFA(params)
    ant.generate_geometry()
    v = ant.validate_geometry()
    if v:
        print(f"  Geometry violation: {v}")
        return
    print(f"  LB arm: {ant.actual_lb_length:.1f}mm, HB arm: {ant.actual_hb_length:.1f}mm")

    print(f"\n  Running full-accuracy simulation (200k timesteps)...")
    t0 = time.time()
    result = simulate_s11_branched(
        ant, verbose=True, fast=False, explore=False,
        gnd_clearance=params.get('GND_CLEARANCE', 10.0))
    dt = time.time() - t0
    print(f"\n  Simulation time: {dt:.0f}s")

    freqs = result['freqs']
    s11 = result['s11_db']
    z = result['z_complex']

    lb = (freqs >= LB_FREQ_MIN) & (freqs <= LB_FREQ_MAX)
    hb = (freqs >= HB_FREQ_MIN) & (freqs <= HB_FREQ_MAX)

    print(f"\n  Full accuracy bare results:")
    print(f"  LB: worst={np.max(s11[lb]):.1f}dB, best={np.min(s11[lb]):.1f}dB, "
          f"res={freqs[lb][np.argmin(s11[lb])]/1e6:.0f}MHz")
    print(f"  HB: worst={np.max(s11[hb]):.1f}dB, best={np.min(s11[hb]):.1f}dB, "
          f"res={freqs[hb][np.argmin(s11[hb])]/1e6:.0f}MHz")

    for f_target in [791, 875, 960, 1710, 1850, 1990]:
        idx = np.argmin(np.abs(freqs - f_target*1e6))
        zv = z[idx]
        print(f"  Z({freqs[idx]/1e6:.0f}MHz) = {zv.real:.1f} + j{zv.imag:.1f} Ω")

    # Save Z11
    np.savez(os.path.join(RESULTS_DIR, 'tuned_full_accuracy.npz'),
             freqs=freqs, z_complex=z,
             s11_db=s11, s11_complex=result['s11_complex'])
    print(f"  Saved Z11 to results/tuned_full_accuracy.npz")

    # Deep matching with multiple topologies
    print(f"\n{'='*70}")
    print(f"  Deep matching optimization")
    print(f"{'='*70}")

    topologies = {
        '6elem_LCLCLC': (match_6elem_LCLCLC,
            [(0.1, 100), (0.05, 50), (0.1, 100), (0.05, 50), (0.1, 100), (0.05, 50)]),
        '8elem_LCLCLCLC': (match_8elem_LCLCLCLC,
            [(0.1, 80), (0.05, 40), (0.1, 80), (0.05, 40),
             (0.1, 80), (0.05, 40), (0.1, 80), (0.05, 40)]),
    }

    all_results = []
    for name, (match_fn, bounds) in topologies.items():
        print(f"\n  {name} (30 runs)...", flush=True)
        t0 = time.time()
        mr = deep_optimize(freqs, z, match_fn, bounds, n_runs=30, name=name)
        dt = time.time() - t0

        if mr is None:
            print(f"    FAILED")
            continue

        print(f"    FINAL: LB={mr['lb_worst']:.1f}  HB={mr['hb_worst']:.1f} ({dt:.0f}s)")
        pstr = ', '.join(f'{v:.3f}' for v in mr['params'])
        print(f"    Components: [{pstr}]")

        all_results.append({
            'name': name, **mr
        })

    all_results.sort(key=lambda r: max(r['lb_worst'], r['hb_worst']))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Full range
    ax1.plot(freqs / 1e6, s11, 'k-', linewidth=2, label='Bare antenna', alpha=0.6)
    colors = ['#2196F3', '#F44336', '#4CAF50']
    for r, color in zip(all_results, colors):
        ax1.plot(freqs / 1e6, r['s11_matched'], '-', color=color, linewidth=2,
                label=f"{r['name']} (LB={r['lb_worst']:.1f}, HB={r['hb_worst']:.1f})")

    ax1.axvspan(LB_MIN/1e6, LB_MAX/1e6, alpha=0.08, color='blue')
    ax1.axvspan(HB_MIN/1e6, HB_MAX/1e6, alpha=0.08, color='red')
    ax1.axhline(-6, color='orange', ls='--', alpha=0.4, label='-6 dB')
    ax1.axhline(-10, color='green', ls='--', alpha=0.4, label='-10 dB')
    ax1.axhline(-20, color='purple', ls='--', alpha=0.4, label='-20 dB')
    ax1.set_xlim(500, 2500)
    ax1.set_ylim(-40, 0)
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('S11 (dB)')
    ax1.set_title('Full-Accuracy: h=14 sx=-25 fo=7 hb=30 ha=8 (Bare vs Matched)')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Zoomed on bands
    best = all_results[0] if all_results else None
    if best:
        fig2, (ax_lb, ax_hb) = plt.subplots(1, 2, figsize=(16, 6))
        for ax, fmin, fmax, band_name in [
            (ax_lb, LB_MIN/1e6 - 50, LB_MAX/1e6 + 50, 'LB 791-960 MHz'),
            (ax_hb, HB_MIN/1e6 - 50, HB_MAX/1e6 + 50, 'HB 1710-1990 MHz'),
        ]:
            mask = (freqs/1e6 >= fmin) & (freqs/1e6 <= fmax)
            ax.plot(freqs[mask]/1e6, s11[mask], 'k-', lw=2, label='Bare', alpha=0.5)
            for r, color in zip(all_results, colors):
                ax.plot(freqs[mask]/1e6, r['s11_matched'][mask], '-', color=color, lw=2,
                        label=f"{r['name']}")
            ax.axhline(-6, color='orange', ls='--', alpha=0.4)
            ax.axhline(-10, color='green', ls='--', alpha=0.4)
            ax.axhline(-20, color='purple', ls='--', alpha=0.4)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('S11 (dB)')
            ax.set_title(band_name)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-40, 0)

        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, 'tuned_validated_zoom.png'), dpi=150)
        plt.close(fig2)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'tuned_validated.png'), dpi=150)
    plt.close(fig)

    # Save config
    if all_results:
        best = all_results[0]
        config = {
            'antenna_params': params,
            'matching_network': {
                'topology': best['name'],
                'params': [float(v) for v in best['params']],
                'lb_worst_s11': best['lb_worst'],
                'hb_worst_s11': best['hb_worst'],
            },
            'bare_performance': {
                'lb_worst_s11': float(np.max(s11[lb])),
                'hb_worst_s11': float(np.max(s11[hb])),
                'lb_res_mhz': float(freqs[lb][np.argmin(s11[lb])]/1e6),
                'hb_res_mhz': float(freqs[hb][np.argmin(s11[hb])]/1e6),
            },
        }
        with open(os.path.join(RESULTS_DIR, 'tuned_validated_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n  Saved: results/tuned_validated_config.json")

    print(f"\n  Plots: results/tuned_validated.png, results/tuned_validated_zoom.png")


if __name__ == '__main__':
    main()
