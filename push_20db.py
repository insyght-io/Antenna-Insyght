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
Try 6-element and 7-element matching networks to push past -20 dB.
Uses the full-accuracy Z11 data from best_candidate_full.npz.
"""

import sys
import os
import numpy as np
from scipy.optimize import differential_evolution

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
sys.path.insert(0, SCRIPT_DIR)

from matching_network import (
    s11_from_z, apply_series, apply_shunt,
    z_inductor, z_capacitor, z_parallel_lc,
    Z0, LB_MIN, LB_MAX, HB_MIN, HB_MAX,
)
from deep_match import cost_function_balanced, ALL_TOPOLOGIES


def match_6elem_LCLCLC(freqs, z_ant, params):
    """6-element: L1, C1, L2, C2, L3, C3."""
    L1, C1, L2, C2, L3, C3 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L3))
    z = apply_shunt(z, z_capacitor(freqs, C3))
    return z


def match_6elem_CLCLCL(freqs, z_ant, params):
    """6-element: C1, L1, C2, L2, C3, L3."""
    C1, L1, C2, L2, C3, L3 = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C3))
    z = apply_series(z, z_inductor(freqs, L3))
    return z


def match_5_trap_LCTLC(freqs, z_ant, params):
    """5-element with trap: L1, C1, trap(Lt,Ct), L2, C2.
    Trap between sections provides band isolation."""
    L1, C1, Lt, Ct, L2 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_parallel_lc(freqs, Lt, Ct))
    z = apply_series(z, z_inductor(freqs, L2))
    return z


def match_5_LCLCL_tight(freqs, z_ant, params):
    """Same as 5elem_LCLCL but with tighter search bounds."""
    L1, C1, L2, C2, L3 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L3))
    return z


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


EXTRA_TOPOLOGIES = {
    '8elem_LCLCLCLC': (match_8elem_LCLCLCLC,
                       [(0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30),
                        (0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30)]),
    '6elem_LCLCLC': (match_6elem_LCLCLC,
                     [(0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30),
                      (0.5, 50), (0.1, 30)]),
    '6elem_CLCLCL': (match_6elem_CLCLCL,
                     [(0.1, 30), (0.5, 50), (0.1, 30), (0.5, 50),
                      (0.1, 30), (0.5, 50)]),
}


def main():
    print("=" * 70)
    print("  PUSH PAST -20 dB: 6-element + refined search")
    print("=" * 70)

    data = np.load(os.path.join(RESULTS_DIR, 'best_candidate_full.npz'), allow_pickle=True)
    freqs = data['freqs']
    z_ant = data['z_complex']
    s11_bare = s11_from_z(z_ant)

    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)
    print(f"\n  Bare: LB worst={np.max(s11_bare[lb_mask]):.1f}dB, "
          f"HB worst={np.max(s11_bare[hb_mask]):.1f}dB")

    results = []

    for name, (match_fn, bounds) in EXTRA_TOPOLOGIES.items():
        print(f"\n  {name} (30 runs, popsize=40)...", flush=True)

        best_cost = 1e9
        best_params = None

        for run in range(30):
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
                    # Print progress
                    z_m = match_fn(freqs, z_ant, best_params)
                    s11_m = s11_from_z(z_m)
                    lb_w = np.max(s11_m[lb_mask])
                    hb_w = np.max(s11_m[hb_mask])
                    print(f"    run {run+1}: LB={lb_w:.1f}  HB={hb_w:.1f}  "
                          f"worst={max(lb_w,hb_w):.1f}")
            except Exception:
                continue

        if best_params is None:
            print("    FAILED")
            continue

        z_matched = match_fn(freqs, z_ant, best_params)
        s11_m = s11_from_z(z_matched)
        lb_worst = float(np.max(s11_m[lb_mask]))
        hb_worst = float(np.max(s11_m[hb_mask]))

        pstr = ', '.join(f'{v:.3f}' for v in best_params)
        print(f"    FINAL: LB={lb_worst:.1f}  HB={hb_worst:.1f}  [{pstr}]")
        results.append({
            'name': name, 'params': best_params,
            'lb_worst': lb_worst, 'hb_worst': hb_worst,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    results.sort(key=lambda r: max(r['lb_worst'], r['hb_worst']))
    for r in results:
        pstr = ', '.join(f'{v:.2f}' for v in r['params'])
        worst = max(r['lb_worst'], r['hb_worst'])
        print(f"  {r['name']:20s} | LB={r['lb_worst']:6.1f}  HB={r['hb_worst']:6.1f}  "
              f"worst={worst:6.1f} | [{pstr}]")

    if results:
        best = results[0]
        print(f"\n  BEST: {best['name']}")
        print(f"    LB worst: {best['lb_worst']:.1f} dB")
        print(f"    HB worst: {best['hb_worst']:.1f} dB")


if __name__ == '__main__':
    main()
