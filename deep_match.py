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
Deep matching network optimization with advanced topologies.
Adds LC trap networks that can decouple LB and HB matching.
"""

import sys
import os
import numpy as np
from scipy.optimize import differential_evolution
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
sys.path.insert(0, SCRIPT_DIR)

from matching_network import (
    s11_from_z, apply_series, apply_shunt,
    z_inductor, z_capacitor, z_parallel_lc, z_series_lc,
    TOPOLOGIES, optimize_topology, print_result, Z0,
    LB_MIN, LB_MAX, HB_MIN, HB_MAX,
)


# ============================================================
# Advanced topologies with LC traps
# ============================================================

def match_series_trap_shunt_C(freqs, z_ant, params):
    """Series parallel-LC trap + shunt C.
    Trap resonant between bands creates different reactance in LB vs HB.
    params: [L_trap_nH, C_trap_pF, C_shunt_pF]"""
    Lt, Ct, Cs = params
    z = apply_series(z_ant, z_parallel_lc(freqs, Lt, Ct))
    z = apply_shunt(z, z_capacitor(freqs, Cs))
    return z


def match_shunt_C_series_trap(freqs, z_ant, params):
    """Shunt C + series parallel-LC trap.
    params: [C_shunt_pF, L_trap_nH, C_trap_pF]"""
    Cs, Lt, Ct = params
    z = apply_shunt(z_ant, z_capacitor(freqs, Cs))
    z = apply_series(z, z_parallel_lc(freqs, Lt, Ct))
    return z


def match_shunt_trap_series_L(freqs, z_ant, params):
    """Shunt series-LC trap + series L.
    params: [L_trap_nH, C_trap_pF, L_series_nH]"""
    Lt, Ct, Ls = params
    z = apply_shunt(z_ant, z_series_lc(freqs, Lt, Ct))
    z = apply_series(z, z_inductor(freqs, Ls))
    return z


def match_series_L_shunt_trap(freqs, z_ant, params):
    """Series L + shunt series-LC trap.
    params: [L_series_nH, L_trap_nH, C_trap_pF]"""
    Ls, Lt, Ct = params
    z = apply_series(z_ant, z_inductor(freqs, Ls))
    z = apply_shunt(z, z_series_lc(freqs, Lt, Ct))
    return z


def match_trap_pi(freqs, z_ant, params):
    """Shunt C1, series trap, shunt C2. Pi with trap as series arm.
    params: [C1_pF, L_trap_nH, C_trap_pF, C2_pF]"""
    C1, Lt, Ct, C2 = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C1))
    z = apply_series(z, z_parallel_lc(freqs, Lt, Ct))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    return z


def match_trap_T(freqs, z_ant, params):
    """Series L1, shunt trap, series L2. T with trap as shunt arm.
    params: [L1_nH, L_trap_nH, C_trap_pF, L2_nH]"""
    L1, Lt, Ct, L2 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_series_lc(freqs, Lt, Ct))
    z = apply_series(z, z_inductor(freqs, L2))
    return z


def match_5elem_CLCLC(freqs, z_ant, params):
    """5-element: shunt C1, series L1, shunt C2, series L2, shunt C3.
    params: [C1, L1, C2, L2, C3]"""
    C1, L1, C2, L2, C3 = params
    z = apply_shunt(z_ant, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C3))
    return z


def match_5elem_LCLCL(freqs, z_ant, params):
    """5-element: series L1, shunt C1, series L2, shunt C2, series L3.
    params: [L1, C1, L2, C2, L3]"""
    L1, C1, L2, C2, L3 = params
    z = apply_series(z_ant, z_inductor(freqs, L1))
    z = apply_shunt(z, z_capacitor(freqs, C1))
    z = apply_series(z, z_inductor(freqs, L2))
    z = apply_shunt(z, z_capacitor(freqs, C2))
    z = apply_series(z, z_inductor(freqs, L3))
    return z


def match_dual_trap(freqs, z_ant, params):
    """Series trap1 (tuned near LB) + series trap2 (tuned near HB) + shunt C.
    params: [Lt1, Ct1, Lt2, Ct2, Cs]"""
    Lt1, Ct1, Lt2, Ct2, Cs = params
    z = apply_series(z_ant, z_parallel_lc(freqs, Lt1, Ct1))
    z = apply_series(z, z_parallel_lc(freqs, Lt2, Ct2))
    z = apply_shunt(z, z_capacitor(freqs, Cs))
    return z


# Extended topology registry
ADVANCED_TOPOLOGIES = {
    'series_trap_shunt_C':   (match_series_trap_shunt_C,
                              [(1, 50), (0.1, 30), (0.1, 30)]),
    'shunt_C_series_trap':   (match_shunt_C_series_trap,
                              [(0.1, 30), (1, 50), (0.1, 30)]),
    'shunt_trap_series_L':   (match_shunt_trap_series_L,
                              [(1, 50), (0.1, 30), (0.5, 40)]),
    'series_L_shunt_trap':   (match_series_L_shunt_trap,
                              [(0.5, 40), (1, 50), (0.1, 30)]),
    'trap_pi':               (match_trap_pi,
                              [(0.1, 30), (1, 50), (0.1, 30), (0.1, 30)]),
    'trap_T':                (match_trap_T,
                              [(0.5, 40), (1, 50), (0.1, 30), (0.5, 40)]),
    '5elem_CLCLC':           (match_5elem_CLCLC,
                              [(0.1, 30), (0.5, 40), (0.1, 30), (0.5, 40), (0.1, 30)]),
    '5elem_LCLCL':           (match_5elem_LCLCL,
                              [(0.5, 40), (0.1, 30), (0.5, 40), (0.1, 30), (0.5, 40)]),
    'dual_trap':             (match_dual_trap,
                              [(1, 50), (0.1, 30), (1, 50), (0.1, 30), (0.1, 30)]),
}

# Also include the standard topologies with wider bounds
WIDER_TOPOLOGIES = {
    'pi_CLC_wide':   (TOPOLOGIES['pi_CLC'][0],
                      [(0.1, 30), (0.5, 50), (0.1, 30)]),
    'T_LCL_wide':    (TOPOLOGIES['T_LCL'][0],
                      [(0.5, 50), (0.1, 30), (0.5, 50)]),
    '4elem_LCLC_wide': (TOPOLOGIES['4elem_LCLC'][0],
                        [(0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30)]),
    '4elem_CLCL_wide': (TOPOLOGIES['4elem_CLCL'][0],
                        [(0.1, 30), (0.5, 50), (0.1, 30), (0.5, 50)]),
}

ALL_TOPOLOGIES = {}
ALL_TOPOLOGIES.update(TOPOLOGIES)
ALL_TOPOLOGIES.update(ADVANCED_TOPOLOGIES)
ALL_TOPOLOGIES.update(WIDER_TOPOLOGIES)


def optimize_deep(name, freqs, z_ant, n_runs=10, maxiter=1000):
    """Deep optimization with more runs and iterations."""
    match_fn, bounds = ALL_TOPOLOGIES[name]

    best_cost = 1e9
    best_params = None

    for run in range(n_runs):
        try:
            result = differential_evolution(
                cost_function_balanced, bounds,
                args=(freqs, z_ant, match_fn),
                maxiter=maxiter, tol=1e-10,
                seed=np.random.randint(0, 100000),
                polish=True, workers=1,
                mutation=(0.5, 1.5), recombination=0.9,
                popsize=25,
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x
        except Exception:
            continue

    if best_params is None:
        return None

    z_matched = match_fn(freqs, z_ant, best_params)
    s11 = s11_from_z(z_matched)
    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    return {
        'name': name,
        'params': best_params,
        'lb_worst_s11': float(np.max(s11[lb_mask])),
        'hb_worst_s11': float(np.max(s11[hb_mask])),
        'lb_best_s11': float(np.min(s11[lb_mask])),
        'hb_best_s11': float(np.min(s11[hb_mask])),
        'cost': best_cost,
        's11_matched': s11,
    }


def cost_function_balanced(params, freqs, z_ant, match_fn):
    """Balanced cost targeting -20 dB in both bands.
    Uses max(lb_worst, hb_worst) so optimizer can't sacrifice one band."""
    try:
        z_matched = match_fn(freqs, z_ant, params)
        s11 = s11_from_z(z_matched)
    except (ZeroDivisionError, FloatingPointError, RuntimeWarning):
        return 0.0

    lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
    hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

    lb_worst = np.max(s11[lb_mask]) if np.any(lb_mask) else 0.0
    hb_worst = np.max(s11[hb_mask]) if np.any(hb_mask) else 0.0

    # Primary: minimize the worst of the two bands (minimax)
    # Secondary: add sum for tiebreaking
    return max(lb_worst, hb_worst) + 0.1 * (lb_worst + hb_worst)


def main():
    print("=" * 70)
    print("  DEEP MATCHING OPTIMIZATION")
    print("=" * 70)

    # Load best candidates from focused sweep
    candidates = {
        'cand5_h13_sx26_fo7': os.path.join(RESULTS_DIR, 'candidate_5.npz'),
        'cand6_h13_sx26_fo6': os.path.join(RESULTS_DIR, 'candidate_6.npz'),
        'cand8_h12_sx27_gc12': os.path.join(RESULTS_DIR, 'candidate_8.npz'),
        'cand2_h12_sx27_fo7': os.path.join(RESULTS_DIR, 'candidate_2.npz'),
        'cand7_h12_sx27_hb30': os.path.join(RESULTS_DIR, 'candidate_7.npz'),
    }

    # Topologies to try (all advanced + best standard)
    topo_names = [
        # Standard (re-optimized with wider bounds + more runs)
        'pi_CLC_wide', 'T_LCL_wide', '4elem_LCLC_wide', '4elem_CLCL_wide',
        # Trap-based
        'series_trap_shunt_C', 'shunt_C_series_trap',
        'shunt_trap_series_L', 'series_L_shunt_trap',
        'trap_pi', 'trap_T',
        # 5-element
        '5elem_CLCLC', '5elem_LCLCL',
        # Dual trap
        'dual_trap',
    ]

    all_results = {}

    for cand_name, npz_path in candidates.items():
        if not os.path.exists(npz_path):
            print(f"\n  SKIP {cand_name}: {npz_path} not found")
            continue

        data = np.load(npz_path, allow_pickle=True)
        freqs = data['freqs']
        z_ant = data['z_complex']
        s11_bare = s11_from_z(z_ant)

        lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
        hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

        print(f"\n{'='*70}")
        print(f"  {cand_name}")
        print(f"  Bare: LB worst={np.max(s11_bare[lb_mask]):.1f}dB, "
              f"HB worst={np.max(s11_bare[hb_mask]):.1f}dB")
        print(f"{'='*70}")

        cand_results = []
        for topo in topo_names:
            print(f"  {topo:30s}...", end='', flush=True)
            r = optimize_deep(topo, freqs, z_ant, n_runs=8, maxiter=800)
            if r is None:
                print(" FAILED")
                continue
            worst = max(r['lb_worst_s11'], r['hb_worst_s11'])
            print(f" LB={r['lb_worst_s11']:6.1f}  HB={r['hb_worst_s11']:6.1f}  "
                  f"worst={worst:6.1f}")
            cand_results.append(r)

        # Sort by worst-of-both-bands
        cand_results.sort(key=lambda r: max(r['lb_worst_s11'], r['hb_worst_s11']))
        all_results[cand_name] = cand_results

        print(f"\n  TOP 5 for {cand_name}:")
        for r in cand_results[:5]:
            pstr = ', '.join(f'{v:.2f}' for v in r['params'])
            print(f"    {r['name']:30s} LB={r['lb_worst_s11']:6.1f}  "
                  f"HB={r['hb_worst_s11']:6.1f}  [{pstr}]")

    # Grand summary
    print(f"\n\n{'='*90}")
    print(f"  GRAND SUMMARY — Best matched result per candidate")
    print(f"{'='*90}")
    grand_best = None
    grand_best_worst = 0
    for cand_name, results in all_results.items():
        if not results:
            continue
        best = results[0]
        worst = max(best['lb_worst_s11'], best['hb_worst_s11'])
        print(f"  {cand_name:30s} | {best['name']:25s} | "
              f"LB={best['lb_worst_s11']:6.1f}  HB={best['hb_worst_s11']:6.1f}  "
              f"worst={worst:6.1f}")
        if grand_best is None or worst < grand_best_worst:
            grand_best = (cand_name, best)
            grand_best_worst = worst

    if grand_best:
        cname, r = grand_best
        print(f"\n  OVERALL BEST: {cname} + {r['name']}")
        print(f"    LB worst: {r['lb_worst_s11']:.1f} dB")
        print(f"    HB worst: {r['hb_worst_s11']:.1f} dB")
        pstr = ', '.join(f'{v:.3f}' for v in r['params'])
        print(f"    Params: [{pstr}]")

    # Save summary
    summary = {}
    for cand_name, results in all_results.items():
        summary[cand_name] = [{
            'name': r['name'],
            'lb_worst': r['lb_worst_s11'],
            'hb_worst': r['hb_worst_s11'],
            'params': list(r['params']),
        } for r in results[:5]]

    with open(os.path.join(RESULTS_DIR, 'deep_match_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to results/deep_match_summary.json")


if __name__ == '__main__':
    main()
