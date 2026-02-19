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
Optimization loop for the dual-band IFA antenna system.

Multi-phase approach:
  Phase 1: Coarse arm length sweep to find approximate resonance
  Phase 2: Feed offset sweep for impedance matching
  Phase 3: Bayesian optimization over all parameters
  Phase 4: Verification with high-resolution simulation

Uses scikit-optimize for Bayesian optimization with Gaussian Process surrogate.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

from config import (
    TARGETS, NEC2_TARGETS, OPT_BOUNDS, SIM_FREQ_MIN, SIM_FREQ_MAX,
    LB_FREQ_CENTER, HB_FREQ_CENTER, NEC2_FREQ_SCALE, SIM_BACKEND,
    BRANCHED_OPT_BOUNDS, ANTENNA_TOPOLOGY,
    LB_FREQ_MIN, LB_FREQ_MAX, HB_FREQ_MIN, HB_FREQ_MAX,
)
from antenna_model import DualBandIFA, BranchedIFA
from simulate_dispatch import (
    simulate_s11, simulate_s11_safe, run_full_simulation,
    validate_geometry, get_backend,
    simulate_s11_branched, validate_geometry_branched,
    run_full_simulation_branched,
)
from utils import impedance_to_s11, s11_db, find_resonances


def _get_targets(backend=None):
    """Return the correct frequency targets for the active backend.

    NEC2 needs frequency-scaled targets; openEMS uses real PCB targets.
    """
    b = backend or SIM_BACKEND
    if b == 'nec2':
        return NEC2_TARGETS
    return TARGETS


def evaluate_band(antenna, band, n_freq=31, backend=None):
    """Evaluate one band's performance. Returns a dict with metrics.

    Uses the correct frequency targets for the active backend.
    """
    targets = _get_targets(backend)
    band_targets = targets[band]
    band_freqs = np.linspace(band_targets['freq_min'], band_targets['freq_max'], n_freq)

    result = simulate_s11(antenna, band, band_freqs, backend=backend)
    s11_vals = result['s11_db']
    vswr_vals = result['vswr']

    worst_s11 = np.max(s11_vals)
    best_s11 = np.min(s11_vals)
    worst_vswr = np.max(vswr_vals)
    mean_s11 = np.mean(s11_vals)

    # Also check for resonance in a wider band
    b = backend or SIM_BACKEND
    if b == 'nec2':
        wide_min = SIM_FREQ_MIN * NEC2_FREQ_SCALE
        wide_max = SIM_FREQ_MAX * NEC2_FREQ_SCALE
    else:
        wide_min = SIM_FREQ_MIN
        wide_max = SIM_FREQ_MAX
    wide_freqs = np.linspace(wide_min, wide_max, 51)
    wide_result = simulate_s11(antenna, band, wide_freqs, backend=backend)
    resonances = find_resonances(wide_result['freqs'], wide_result['s11_db'], -3.0)

    return {
        'worst_s11': worst_s11,
        'best_s11': best_s11,
        'worst_vswr': worst_vswr,
        'mean_s11': mean_s11,
        'resonances': resonances,
        'target_met': worst_s11 < band_targets['s11_db'],
    }


def objective(params_dict, verbose=True, backend=None):
    """Compute optimization objective.

    Returns a scalar cost (lower is better).
    0 = all targets met. Positive = how far from targets.

    Validates geometry before simulation to avoid crashes.
    """
    try:
        antenna = DualBandIFA(params_dict)
        geom = antenna.generate(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  Geometry error: {e}")
        return 100.0

    total_cost = 0.0
    targets = _get_targets(backend)

    for band in ['lowband', 'highband']:
        ok, reason = validate_geometry(antenna, band, backend=backend)
        if not ok:
            if verbose:
                print(f"  Geometry invalid ({band}): {reason}")
            total_cost += 50.0
            continue

        band_targets = targets[band]
        band_freqs = np.linspace(band_targets['freq_min'], band_targets['freq_max'], 21)

        try:
            result = simulate_s11(antenna, band, band_freqs, verbose=verbose, backend=backend)
        except Exception as e:
            if verbose:
                print(f"  Sim error ({band}): {e}")
            total_cost += 50.0
            continue

        # Check for solver failure indicators
        if np.any(np.real(result['z_complex']) < -900):
            if verbose:
                print(f"  Solver failure ({band})")
            total_cost += 50.0
            continue

        worst_s11 = np.max(result['s11_db'])
        threshold = band_targets['s11_db']

        if worst_s11 > threshold:
            total_cost += (worst_s11 - threshold) ** 2
        else:
            total_cost -= 0.1 * abs(worst_s11 - threshold)

    return total_cost


def phase1_arm_sweep(verbose=True, backend=None):
    """Phase 1: Sweep arm lengths AND meander spacing to find approximate resonance.

    Sweeps arm length together with meander spacing because tighter spacing
    allows fitting more trace in the half-circle boundary. This is critical
    for the low band where we need long traces to resonate at 700-960 MHz.

    Returns best parameters found.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Arm Length + Meander Spacing Sweep")
    print("=" * 60)

    best_cost = float('inf')
    best_params = {}
    results_log = []

    # Low band: sweep total length (longer trace = lower frequency)
    lb_lengths = np.arange(85, 285, 15)
    # Low band: sweep meander spacing (tighter = more trace fits)
    lb_spacings = [2.0, 3.0, 4.0, 5.0]
    # High band: sweep total length
    hb_lengths = np.arange(35, 95, 10)

    total_evals = len(lb_lengths) * len(lb_spacings) * len(hb_lengths)
    eval_num = 0

    for lb_len in lb_lengths:
        for lb_sp in lb_spacings:
            for hb_len in hb_lengths:
                eval_num += 1
                params = {
                    'LB_TOTAL_LENGTH': lb_len,
                    'LB_MEANDER_SPACING': lb_sp,
                    'HB_TOTAL_LENGTH': hb_len,
                }

                cost = objective(params, verbose=False, backend=backend)

                if verbose and eval_num % 20 == 0:
                    print(f"  [{eval_num}/{total_evals}] LB={lb_len:.0f}mm(sp={lb_sp:.0f}), "
                          f"HB={hb_len:.0f}mm -> cost={cost:.2f}")

                results_log.append({
                    'lb_length': lb_len,
                    'lb_spacing': lb_sp,
                    'hb_length': hb_len,
                    'cost': cost,
                })

                if cost < best_cost:
                    best_cost = cost
                    best_params = params.copy()
                    if verbose:
                        print(f"  *** New best: LB={lb_len:.0f}mm(sp={lb_sp:.0f}), "
                              f"HB={hb_len:.0f}mm, cost={cost:.2f}")

    print(f"\nPhase 1 best: {best_params}, cost={best_cost:.2f}")
    return best_params, results_log


def phase2_feed_sweep(base_params, verbose=True, backend=None):
    """Phase 2: Sweep feed offsets for impedance matching.

    Returns best parameters found.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: Feed Offset Sweep")
    print("=" * 60)

    best_cost = float('inf')
    best_params = base_params.copy()
    results_log = []

    lb_offsets = np.arange(2.0, 6.5, 0.5)
    hb_offsets = np.arange(2.0, 5.5, 0.5)

    for lb_off in lb_offsets:
        for hb_off in hb_offsets:
            params = base_params.copy()
            params['LB_FEED_OFFSET'] = lb_off
            params['HB_FEED_OFFSET'] = hb_off

            cost = objective(params, verbose=False, backend=backend)

            results_log.append({
                'lb_feed_offset': lb_off,
                'hb_feed_offset': hb_off,
                'cost': cost,
            })

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
                if verbose:
                    print(f"  *** New best: LB_off={lb_off:.1f}, HB_off={hb_off:.1f}, "
                          f"cost={cost:.2f}")

    print(f"\nPhase 2 best: cost={best_cost:.2f}")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    return best_params, results_log


def phase3_bayesian(base_params, n_calls=30, verbose=True, backend=None):
    """Phase 3: Bayesian optimization over all parameters.

    Uses scikit-optimize with Gaussian Process surrogate model.
    """
    print("\n" + "=" * 60)
    print(f"  PHASE 3: Bayesian Optimization ({n_calls} iterations)")
    print("=" * 60)

    from skopt import gp_minimize
    from skopt.space import Real

    # Define search space — expand arm length bounds for meander antennas
    dimensions = [
        Real(85.0, 280.0, name='LB_TOTAL_LENGTH'),
        Real(2.0, 6.0, name='LB_FEED_OFFSET'),
        Real(3.0, 6.0, name='LB_ELEMENT_HEIGHT'),
        Real(3.0, 6.0, name='LB_MEANDER_SPACING'),
        Real(35.0, 90.0, name='HB_TOTAL_LENGTH'),
        Real(2.0, 5.0, name='HB_FEED_OFFSET'),
        Real(3.0, 6.0, name='HB_ELEMENT_HEIGHT'),
        Real(2.5, 5.0, name='HB_MEANDER_SPACING'),
    ]

    param_names = [d.name for d in dimensions]

    # Starting point from phase 2
    x0 = [base_params.get(name, d.low + (d.high - d.low) / 2)
           for name, d in zip(param_names, dimensions)]

    eval_count = [0]

    def skopt_objective(x):
        eval_count[0] += 1
        params = {name: val for name, val in zip(param_names, x)}
        cost = objective(params, verbose=False, backend=backend)
        if verbose and eval_count[0] % 5 == 0:
            print(f"  [{eval_count[0]}/{n_calls}] cost={cost:.2f} | "
                  f"LB={params['LB_TOTAL_LENGTH']:.0f}mm, "
                  f"HB={params['HB_TOTAL_LENGTH']:.0f}mm")
        return cost

    result = gp_minimize(
        skopt_objective,
        dimensions,
        x0=x0,
        n_calls=n_calls,
        n_random_starts=max(5, n_calls // 3),
        random_state=42,
        verbose=False,
    )

    best_params = {name: val for name, val in zip(param_names, result.x)}
    best_cost = result.fun

    print(f"\nPhase 3 best: cost={best_cost:.2f}")
    for k, v in best_params.items():
        print(f"  {k} = {v:.2f}")

    return best_params, result


def phase4_verify(params, output_dir='results', verbose=True, backend=None):
    """Phase 4: High-resolution verification of final design."""
    print("\n" + "=" * 60)
    print("  PHASE 4: Verification (high resolution)")
    print("=" * 60)

    results = run_full_simulation(params, output_dir=output_dir, quick=False, backend=backend)

    # Generate PDF report (also called by run_full_simulation, but ensure
    # it has the optimization params available)
    try:
        from report import generate_report
        generate_report(results_dir=output_dir, params=params)
    except Exception as e:
        print(f"WARNING: Report generation failed: {e}")

    return results


def run_optimization(phases='all', n_bayesian=30, output_dir='results', backend=None,
                     initial_params=None):
    """Run the full optimization pipeline.

    Args:
        phases: 'all', '1', '12', '123', or '4'
        n_bayesian: number of Bayesian optimization iterations
        output_dir: directory for results
        backend: 'nec2', 'openems', or 'hybrid' (None = use config default)
        initial_params: dict or path to JSON with starting parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    b = backend or SIM_BACKEND
    is_hybrid = (b == 'hybrid')

    log = {
        'start_time': datetime.now().isoformat(),
        'backend': b,
        'phases': {},
    }

    best_params = {}

    # Load initial parameters if provided
    if initial_params:
        if isinstance(initial_params, str):
            with open(initial_params) as f:
                best_params = json.load(f)
            print(f"Loaded initial params from {initial_params}")
        elif isinstance(initial_params, dict):
            best_params = initial_params.copy()
        print(f"Starting from: {best_params}")
        log['initial_params'] = {k: float(v) for k, v in best_params.items()}

    # Hybrid mode: phases 1-2 use NEC2 (fast), phases 3-4 use openEMS (accurate)
    coarse_backend = 'nec2' if is_hybrid else (b if b != 'hybrid' else None)
    fine_backend = 'openems' if is_hybrid else (b if b != 'hybrid' else None)

    if is_hybrid:
        print(f"\n  Hybrid mode: Phases 1-2 via NEC2, Phases 3-4 via openEMS")

    if '1' in phases or phases == 'all':
        p1_params, p1_log = phase1_arm_sweep(backend=coarse_backend)
        best_params.update(p1_params)
        log['phases']['phase1'] = {
            'best_params': {k: float(v) for k, v in p1_params.items()},
            'n_evals': len(p1_log),
            'backend': coarse_backend,
        }

    if '2' in phases or phases == 'all':
        p2_params, p2_log = phase2_feed_sweep(best_params, backend=coarse_backend)
        best_params.update(p2_params)
        log['phases']['phase2'] = {
            'best_params': {k: float(v) for k, v in p2_params.items()},
            'n_evals': len(p2_log),
            'backend': coarse_backend,
        }

    if '3' in phases or phases == 'all':
        p3_params, p3_result = phase3_bayesian(best_params, n_calls=n_bayesian,
                                                backend=fine_backend)
        best_params.update(p3_params)
        log['phases']['phase3'] = {
            'best_params': {k: float(v) for k, v in p3_params.items()},
            'best_cost': float(p3_result.fun),
            'n_evals': n_bayesian,
            'backend': fine_backend,
        }

    if '4' in phases or phases == 'all':
        results = phase4_verify(best_params, output_dir=output_dir, backend=fine_backend)
        log['phases']['phase4'] = {'status': 'completed', 'backend': fine_backend}

    elapsed = time.time() - start_time
    log['elapsed_seconds'] = elapsed
    log['best_params'] = {k: float(v) for k, v in best_params.items()}

    # Save optimization log
    log_path = os.path.join(output_dir, 'optimization_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nOptimization log saved to {log_path}")

    # Save best parameters
    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump({k: float(v) for k, v in best_params.items()}, f, indent=2)
    print(f"Best parameters saved to {params_path}")

    print(f"\nTotal optimization time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("\nBest parameters:")
    for k, v in sorted(best_params.items()):
        print(f"  {k} = {v:.2f}")

    return best_params


# ============================================================
# Branched IFA Optimization
# ============================================================

def objective_branched(params_dict, verbose=True, explore=False):
    """Compute optimization objective for branched IFA.

    One simulation covers both bands. Returns scalar cost (lower = better).

    Args:
        explore: if True, use minimal timesteps (~80s vs ~3min for fast)
    """
    try:
        antenna = BranchedIFA(params_dict)
        antenna.generate_geometry()
    except Exception as e:
        if verbose:
            print(f"  Geometry error: {e}")
        return 100.0

    ok, reason = validate_geometry_branched(antenna)
    if not ok:
        if verbose:
            print(f"  Geometry invalid: {reason}")
        return 50.0

    # Single FDTD run covering both bands
    eval_freqs = np.linspace(SIM_FREQ_MIN, SIM_FREQ_MAX, 101)
    try:
        result = simulate_s11_branched(antenna, freqs=eval_freqs,
                                        verbose=False,
                                        fast=(not explore),
                                        explore=explore)
    except Exception as e:
        if verbose:
            print(f"  Sim error: {e}")
        return 50.0

    # Check for solver failure
    if np.any(np.real(result['z_complex']) < -900):
        if verbose:
            print(f"  Solver failure")
        return 50.0

    total_cost = 0.0

    # LB region cost (700-960 MHz)
    lb_mask = (result['freqs'] >= LB_FREQ_MIN) & (result['freqs'] <= LB_FREQ_MAX)
    if np.any(lb_mask):
        lb_worst_s11 = np.max(result['s11_db'][lb_mask])
        threshold = TARGETS['lowband']['s11_db']
        if lb_worst_s11 > threshold:
            total_cost += (lb_worst_s11 - threshold) ** 2
        else:
            total_cost -= 0.1 * abs(lb_worst_s11 - threshold)

    # HB region cost (1710-2170 MHz)
    hb_mask = (result['freqs'] >= HB_FREQ_MIN) & (result['freqs'] <= HB_FREQ_MAX)
    if np.any(hb_mask):
        hb_worst_s11 = np.max(result['s11_db'][hb_mask])
        threshold = TARGETS['highband']['s11_db']
        if hb_worst_s11 > threshold:
            total_cost += (hb_worst_s11 - threshold) ** 2
        else:
            total_cost -= 0.1 * abs(hb_worst_s11 - threshold)

    if verbose:
        lb_ws = np.max(result['s11_db'][lb_mask]) if np.any(lb_mask) else 0
        hb_ws = np.max(result['s11_db'][hb_mask]) if np.any(hb_mask) else 0
        print(f"  LB worst={lb_ws:.1f}dB, HB worst={hb_ws:.1f}dB, cost={total_cost:.2f}")

    return total_cost


def phase1_branched_sweep(verbose=True):
    """Phase 1: Grid search over key branched IFA parameters.

    Uses explore mode (~80s/sim) for coarse landscape mapping.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: Branched IFA Parameter Sweep (explore mode)")
    print("=" * 60)

    best_cost = float('inf')
    best_params = {}
    results_log = []

    # Key parameters to sweep (explore mode ~80s/sim)
    short_xs = [-4.0, 0.0, 4.0]
    feed_offsets = [3.0, 5.0, 7.0]
    elem_heights = [5.0, 7.0, 10.0]
    hb_angles = [20.0, 40.0, 60.0]

    total_evals = len(short_xs) * len(feed_offsets) * len(elem_heights) * len(hb_angles)
    eval_num = 0

    print(f"  {total_evals} evaluations, ~{total_evals * 80 / 60:.0f} min estimated")

    for sx in short_xs:
        for fo in feed_offsets:
            for eh in elem_heights:
                for ha in hb_angles:
                    eval_num += 1
                    params = {
                        'SHORT_X': sx,
                        'FEED_OFFSET': fo,
                        'ELEM_HEIGHT': eh,
                        'HB_ANGLE': ha,
                    }

                    cost = objective_branched(params, verbose=False, explore=True)

                    if verbose:
                        print(f"  [{eval_num}/{total_evals}] sx={sx:.0f}, fo={fo:.0f}, "
                              f"eh={eh:.0f}, ha={ha:.0f} -> cost={cost:.2f}")

                    results_log.append({**params, 'cost': cost})

                    if cost < best_cost:
                        best_cost = cost
                        best_params = params.copy()
                        if verbose:
                            print(f"  *** New best: cost={cost:.2f}")

    print(f"\nPhase 1 best: {best_params}, cost={best_cost:.2f}")
    return best_params, results_log


def phase2_branched_feed(base_params, verbose=True):
    """Phase 2: Feed offset and HB length refinement (explore mode)."""
    print("\n" + "=" * 60)
    print("  PHASE 2: Branched Feed + HB Length Refinement (explore)")
    print("=" * 60)

    best_cost = float('inf')
    best_params = base_params.copy()
    results_log = []

    feed_offsets = np.arange(2.0, 8.1, 1.0)
    hb_lengths = np.arange(18.0, 42.1, 4.0)

    total_evals = len(feed_offsets) * len(hb_lengths)
    eval_num = 0
    print(f"  {total_evals} evaluations, ~{total_evals * 80 / 60:.0f} min estimated")

    for fo in feed_offsets:
        for hl in hb_lengths:
            eval_num += 1
            params = base_params.copy()
            params['FEED_OFFSET'] = fo
            params['HB_LENGTH'] = hl

            cost = objective_branched(params, verbose=False, explore=True)
            results_log.append({**params, 'cost': cost})

            if verbose:
                marker = ""
                if cost < best_cost:
                    marker = " ***"
                print(f"  [{eval_num}/{total_evals}] fo={fo:.1f}, hb_len={hl:.0f} -> cost={cost:.2f}{marker}")

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()

    print(f"\nPhase 2 best: cost={best_cost:.2f}")
    for k, v in sorted(best_params.items()):
        print(f"  {k} = {v:.2f}")
    return best_params, results_log


def phase3_branched_bayesian(base_params, n_calls=30, verbose=True):
    """Phase 3: Bayesian optimization over all branched IFA parameters."""
    print("\n" + "=" * 60)
    print(f"  PHASE 3: Bayesian Optimization ({n_calls} iterations)")
    print("=" * 60)

    from skopt import gp_minimize
    from skopt.space import Real

    bounds = BRANCHED_OPT_BOUNDS

    # Build dimensions, skipping fixed-value bounds (lower == upper)
    all_param_names = [
        'SHORT_X', 'FEED_OFFSET', 'ELEM_HEIGHT', 'LB_LENGTH', 'LB_SPACING',
        'LB_CAP_W', 'LB_CAP_L', 'HB_LENGTH', 'HB_ANGLE', 'GND_CLEARANCE',
        'TRACE_WIDTH',
    ]
    dimensions = []
    opt_param_names = []
    fixed_params = {}
    for name in all_param_names:
        lo, hi = bounds[name]
        if lo == hi:
            fixed_params[name] = lo
        else:
            dimensions.append(Real(lo, hi, name=name))
            opt_param_names.append(name)

    if fixed_params:
        print(f"  Fixed params (not optimized): {fixed_params}")

    x0 = [max(d.low, min(d.high, base_params.get(name, d.low + (d.high - d.low) / 2)))
           for name, d in zip(opt_param_names, dimensions)]

    eval_count = [0]

    def skopt_objective(x):
        eval_count[0] += 1
        params = {name: val for name, val in zip(opt_param_names, x)}
        params.update(fixed_params)
        cost = objective_branched(params, verbose=False)
        if verbose and eval_count[0] % 5 == 0:
            print(f"  [{eval_count[0]}/{n_calls}] cost={cost:.2f} | "
                  f"LB_len={params['LB_LENGTH']:.0f}, HB_len={params['HB_LENGTH']:.0f}, "
                  f"fo={params['FEED_OFFSET']:.1f}")
        return cost

    result = gp_minimize(
        skopt_objective, dimensions,
        x0=x0, n_calls=n_calls,
        n_random_starts=max(5, n_calls // 3),
        random_state=42, verbose=False,
    )

    best_params = {name: val for name, val in zip(opt_param_names, result.x)}
    best_params.update(fixed_params)
    best_cost = result.fun

    print(f"\nPhase 3 best: cost={best_cost:.2f}")
    for k, v in best_params.items():
        print(f"  {k} = {v:.2f}")

    return best_params, result


def phase4_branched_verify(params, output_dir='results', verbose=True):
    """Phase 4: Full verification of branched IFA."""
    print("\n" + "=" * 60)
    print("  PHASE 4: Verification (high resolution)")
    print("=" * 60)

    results = run_full_simulation_branched(params, output_dir=output_dir, quick=False)

    try:
        from report import generate_report_branched
        generate_report_branched(results_dir=output_dir, params=params)
    except Exception as e:
        print(f"WARNING: Report generation failed: {e}")

    return results


def run_optimization_branched(phases='all', n_bayesian=30, output_dir='results',
                               initial_params=None):
    """Run optimization pipeline for branched IFA topology."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    log = {
        'start_time': datetime.now().isoformat(),
        'topology': 'branched',
        'backend': 'openems',
        'phases': {},
    }

    best_params = {}

    if initial_params:
        if isinstance(initial_params, str):
            with open(initial_params) as f:
                best_params = json.load(f)
            print(f"Loaded initial params from {initial_params}")
        elif isinstance(initial_params, dict):
            best_params = initial_params.copy()
        print(f"Starting from: {best_params}")
        log['initial_params'] = {k: float(v) for k, v in best_params.items()}

    if '1' in phases or phases == 'all':
        p1_params, p1_log = phase1_branched_sweep()
        best_params.update(p1_params)
        log['phases']['phase1'] = {
            'best_params': {k: float(v) for k, v in p1_params.items()},
            'n_evals': len(p1_log),
        }

    if '2' in phases or phases == 'all':
        p2_params, p2_log = phase2_branched_feed(best_params)
        best_params.update(p2_params)
        log['phases']['phase2'] = {
            'best_params': {k: float(v) for k, v in p2_params.items()},
            'n_evals': len(p2_log),
        }

    if '3' in phases or phases == 'all':
        p3_params, p3_result = phase3_branched_bayesian(
            best_params, n_calls=n_bayesian)
        best_params.update(p3_params)
        log['phases']['phase3'] = {
            'best_params': {k: float(v) for k, v in p3_params.items()},
            'best_cost': float(p3_result.fun),
            'n_evals': n_bayesian,
        }

    if '4' in phases or phases == 'all':
        results = phase4_branched_verify(best_params, output_dir=output_dir)
        log['phases']['phase4'] = {'status': 'completed'}

    elapsed = time.time() - start_time
    log['elapsed_seconds'] = elapsed
    log['best_params'] = {k: float(v) for k, v in best_params.items()}

    log_path = os.path.join(output_dir, 'optimization_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nOptimization log saved to {log_path}")

    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump({k: float(v) for k, v in best_params.items()}, f, indent=2)
    print(f"Best parameters saved to {params_path}")

    print(f"\nTotal optimization time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("\nBest parameters:")
    for k, v in sorted(best_params.items()):
        print(f"  {k} = {v:.2f}")

    return best_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optimize IFA antenna')
    parser.add_argument('--phases', default='all',
                        help='Which phases to run: "all", "1", "12", "123", "4"')
    parser.add_argument('--n-bayesian', type=int, default=30,
                        help='Number of Bayesian optimization iterations')
    parser.add_argument('--output', default='results',
                        help='Output directory')
    parser.add_argument('--backend', choices=['nec2', 'openems', 'hybrid'], default=None,
                        help='Simulation backend (default: from config.py)')
    parser.add_argument('--topology', choices=['dual_ifa', 'branched'], default=None,
                        help='Antenna topology (default: from config.py)')
    parser.add_argument('--initial-params', default=None,
                        help='Path to JSON with initial parameters (e.g. results/best_params.json)')
    args = parser.parse_args()

    topology = args.topology or ANTENNA_TOPOLOGY
    backend_name = args.backend or SIM_BACKEND

    print("=" * 60)
    if topology == 'branched':
        print("  Branched IFA Antenna Optimizer")
    else:
        print("  Dual-Band IFA Antenna Optimizer")
    print(f"  Backend: {backend_name}")
    print("=" * 60)

    if topology == 'branched':
        best = run_optimization_branched(
            phases=args.phases,
            n_bayesian=args.n_bayesian,
            output_dir=args.output,
            initial_params=args.initial_params,
        )
    else:
        best = run_optimization(
            phases=args.phases,
            n_bayesian=args.n_bayesian,
            output_dir=args.output,
            backend=args.backend,
            initial_params=args.initial_params,
        )
