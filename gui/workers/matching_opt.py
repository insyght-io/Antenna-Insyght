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

"""QThread worker for matching network optimization.

Runs differential_evolution multiple times to find the best LC component
values for a given matching topology, minimizing worst-case S11 across
both LB and HB frequency bands.
"""

import sys
import os
import traceback
import time

import numpy as np
from scipy.optimize import differential_evolution

from PySide6.QtCore import QThread, Signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from matching_network import (
    s11_from_z, apply_series, apply_shunt,
    z_inductor, z_capacitor,
    LB_MIN, LB_MAX, HB_MIN, HB_MAX,
)
from deep_match import cost_function_balanced, ALL_TOPOLOGIES


# --- Extra topologies not in ALL_TOPOLOGIES ---

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


# Extend topology registry with push_20db extra topologies
EXTRA_TOPOLOGIES = {
    '6elem_LCLCLC': (match_6elem_LCLCLC,
                     [(0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30),
                      (0.5, 50), (0.1, 30)]),
    '6elem_CLCLCL': (match_6elem_CLCLCL,
                     [(0.1, 30), (0.5, 50), (0.1, 30), (0.5, 50),
                      (0.1, 30), (0.5, 50)]),
    '8elem_LCLCLCLC': (match_8elem_LCLCLCLC,
                       [(0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30),
                        (0.5, 50), (0.1, 30), (0.5, 50), (0.1, 30)]),
}

# Combined lookup: ALL_TOPOLOGIES from deep_match + extras defined here
_COMBINED_TOPOLOGIES = {}
_COMBINED_TOPOLOGIES.update(ALL_TOPOLOGIES)
_COMBINED_TOPOLOGIES.update(EXTRA_TOPOLOGIES)

# Also try importing from push_20db in case it has additional topologies
try:
    from push_20db import EXTRA_TOPOLOGIES as PUSH_EXTRA
    _COMBINED_TOPOLOGIES.update(PUSH_EXTRA)
except ImportError:
    pass


class MatchingOptWorker(QThread):
    """Background worker that optimizes matching network component values.

    Uses differential_evolution with multiple restarts to find the global
    optimum for a given topology and antenna impedance data.

    Signals:
        finished(dict)  -- emitted with best params and performance
        progress(str)   -- emitted with status messages during optimization
        error(str)      -- emitted with error message on failure
    """

    finished = Signal(dict)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, topology_name, freqs, z_ant, n_runs=10, parent=None):
        """
        Args:
            topology_name: key into the topology registry (e.g. '6elem_LCLCLC')
            freqs: frequency array in Hz
            z_ant: complex antenna impedance array (same length as freqs)
            n_runs: number of independent DE restarts
        """
        super().__init__(parent)
        self.topology_name = topology_name
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.z_ant = np.asarray(z_ant, dtype=np.complex128)
        self.n_runs = n_runs

    def run(self):
        try:
            name = self.topology_name

            # Look up topology
            if name not in _COMBINED_TOPOLOGIES:
                self.error.emit(
                    f'Unknown topology: {name}\n'
                    f'Available: {sorted(_COMBINED_TOPOLOGIES.keys())}'
                )
                return

            match_fn, bounds = _COMBINED_TOPOLOGIES[name]
            freqs = self.freqs
            z_ant = self.z_ant

            best_cost = 1e9
            best_params = None
            start_time = time.time()

            for run_idx in range(self.n_runs):
                self.progress.emit(
                    f'Optimizing {name}: run {run_idx + 1}/{self.n_runs}...'
                )

                try:
                    result = differential_evolution(
                        cost_function_balanced,
                        bounds,
                        args=(freqs, z_ant, match_fn),
                        maxiter=1500,
                        tol=1e-10,
                        seed=np.random.randint(0, 100000),
                        polish=True,
                        workers=1,
                        mutation=(0.5, 1.5),
                        recombination=0.9,
                        popsize=30,
                    )

                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_params = result.x.copy()

                        # Compute per-band performance for progress reporting
                        z_matched = match_fn(freqs, z_ant, best_params)
                        s11 = s11_from_z(z_matched)
                        lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
                        hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)
                        lb_worst = float(np.max(s11[lb_mask])) if np.any(lb_mask) else 0.0
                        hb_worst = float(np.max(s11[hb_mask])) if np.any(hb_mask) else 0.0

                        self.progress.emit(
                            f'  Run {run_idx + 1}: new best -- '
                            f'LB={lb_worst:.1f} dB, HB={hb_worst:.1f} dB'
                        )

                except Exception:
                    # Individual run failed; continue with remaining runs
                    continue

            elapsed = time.time() - start_time

            if best_params is None:
                self.error.emit(f'All {self.n_runs} optimization runs failed for {name}.')
                return

            # Final evaluation
            z_matched = match_fn(freqs, z_ant, best_params)
            s11_matched = s11_from_z(z_matched)
            lb_mask = (freqs >= LB_MIN) & (freqs <= LB_MAX)
            hb_mask = (freqs >= HB_MIN) & (freqs <= HB_MAX)

            lb_worst = float(np.max(s11_matched[lb_mask])) if np.any(lb_mask) else 0.0
            hb_worst = float(np.max(s11_matched[hb_mask])) if np.any(hb_mask) else 0.0

            self.progress.emit(
                f'Optimization complete in {elapsed:.1f}s. '
                f'LB={lb_worst:.1f} dB, HB={hb_worst:.1f} dB'
            )

            self.finished.emit({
                'topology': name,
                'values': list(best_params),
                'lb_worst': lb_worst,
                'hb_worst': hb_worst,
                'cost': float(best_cost),
                's11_matched_db': s11_matched,
                'z_matched': z_matched,
                'elapsed_s': elapsed,
                'n_runs': self.n_runs,
            })

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f'{str(e)}\n\n{tb}')
