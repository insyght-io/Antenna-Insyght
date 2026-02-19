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

"""QThread worker for antenna geometry optimization using differential_evolution.

Optimizes BranchedIFA parameters (SHORT_X, FEED_OFFSET, ELEM_HEIGHT, etc.)
to minimize worst-case S11 across both LB and HB frequency bands.
"""

import sys
import os
import traceback
import time

import numpy as np
from scipy.optimize import differential_evolution

from PySide6.QtCore import QThread, Signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antenna_model import BranchedIFA
from simulate_openems import simulate_s11_branched
from config import (
    BRANCHED_OPT_BOUNDS,
    LB_FREQ_MIN, LB_FREQ_MAX,
    HB_FREQ_MIN, HB_FREQ_MAX,
    SIM_FREQ_MIN, SIM_FREQ_MAX,
    TARGETS,
)

# Parameter names in the order used by the optimizer vector
PARAM_NAMES = [
    'SHORT_X', 'FEED_OFFSET', 'ELEM_HEIGHT',
    'HB_LENGTH', 'HB_ANGLE',
    'GND_CLEARANCE', 'TRACE_WIDTH',
]


class OptimizationWorker(QThread):
    """Background worker that runs differential_evolution to optimize
    antenna geometry parameters for dual-band S11 performance.

    Signals:
        finished(dict)  -- emitted with best params and performance on completion
        iteration(dict) -- emitted after each function evaluation with current best
        error(str)      -- emitted with error message on failure
    """

    finished = Signal(dict)
    iteration = Signal(dict)
    error = Signal(str)

    def __init__(self, bounds_dict=None, parent=None):
        """
        Args:
            bounds_dict: dict mapping param name to (min, max) tuple.
                         Defaults to BRANCHED_OPT_BOUNDS from config.
                         Only keys in PARAM_NAMES are used; others are ignored.
        """
        super().__init__(parent)

        # Build bounds from the provided dict or defaults
        src = bounds_dict if bounds_dict is not None else BRANCHED_OPT_BOUNDS
        self.bounds = [src.get(name, BRANCHED_OPT_BOUNDS.get(name, (0, 1)))
                       for name in PARAM_NAMES]

        self._best_cost = 1e9
        self._best_params = None
        self._eval_count = 0
        self._start_time = None

    def _params_to_dict(self, x):
        """Convert optimizer vector to BranchedIFA params dict."""
        return {name: float(val) for name, val in zip(PARAM_NAMES, x)}

    def _objective(self, x):
        """Cost function for differential_evolution.

        Creates a BranchedIFA, runs a fast FDTD sim, and returns a scalar cost
        based on worst-case S11 in each band. Lower is better.
        """
        params_dict = self._params_to_dict(x)
        self._eval_count += 1

        try:
            antenna = BranchedIFA(params_dict)
            antenna.generate_geometry()
        except Exception:
            return 100.0

        # Use explore mode for speed during optimization
        eval_freqs = np.linspace(SIM_FREQ_MIN, SIM_FREQ_MAX, 101)
        try:
            result = simulate_s11_branched(
                antenna, freqs=eval_freqs,
                verbose=False, fast=False, explore=True,
                gnd_clearance=params_dict.get('GND_CLEARANCE'),
            )
        except Exception:
            return 50.0

        # Check for solver failure
        if np.any(np.real(result['z_complex']) < -900):
            return 50.0

        s11_db = result['s11_db']
        freqs = result['freqs']
        total_cost = 0.0

        # LB cost
        lb_mask = (freqs >= LB_FREQ_MIN) & (freqs <= LB_FREQ_MAX)
        lb_worst = np.max(s11_db[lb_mask]) if np.any(lb_mask) else 0.0
        lb_threshold = TARGETS['lowband']['s11_db']
        if lb_worst > lb_threshold:
            total_cost += (lb_worst - lb_threshold) ** 2
        else:
            total_cost -= 0.1 * abs(lb_worst - lb_threshold)

        # HB cost
        hb_mask = (freqs >= HB_FREQ_MIN) & (freqs <= HB_FREQ_MAX)
        hb_worst = np.max(s11_db[hb_mask]) if np.any(hb_mask) else 0.0
        hb_threshold = TARGETS['highband']['s11_db']
        if hb_worst > hb_threshold:
            total_cost += (hb_worst - hb_threshold) ** 2
        else:
            total_cost -= 0.1 * abs(hb_worst - hb_threshold)

        # Track best and emit iteration update
        if total_cost < self._best_cost:
            self._best_cost = total_cost
            self._best_params = params_dict.copy()

            elapsed = time.time() - self._start_time
            self.iteration.emit({
                'eval': self._eval_count,
                'cost': total_cost,
                'lb_worst': float(lb_worst),
                'hb_worst': float(hb_worst),
                'params': params_dict,
                'elapsed_s': elapsed,
            })

        return total_cost

    def run(self):
        try:
            self._start_time = time.time()
            self._eval_count = 0
            self._best_cost = 1e9
            self._best_params = None

            result = differential_evolution(
                self._objective,
                self.bounds,
                maxiter=200,
                tol=1e-6,
                seed=np.random.randint(0, 100000),
                polish=False,
                workers=1,
                mutation=(0.5, 1.5),
                recombination=0.9,
                popsize=15,
            )

            elapsed = time.time() - self._start_time
            best_params = self._params_to_dict(result.x)

            self.finished.emit({
                'params': best_params,
                'cost': float(result.fun),
                'evaluations': self._eval_count,
                'elapsed_s': elapsed,
                'success': result.success,
                'message': result.message,
            })

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f'{str(e)}\n\n{tb}')
