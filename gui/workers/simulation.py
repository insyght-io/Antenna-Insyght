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

"""QThread worker for running openEMS FDTD simulation in the background.

Creates a BranchedIFA from parameters, runs simulate_s11_branched,
and emits the result dict or error string.
"""

import sys
import os
import traceback

from PySide6.QtCore import QThread, Signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antenna_model import BranchedIFA
from simulate_openems import simulate_s11_branched


class SimulationWorker(QThread):
    """Background worker that runs an openEMS FDTD S11 simulation.

    Modes control the number of FDTD timesteps:
        'explore'  -- 30k steps, ~50s, rough S11 shape
        'standard' -- 80k steps, ~100s, -15dB convergence
        'full'     -- 200k steps, ~200s, -40dB convergence

    Signals:
        finished(dict)  -- emitted with simulation result dict on success
        error(str)      -- emitted with error message on failure
        progress(str)   -- emitted with status messages during execution
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params_dict, mode='standard', parent=None):
        """
        Args:
            params_dict: dict of antenna parameters for BranchedIFA constructor.
                         Keys: SHORT_X, FEED_OFFSET, ELEM_HEIGHT, LB_LENGTH,
                               LB_SPACING, HB_LENGTH, HB_ANGLE, TRACE_WIDTH,
                               GND_CLEARANCE, etc.
            mode: 'explore', 'standard', or 'full'
        """
        super().__init__(parent)
        self.params_dict = dict(params_dict)
        self.mode = mode

    def run(self):
        try:
            self.progress.emit(f'Creating antenna geometry...')

            antenna = BranchedIFA(self.params_dict)
            antenna.generate_geometry()

            # Determine fast/explore flags from mode
            fast = False
            explore = False
            if self.mode == 'explore':
                explore = True
                self.progress.emit('Running FDTD (explore mode, ~30k steps)...')
            elif self.mode == 'standard':
                fast = True
                self.progress.emit('Running FDTD (standard mode, ~80k steps)...')
            elif self.mode == 'full':
                self.progress.emit('Running FDTD (full accuracy, ~200k steps)...')
            else:
                fast = True
                self.progress.emit(f'Running FDTD ({self.mode} mode)...')

            gnd_clearance = self.params_dict.get('GND_CLEARANCE', None)

            result = simulate_s11_branched(
                antenna,
                freqs=None,  # use default SIM_FREQUENCIES from config
                verbose=False,
                fast=fast,
                explore=explore,
                gnd_clearance=gnd_clearance,
            )

            self.progress.emit('Simulation complete.')
            self.finished.emit(result)

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f'{str(e)}\n\n{tb}')
