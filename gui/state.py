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

"""Central design state model for the antenna designer GUI."""

import json
import numpy as np
from PySide6.QtCore import QObject, Signal


class DesignState(QObject):
    """Holds all antenna parameters, simulation results, and matching data.

    Emits signals when state changes so panels can update.
    """

    params_changed = Signal()
    simulation_finished = Signal(dict)
    matching_finished = Signal(dict)
    freq_bands_changed = Signal()

    # Default antenna parameters (Design B, R=31)
    DEFAULT_PARAMS = {
        'BOARD_RADIUS': 31.0,
        'SUBSTRATE_ER': 4.4,
        'SUBSTRATE_TAND': 0.02,
        'SUBSTRATE_THICKNESS': 1.6,
        'SHORT_X': -25.0,
        'FEED_OFFSET': 7.0,
        'ELEM_HEIGHT': 14.0,
        'LB_LENGTH': 200.0,
        'LB_SPACING': 10.0,
        'HB_LENGTH': 30.0,
        'HB_ANGLE': 8.0,
        'TRACE_WIDTH': 1.5,
        'GND_CLEARANCE': 10.0,
    }

    DEFAULT_FREQ = {
        'LB_MIN': 791e6,
        'LB_MAX': 960e6,
        'HB_MIN': 1710e6,
        'HB_MAX': 1990e6,
    }

    PARAM_BOUNDS = {
        'BOARD_RADIUS': (15.0, 50.0),
        'SUBSTRATE_ER': (2.0, 10.0),
        'SUBSTRATE_TAND': (0.001, 0.05),
        'SUBSTRATE_THICKNESS': (0.4, 3.2),
        'SHORT_X': (-30.0, 10.0),
        'FEED_OFFSET': (2.0, 15.0),
        'ELEM_HEIGHT': (3.0, 25.0),
        'LB_LENGTH': (50.0, 300.0),
        'LB_SPACING': (3.0, 20.0),
        'HB_LENGTH': (10.0, 60.0),
        'HB_ANGLE': (2.0, 60.0),
        'TRACE_WIDTH': (0.5, 3.0),
        'GND_CLEARANCE': (3.0, 20.0),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = dict(self.DEFAULT_PARAMS)
        self.freq_bands = dict(self.DEFAULT_FREQ)

        # Simulation results (None until first sim)
        self.sim_result = None  # dict: freqs, z_complex, s11_complex, s11_db, vswr

        # Matching network state
        self.matching_topology = None
        self.matching_values = None  # list of (name, value_nH_or_pF)
        self.matching_result = None  # dict: s11_matched_db, worst_lb, worst_hb

    def set_param(self, key, value):
        if key in self.params and self.params[key] != value:
            self.params[key] = value
            self.params_changed.emit()

    def set_freq_band(self, key, value):
        if key in self.freq_bands and self.freq_bands[key] != value:
            self.freq_bands[key] = value
            self.freq_bands_changed.emit()

    def antenna_params_dict(self):
        """Return params dict suitable for BranchedIFA constructor."""
        return {
            'SHORT_X': self.params['SHORT_X'],
            'FEED_OFFSET': self.params['FEED_OFFSET'],
            'ELEM_HEIGHT': self.params['ELEM_HEIGHT'],
            'LB_LENGTH': self.params['LB_LENGTH'],
            'LB_SPACING': self.params['LB_SPACING'],
            'LB_CAP_W': 0.0,
            'LB_CAP_L': 0.0,
            'HB_LENGTH': self.params['HB_LENGTH'],
            'HB_ANGLE': self.params['HB_ANGLE'],
            'TRACE_WIDTH': self.params['TRACE_WIDTH'],
            'GND_CLEARANCE': self.params['GND_CLEARANCE'],
        }

    def save_json(self, path):
        data = {
            'params': self.params,
            'freq_bands': self.freq_bands,
            'matching_topology': self.matching_topology,
            'matching_values': self.matching_values,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_json(self, path):
        with open(path) as f:
            data = json.load(f)
        if 'params' in data:
            self.params.update(data['params'])
        if 'freq_bands' in data:
            self.freq_bands.update(data['freq_bands'])
        if 'matching_topology' in data:
            self.matching_topology = data['matching_topology']
        if 'matching_values' in data:
            self.matching_values = data['matching_values']
        self.params_changed.emit()
        self.freq_bands_changed.emit()

    def save_npz(self, path):
        data = {'params': self.params, 'freq_bands': self.freq_bands}
        if self.sim_result is not None:
            data['freqs'] = self.sim_result['freqs']
            data['z_complex'] = self.sim_result['z_complex']
            data['s11_db'] = self.sim_result['s11_db']
        np.savez(path, **data)
