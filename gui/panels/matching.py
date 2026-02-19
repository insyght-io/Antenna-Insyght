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

"""Matching network design panel for the antenna designer GUI.

Provides topology selection, component value editing with live S11 preview,
E24 rounding, and optimization triggering.
"""

import sys
import os
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import matching network primitives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from matching_network import s11_from_z, apply_series, apply_shunt, z_inductor, z_capacitor

# E24 standard values
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]

# Available topologies with their component sequences.
# Each entry: (topology_name, [(ref, type, unit), ...])
# type is 'L' (series inductor) or 'C' (shunt capacitor)
# The chain alternates: first element of name indicates first operation.
TOPOLOGY_DEFS = {
    'series_L_shunt_C': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
    ],
    'shunt_C_series_L': [
        ('C1', 'C', 'pF'),
        ('L1', 'L', 'nH'),
    ],
    'pi_CLC': [
        ('C1', 'C', 'pF'),
        ('L1', 'L', 'nH'),
        ('C2', 'C', 'pF'),
    ],
    'T_LCL': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
        ('L2', 'L', 'nH'),
    ],
    '4elem_LCLC': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C2', 'C', 'pF'),
    ],
    '4elem_CLCL': [
        ('C1', 'C', 'pF'),
        ('L1', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L2', 'L', 'nH'),
    ],
    '5elem_LCLCL': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L3', 'L', 'nH'),
    ],
    '5elem_CLCLC': [
        ('C1', 'C', 'pF'),
        ('L1', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C3', 'C', 'pF'),
    ],
    '6elem_LCLCLC': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L3', 'L', 'nH'),
        ('C3', 'C', 'pF'),
    ],
    '6elem_CLCLCL': [
        ('C1', 'C', 'pF'),
        ('L1', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C3', 'C', 'pF'),
        ('L3', 'L', 'nH'),
    ],
    '8elem_LCLCLCLC': [
        ('L1', 'L', 'nH'),
        ('C1', 'C', 'pF'),
        ('L2', 'L', 'nH'),
        ('C2', 'C', 'pF'),
        ('L3', 'L', 'nH'),
        ('C3', 'C', 'pF'),
        ('L4', 'L', 'nH'),
        ('C4', 'C', 'pF'),
    ],
}

# Ordered list for the combo box
TOPOLOGY_NAMES = [
    'series_L_shunt_C', 'shunt_C_series_L', 'pi_CLC', 'T_LCL',
    '4elem_LCLC', '4elem_CLCL', '5elem_LCLCL', '5elem_CLCLC',
    '6elem_LCLCLC', '6elem_CLCLCL', '8elem_LCLCLCLC',
]


def round_to_e24(value):
    """Round a component value to the nearest E24 standard value."""
    if value <= 0:
        return E24[0]
    decade = 10 ** int(np.floor(np.log10(value)))
    normalized = value / decade
    closest = min(E24, key=lambda x: abs(x - normalized))
    return closest * decade


def _apply_matching_chain(freqs_hz, z_ant, topology_name, values):
    """Apply a matching network chain and return matched impedance.

    Args:
        freqs_hz: frequency array in Hz
        z_ant: complex antenna impedance array
        topology_name: key into TOPOLOGY_DEFS
        values: list of float component values (nH for L, pF for C)

    Returns:
        z_matched: complex impedance array after matching
    """
    comp_defs = TOPOLOGY_DEFS[topology_name]
    if len(values) != len(comp_defs):
        return z_ant

    z = z_ant.copy()
    for (ref, comp_type, unit), val in zip(comp_defs, values):
        if val <= 0:
            continue
        if comp_type == 'L':
            # L topologies: series_L or shunt depends on position in name
            # Parse from topology name pattern: L = series, C = shunt
            z = apply_series(z, z_inductor(freqs_hz, val))
        elif comp_type == 'C':
            z = apply_shunt(z, z_capacitor(freqs_hz, val))

    return z


class S11Canvas(FigureCanvas):
    """Small matplotlib canvas for S11 comparison plot."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._init_plot()

    def _init_plot(self):
        """Set up empty axes with labels."""
        self.ax.set_xlabel('Frequency (MHz)', fontsize=9)
        self.ax.set_ylabel('S11 (dB)', fontsize=9)
        self.ax.set_xlim(500, 2500)
        self.ax.set_ylim(-35, 0)
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(-6, color='orange', linestyle='--', alpha=0.4, linewidth=0.8)
        self.ax.axhline(-10, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
        self.draw()

    def update_plot(self, freqs_hz, s11_bare_db, s11_matched_db,
                    lb_min=791e6, lb_max=960e6, hb_min=1710e6, hb_max=1990e6):
        """Redraw the S11 comparison plot.

        Args:
            freqs_hz: frequency array in Hz
            s11_bare_db: bare antenna S11 in dB
            s11_matched_db: matched S11 in dB (or None)
            lb_min, lb_max, hb_min, hb_max: band edges in Hz
        """
        self.ax.clear()

        freqs_mhz = freqs_hz / 1e6

        # Band shading
        self.ax.axvspan(lb_min / 1e6, lb_max / 1e6, alpha=0.08, color='blue')
        self.ax.axvspan(hb_min / 1e6, hb_max / 1e6, alpha=0.08, color='red')

        # Target lines
        self.ax.axhline(-6, color='orange', linestyle='--', alpha=0.4, linewidth=0.8)
        self.ax.axhline(-10, color='green', linestyle='--', alpha=0.4, linewidth=0.8)

        # Bare S11
        self.ax.plot(freqs_mhz, s11_bare_db, 'k-', linewidth=1.2,
                     alpha=0.5, label='Bare')

        # Matched S11
        if s11_matched_db is not None:
            self.ax.plot(freqs_mhz, s11_matched_db, 'b-', linewidth=1.5,
                         label='Matched')

        self.ax.set_xlabel('Frequency (MHz)', fontsize=9)
        self.ax.set_ylabel('S11 (dB)', fontsize=9)
        self.ax.set_xlim(500, 2500)
        self.ax.set_ylim(-35, 0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=8, loc='lower right')
        self.draw()


class MatchingPanel(QWidget):
    """Panel for matching network design and optimization.

    Provides topology selection, component value editing with live S11
    preview, E24 rounding, and an optimize button.
    """

    # Signals
    optimize_clicked = Signal(str)    # topology_name
    values_changed = Signal(list)     # list of component values

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._updating_table = False  # guard against re-entrant updates

        self._build_ui()
        self._connect_signals()

        # Initialize table for the default topology
        self._on_topology_changed(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Topology selector ---
        topo_group = QGroupBox('Topology')
        topo_layout = QFormLayout(topo_group)
        topo_layout.setContentsMargins(6, 10, 6, 6)
        self.topo_combo = QComboBox()
        for name in TOPOLOGY_NAMES:
            self.topo_combo.addItem(name)
        topo_layout.addRow('Network:', self.topo_combo)
        layout.addWidget(topo_group)

        # --- Component table ---
        table_group = QGroupBox('Components')
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(6, 10, 6, 6)

        self.comp_table = QTableWidget()
        self.comp_table.setColumnCount(4)
        self.comp_table.setHorizontalHeaderLabels(['Ref', 'Type', 'Value', 'Unit'])
        self.comp_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.comp_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.comp_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self.comp_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents)
        self.comp_table.verticalHeader().setVisible(False)
        self.comp_table.setMinimumHeight(120)
        table_layout.addWidget(self.comp_table)

        # Buttons row
        btn_row = QHBoxLayout()
        self.round_btn = QPushButton('Round to E24')
        self.optimize_btn = QPushButton('Optimize')
        self.optimize_btn.setMinimumHeight(32)
        btn_row.addWidget(self.round_btn)
        btn_row.addWidget(self.optimize_btn)
        table_layout.addLayout(btn_row)

        layout.addWidget(table_group)

        # --- Performance labels ---
        perf_group = QGroupBox('Performance')
        perf_layout = QFormLayout(perf_group)
        perf_layout.setContentsMargins(6, 10, 6, 6)
        self.lb_label = QLabel('--')
        self.hb_label = QLabel('--')
        perf_layout.addRow('LB worst S11:', self.lb_label)
        perf_layout.addRow('HB worst S11:', self.hb_label)
        layout.addWidget(perf_group)

        # --- S11 plot ---
        self.canvas = S11Canvas(self)
        layout.addWidget(self.canvas)

        layout.addStretch()

    def _connect_signals(self):
        self.topo_combo.currentIndexChanged.connect(self._on_topology_changed)
        self.comp_table.cellChanged.connect(self._on_cell_changed)
        self.round_btn.clicked.connect(self._on_round_e24)
        self.optimize_btn.clicked.connect(self._on_optimize_clicked)

        # React to new simulation results
        self.state.simulation_finished.connect(self._on_simulation_finished)
        self.state.matching_finished.connect(self._on_matching_finished)

    # ------------------------------------------------------------------
    # Topology / table management
    # ------------------------------------------------------------------

    def _on_topology_changed(self, index):
        """Rebuild the component table for the selected topology."""
        topo_name = TOPOLOGY_NAMES[index]
        comp_defs = TOPOLOGY_DEFS[topo_name]

        self._updating_table = True
        self.comp_table.setRowCount(len(comp_defs))

        for row, (ref, comp_type, unit) in enumerate(comp_defs):
            # Ref (read-only)
            ref_item = QTableWidgetItem(ref)
            ref_item.setFlags(ref_item.flags() & ~Qt.ItemIsEditable)
            self.comp_table.setItem(row, 0, ref_item)

            # Type (read-only)
            type_label = 'Inductor' if comp_type == 'L' else 'Capacitor'
            type_item = QTableWidgetItem(type_label)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.comp_table.setItem(row, 1, type_item)

            # Value (editable) -- default to a reasonable starting value
            default_val = 10.0 if comp_type == 'L' else 5.0
            val_item = QTableWidgetItem(f'{default_val:.2f}')
            self.comp_table.setItem(row, 2, val_item)

            # Unit (read-only)
            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.comp_table.setItem(row, 3, unit_item)

        self._updating_table = False

        # Store topology on state
        self.state.matching_topology = topo_name

        # Recompute with new topology
        self._recompute_matched_s11()

    def _get_table_values(self):
        """Read component values from the table. Returns list of floats."""
        values = []
        for row in range(self.comp_table.rowCount()):
            item = self.comp_table.item(row, 2)
            if item is None:
                values.append(0.0)
                continue
            try:
                values.append(float(item.text()))
            except ValueError:
                values.append(0.0)
        return values

    def _set_table_values(self, values):
        """Write component values into the table without triggering recompute."""
        self._updating_table = True
        for row, val in enumerate(values):
            if row < self.comp_table.rowCount():
                item = self.comp_table.item(row, 2)
                if item is not None:
                    item.setText(f'{val:.2f}')
        self._updating_table = False

    def set_values(self, values):
        """Public method to set component values and update the plot.

        Called by external code (e.g., after optimization completes).
        """
        self._set_table_values(values)
        self.state.matching_values = [
            (TOPOLOGY_DEFS[self.current_topology()][i][0], v)
            for i, v in enumerate(values)
        ]
        self._recompute_matched_s11()

    def current_topology(self):
        """Return the currently selected topology name."""
        return TOPOLOGY_NAMES[self.topo_combo.currentIndex()]

    def select_topology(self, name):
        """Programmatically select a topology by name."""
        if name in TOPOLOGY_NAMES:
            self.topo_combo.setCurrentIndex(TOPOLOGY_NAMES.index(name))

    # ------------------------------------------------------------------
    # Live recompute
    # ------------------------------------------------------------------

    def _on_cell_changed(self, row, col):
        """User edited a component value -- recompute matched S11."""
        if self._updating_table:
            return
        if col != 2:
            return
        self._recompute_matched_s11()
        values = self._get_table_values()
        self.values_changed.emit(values)

    def _recompute_matched_s11(self):
        """Recompute matched S11 from current table values and update plot.

        Uses z_complex from state.sim_result (no FDTD needed).
        """
        if self.state.sim_result is None:
            return

        freqs = self.state.sim_result['freqs']
        z_ant = self.state.sim_result['z_complex']
        s11_bare = s11_from_z(z_ant)

        topo_name = self.current_topology()
        values = self._get_table_values()

        # Validate all values are positive
        if any(v <= 0 for v in values):
            self.canvas.update_plot(
                freqs, s11_bare, None,
                lb_min=self.state.freq_bands['LB_MIN'],
                lb_max=self.state.freq_bands['LB_MAX'],
                hb_min=self.state.freq_bands['HB_MIN'],
                hb_max=self.state.freq_bands['HB_MAX'],
            )
            self.lb_label.setText('--')
            self.hb_label.setText('--')
            return

        try:
            z_matched = _apply_matching_chain(freqs, z_ant, topo_name, values)
            s11_matched = s11_from_z(z_matched)
        except Exception:
            self.canvas.update_plot(
                freqs, s11_bare, None,
                lb_min=self.state.freq_bands['LB_MIN'],
                lb_max=self.state.freq_bands['LB_MAX'],
                hb_min=self.state.freq_bands['HB_MIN'],
                hb_max=self.state.freq_bands['HB_MAX'],
            )
            self.lb_label.setText('Error')
            self.hb_label.setText('Error')
            return

        # Compute worst-case S11 per band
        lb_min = self.state.freq_bands['LB_MIN']
        lb_max = self.state.freq_bands['LB_MAX']
        hb_min = self.state.freq_bands['HB_MIN']
        hb_max = self.state.freq_bands['HB_MAX']

        lb_mask = (freqs >= lb_min) & (freqs <= lb_max)
        hb_mask = (freqs >= hb_min) & (freqs <= hb_max)

        lb_worst = np.max(s11_matched[lb_mask]) if np.any(lb_mask) else 0.0
        hb_worst = np.max(s11_matched[hb_mask]) if np.any(hb_mask) else 0.0

        self.lb_label.setText(f'{lb_worst:.1f} dB')
        self.hb_label.setText(f'{hb_worst:.1f} dB')

        # Update state
        self.state.matching_values = [
            (TOPOLOGY_DEFS[topo_name][i][0], v)
            for i, v in enumerate(values)
        ]
        self.state.matching_result = {
            's11_matched_db': s11_matched,
            'worst_lb': lb_worst,
            'worst_hb': hb_worst,
        }

        # Update plot
        self.canvas.update_plot(
            freqs, s11_bare, s11_matched,
            lb_min=lb_min, lb_max=lb_max,
            hb_min=hb_min, hb_max=hb_max,
        )

    # ------------------------------------------------------------------
    # E24 rounding
    # ------------------------------------------------------------------

    def _on_round_e24(self):
        """Round all component values to nearest E24 standard values."""
        values = self._get_table_values()
        rounded = [round_to_e24(v) if v > 0 else v for v in values]
        self._set_table_values(rounded)
        self._recompute_matched_s11()
        self.values_changed.emit(rounded)

    # ------------------------------------------------------------------
    # Optimize button
    # ------------------------------------------------------------------

    def _on_optimize_clicked(self):
        """Emit signal to start matching network optimization."""
        self.optimize_clicked.emit(self.current_topology())

    # ------------------------------------------------------------------
    # External update handlers
    # ------------------------------------------------------------------

    def _on_simulation_finished(self, result):
        """New simulation data available -- replot with current matching."""
        self._recompute_matched_s11()

    def _on_matching_finished(self, result):
        """Matching optimization completed -- update table and plot.

        Expected result dict keys:
            topology: str
            values: list of float
            lb_worst: float
            hb_worst: float
        """
        if 'topology' in result:
            self.select_topology(result['topology'])
        if 'values' in result:
            self.set_values(result['values'])
