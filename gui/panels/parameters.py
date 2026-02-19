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

"""Parameter panel: QDockWidget with grouped spinbox controls for antenna design."""

import os

from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QScrollArea,
                                QGroupBox, QFormLayout, QDoubleSpinBox, QPushButton,
                                QLabel)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap


# Parameter definitions: (key, label, min, max, step, decimals, suffix)
BOARD_PARAMS = [
    ('BOARD_RADIUS',        'Board Radius',         15.0,  50.0,  0.5, 1, ' mm'),
    ('SUBSTRATE_ER',        'Substrate Er',          2.0,  10.0,  0.1, 1, ''),
    ('SUBSTRATE_TAND',      'Loss Tangent',          0.001, 0.05, 0.001, 3, ''),
    ('SUBSTRATE_THICKNESS', 'Substrate Thickness',   0.4,   3.2,  0.1, 1, ' mm'),
]

ANTENNA_PARAMS = [
    ('SHORT_X',       'Short X',          -30.0,  10.0, 0.5, 1, ' mm'),
    ('FEED_OFFSET',   'Feed Offset',        2.0,  15.0, 0.5, 1, ' mm'),
    ('ELEM_HEIGHT',   'Element Height',     3.0,  25.0, 0.5, 1, ' mm'),
    ('HB_LENGTH',     'HB Length',         10.0,  60.0, 0.5, 1, ' mm'),
    ('HB_ANGLE',      'HB Angle',           2.0,  60.0, 0.5, 1, ' mm'),
    ('LB_LENGTH',     'LB Length',         50.0, 300.0, 5.0, 0, ' mm'),
    ('LB_SPACING',    'LB Spacing',         3.0,  20.0, 0.5, 1, ' mm'),
    ('TRACE_WIDTH',   'Trace Width',        0.5,   3.0, 0.1, 1, ' mm'),
    ('GND_CLEARANCE', 'Ground Clearance',   3.0,  20.0, 0.5, 1, ' mm'),
]

# Frequency params: (key, label, min_mhz, max_mhz)
FREQ_PARAMS = [
    ('LB_MIN', 'LB Min',  500.0, 1200.0),
    ('LB_MAX', 'LB Max',  500.0, 1200.0),
    ('HB_MIN', 'HB Min', 1200.0, 3000.0),
    ('HB_MAX', 'HB Max', 1200.0, 3000.0),
]


class ParameterPanel(QDockWidget):
    """Dock panel with grouped spinbox controls for all antenna parameters."""

    simulate_clicked = Signal()

    def __init__(self, state, parent=None):
        super().__init__('Parameters', parent)
        self.state = state
        self._spinboxes = {}   # key -> QDoubleSpinBox (antenna/board params)
        self._freq_spinboxes = {}  # key -> QDoubleSpinBox (frequency params)

        self._build_ui()

        # Connect state signals for external updates
        self.state.params_changed.connect(self._on_state_params_changed)
        self.state.freq_bands_changed.connect(self._on_state_freq_changed)

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Board group
        layout.addWidget(self._make_param_group('Board', BOARD_PARAMS))

        # Antenna group
        layout.addWidget(self._make_param_group('Antenna', ANTENNA_PARAMS))

        # Frequency group
        layout.addWidget(self._make_freq_group())

        # Simulate button
        sim_btn = QPushButton('Simulate')
        sim_btn.setMinimumHeight(36)
        sim_btn.clicked.connect(self.simulate_clicked.emit)
        layout.addWidget(sim_btn)

        layout.addStretch()

        # Logo
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'insyght-logo.png')
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaledToWidth(180, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setContentsMargins(0, 8, 0, 4)
            layout.addWidget(logo_label)

        scroll.setWidget(container)
        self.setWidget(scroll)

    def _make_param_group(self, title, param_defs):
        """Create a QGroupBox with QFormLayout of QDoubleSpinBoxes."""
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setContentsMargins(6, 10, 6, 6)
        form.setSpacing(4)

        for key, label, vmin, vmax, step, decimals, suffix in param_defs:
            spin = QDoubleSpinBox()
            spin.setRange(vmin, vmax)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            if suffix:
                spin.setSuffix(suffix)
            spin.setValue(self.state.params.get(key, vmin))
            spin.valueChanged.connect(lambda val, k=key: self._on_spin_changed(k, val))
            form.addRow(label, spin)
            self._spinboxes[key] = spin

        return group

    def _make_freq_group(self):
        """Create frequency group with MHz display / Hz storage."""
        group = QGroupBox('Frequency')
        form = QFormLayout(group)
        form.setContentsMargins(6, 10, 6, 6)
        form.setSpacing(4)

        for key, label, vmin_mhz, vmax_mhz in FREQ_PARAMS:
            spin = QDoubleSpinBox()
            spin.setRange(vmin_mhz, vmax_mhz)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setSuffix(' MHz')
            # State stores Hz, display MHz
            current_hz = self.state.freq_bands.get(key, vmin_mhz * 1e6)
            spin.setValue(current_hz / 1e6)
            spin.valueChanged.connect(
                lambda val, k=key: self._on_freq_spin_changed(k, val))
            form.addRow(label, spin)
            self._freq_spinboxes[key] = spin

        return group

    def _on_spin_changed(self, key, value):
        """User changed a parameter spinbox."""
        self.state.set_param(key, value)

    def _on_freq_spin_changed(self, key, value_mhz):
        """User changed a frequency spinbox (MHz -> Hz for state)."""
        self.state.set_freq_band(key, value_mhz * 1e6)

    def _on_state_params_changed(self):
        """State was updated externally; refresh spinbox values without re-emitting."""
        for key, spin in self._spinboxes.items():
            val = self.state.params.get(key)
            if val is not None and spin.value() != val:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)

    def _on_state_freq_changed(self):
        """Frequency bands updated externally; refresh spinboxes."""
        for key, spin in self._freq_spinboxes.items():
            val_hz = self.state.freq_bands.get(key)
            if val_hz is not None:
                val_mhz = val_hz / 1e6
                if spin.value() != val_mhz:
                    spin.blockSignals(True)
                    spin.setValue(val_mhz)
                    spin.blockSignals(False)
