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

"""Dialog for editing LB/HB frequency band targets."""

from PySide6.QtWidgets import (QDialog, QFormLayout, QDoubleSpinBox,
                                QDialogButtonBox)


class FreqBandsDialog(QDialog):
    """Modal dialog to edit the four frequency band edges (LB/HB min/max).

    Reads current values from DesignState.freq_bands (stored in Hz),
    displays in MHz, and writes back in Hz on accept.
    """

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle('Frequency Band Targets')
        self.setMinimumWidth(300)

        self._spinboxes = {}
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        fields = [
            ('LB_MIN', 'LB Min'),
            ('LB_MAX', 'LB Max'),
            ('HB_MIN', 'HB Min'),
            ('HB_MAX', 'HB Max'),
        ]

        for key, label in fields:
            spin = QDoubleSpinBox()
            spin.setRange(300.0, 3000.0)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setSuffix(' MHz')
            # State stores Hz, display as MHz
            current_hz = self.state.freq_bands.get(key, 0.0)
            spin.setValue(current_hz / 1e6)
            layout.addRow(label, spin)
            self._spinboxes[key] = spin

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self):
        """Write spinbox values back to state (MHz -> Hz) and close."""
        for key, spin in self._spinboxes.items():
            self.state.set_freq_band(key, spin.value() * 1e6)
        self.accept()
