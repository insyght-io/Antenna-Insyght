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

"""Dialog for FDTD simulation settings."""

from PySide6.QtWidgets import (QDialog, QFormLayout, QSpinBox, QComboBox,
                                QDialogButtonBox)


# Convergence threshold options: display label -> EndCriteria value
CONVERGENCE_OPTIONS = [
    ('-30 dB  (1e-3)', 1e-3),
    ('-40 dB  (1e-4)', 1e-4),
    ('-50 dB  (1e-5)', 1e-5),
]


class SimSettingsDialog(QDialog):
    """Modal dialog to configure FDTD simulation parameters.

    Returns a dict with keys: max_timesteps, end_criteria, mesh_res
    accessible via the settings() method after accept.
    """

    # Defaults matching typical openEMS usage
    DEFAULT_MAX_TIMESTEPS = 200000
    DEFAULT_CONVERGENCE_IDX = 1   # -40 dB
    DEFAULT_MESH_RES = 20

    def __init__(self, current_settings=None, parent=None):
        """
        Parameters
        ----------
        current_settings : dict, optional
            Keys: max_timesteps, end_criteria, mesh_res.
            If provided, dialog opens with these values.
        """
        super().__init__(parent)
        self.setWindowTitle('Simulation Settings')
        self.setMinimumWidth(320)

        self._current = current_settings or {}
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Max timesteps
        self._ts_spin = QSpinBox()
        self._ts_spin.setRange(10000, 500000)
        self._ts_spin.setSingleStep(10000)
        self._ts_spin.setValue(
            self._current.get('max_timesteps', self.DEFAULT_MAX_TIMESTEPS))
        layout.addRow('Max Timesteps', self._ts_spin)

        # Convergence threshold
        self._conv_combo = QComboBox()
        for label, _ in CONVERGENCE_OPTIONS:
            self._conv_combo.addItem(label)
        # Select based on current value or default
        current_ec = self._current.get('end_criteria')
        sel_idx = self.DEFAULT_CONVERGENCE_IDX
        if current_ec is not None:
            for i, (_, val) in enumerate(CONVERGENCE_OPTIONS):
                if val == current_ec:
                    sel_idx = i
                    break
        self._conv_combo.setCurrentIndex(sel_idx)
        layout.addRow('Convergence', self._conv_combo)

        # Mesh resolution (cells per wavelength)
        self._mesh_spin = QSpinBox()
        self._mesh_spin.setRange(10, 40)
        self._mesh_spin.setSingleStep(1)
        self._mesh_spin.setValue(
            self._current.get('mesh_res', self.DEFAULT_MESH_RES))
        self._mesh_spin.setSuffix(' cells/\u03bb')
        layout.addRow('Mesh Resolution', self._mesh_spin)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def settings(self):
        """Return the configured settings as a dict.

        Call after exec() returns QDialog.Accepted.
        """
        _, end_criteria = CONVERGENCE_OPTIONS[self._conv_combo.currentIndex()]
        return {
            'max_timesteps': self._ts_spin.value(),
            'end_criteria': end_criteria,
            'mesh_res': self._mesh_spin.value(),
        }
