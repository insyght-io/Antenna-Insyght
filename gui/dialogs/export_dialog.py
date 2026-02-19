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

"""Dialog for selecting export formats and output directory."""

import os

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, QFormLayout,
                                QCheckBox, QHBoxLayout, QLineEdit,
                                QPushButton, QFileDialog, QDialogButtonBox)


# Export format options: (key, label, default_checked)
EXPORT_FORMATS = [
    ('dxf',           'DXF Geometry',   True),
    ('kicad_project', 'KiCad Project',  False),
    ('pdf_report',    'PDF Report',     True),
    ('npz_data',      'NPZ Data',       False),
]


class ExportDialog(QDialog):
    """Modal dialog to select export formats and output directory.

    After accept, call result_options() to get a dict:
        {
            'formats': {'dxf': True, 'kicad_project': False, ...},
            'output_dir': '/path/to/output',
        }
    """

    def __init__(self, default_output_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export')
        self.setMinimumWidth(400)

        if default_output_dir is None:
            default_output_dir = os.path.join(os.getcwd(), 'export')
        self._default_dir = default_output_dir

        self._checkboxes = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Format selection group
        fmt_group = QGroupBox('Formats')
        fmt_layout = QVBoxLayout(fmt_group)
        fmt_layout.setContentsMargins(8, 10, 8, 8)
        fmt_layout.setSpacing(4)

        for key, label, default_on in EXPORT_FORMATS:
            cb = QCheckBox(label)
            cb.setChecked(default_on)
            fmt_layout.addWidget(cb)
            self._checkboxes[key] = cb

        layout.addWidget(fmt_group)

        # Output directory
        dir_group = QGroupBox('Output Directory')
        dir_layout = QHBoxLayout(dir_group)
        dir_layout.setContentsMargins(8, 10, 8, 8)

        self._dir_edit = QLineEdit(self._default_dir)
        dir_layout.addWidget(self._dir_edit)

        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse)
        dir_layout.addWidget(browse_btn)

        layout.addWidget(dir_group)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        """Open a directory chooser and update the line edit."""
        path = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory', self._dir_edit.text())
        if path:
            self._dir_edit.setText(path)

    def result_options(self):
        """Return selected formats and output directory.

        Call after exec() returns QDialog.Accepted.
        """
        formats = {key: cb.isChecked() for key, cb in self._checkboxes.items()}
        return {
            'formats': formats,
            'output_dir': self._dir_edit.text(),
        }
