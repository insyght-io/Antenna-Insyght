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

"""Impedance (R + jX) plotting panel for the antenna designer GUI."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ImpedanceView(QWidget):
    """Two-subplot panel: Re(Z) on top, Im(Z) on bottom."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state

        self._fig = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax_re = self._fig.add_subplot(2, 1, 1)
        self._ax_im = self._fig.add_subplot(2, 1, 2, sharex=self._ax_re)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._line_re = None
        self._line_im = None

        self._setup_axes()

        self._state.simulation_finished.connect(self._on_simulation_finished)
        self._state.freq_bands_changed.connect(self._redraw_bands)

    # ------------------------------------------------------------------
    # Static decorations
    # ------------------------------------------------------------------

    def _band_edges_mhz(self):
        fb = self._state.freq_bands
        return (
            fb['LB_MIN'] / 1e6, fb['LB_MAX'] / 1e6,
            fb['HB_MIN'] / 1e6, fb['HB_MAX'] / 1e6,
        )

    def _setup_axes(self):
        ax_r = self._ax_re
        ax_i = self._ax_im

        ax_r.set_ylabel('Resistance (\u03A9)')
        ax_r.set_title('Input Impedance')
        ax_r.grid(True, alpha=0.3)

        ax_i.set_xlabel('Frequency (MHz)')
        ax_i.set_ylabel('Reactance (\u03A9)')
        ax_i.grid(True, alpha=0.3)
        ax_i.axhline(0, color='black', lw=0.6, zorder=0)

        self._draw_band_shading()
        self._canvas.draw_idle()

    def _draw_band_shading(self):
        lb_min, lb_max, hb_min, hb_max = self._band_edges_mhz()
        for ax in (self._ax_re, self._ax_im):
            for attr in ('_lb_span', '_hb_span'):
                old = getattr(ax, attr, None)
                if old is not None:
                    old.remove()
            ax._lb_span = ax.axvspan(lb_min, lb_max,
                                     color='blue', alpha=0.15, zorder=0)
            ax._hb_span = ax.axvspan(hb_min, hb_max,
                                     color='red', alpha=0.15, zorder=0)

    def _redraw_bands(self):
        self._draw_band_shading()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------

    def _on_simulation_finished(self, result):
        freqs_hz = np.asarray(result['freqs'], dtype=float)
        z_complex = np.asarray(result['z_complex'], dtype=complex)

        freqs_mhz = freqs_hz / 1e6
        re_z = np.real(z_complex)
        im_z = np.imag(z_complex)

        # --- Resistance subplot ---
        if self._line_re is not None:
            self._line_re.remove()
        self._line_re, = self._ax_re.plot(
            freqs_mhz, re_z, color='#1f77b4', lw=1.5)

        # --- Reactance subplot ---
        if self._line_im is not None:
            self._line_im.remove()
        self._line_im, = self._ax_im.plot(
            freqs_mhz, im_z, color='#d62728', lw=1.5)

        for ax in (self._ax_re, self._ax_im):
            ax.relim()
            ax.autoscale_view()

        self._canvas.draw_idle()
