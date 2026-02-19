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

"""S11 and VSWR plotting panel for the antenna designer GUI."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class S11View(QWidget):
    """Two-subplot panel: S11 (dB) on top, VSWR on bottom."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state

        # Matplotlib figure and canvas
        self._fig = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax_s11 = self._fig.add_subplot(2, 1, 1)
        self._ax_vswr = self._fig.add_subplot(2, 1, 2, sharex=self._ax_s11)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        # Store line references for bare and matched traces
        self._line_s11_bare = None
        self._line_s11_matched = None
        self._line_vswr_bare = None
        self._line_vswr_matched = None

        # Cache simulation data for recomputing matched VSWR
        self._freqs_mhz = None
        self._s11_db = None
        self._s11_complex = None

        self._setup_axes()

        # Connect signals
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
        """Draw reference lines, band shading, and labels."""
        ax_s = self._ax_s11
        ax_v = self._ax_vswr

        # --- S11 subplot ---
        ax_s.set_ylabel('S11 (dB)')
        ax_s.set_title('Return Loss')
        ax_s.grid(True, alpha=0.3)
        ax_s.axhline(-6, color='orange', ls='--', lw=0.8, label='-6 dB')
        ax_s.axhline(-10, color='green', ls='--', lw=0.8, label='-10 dB')
        ax_s.legend(loc='upper right', fontsize=7, framealpha=0.7)

        # --- VSWR subplot ---
        ax_v.set_xlabel('Frequency (MHz)')
        ax_v.set_ylabel('VSWR')
        ax_v.set_title('VSWR')
        ax_v.grid(True, alpha=0.3)
        ax_v.set_ylim(1, 6)
        ax_v.axhline(3, color='orange', ls='--', lw=0.8, label='3:1')
        ax_v.axhline(2, color='green', ls='--', lw=0.8, label='2:1')
        ax_v.legend(loc='upper right', fontsize=7, framealpha=0.7)

        self._draw_band_shading()
        self._canvas.draw_idle()

    def _draw_band_shading(self):
        """Add vertical band shading to both subplots."""
        lb_min, lb_max, hb_min, hb_max = self._band_edges_mhz()
        for ax in (self._ax_s11, self._ax_vswr):
            # Remove old band spans if any
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

    @staticmethod
    def _s11_db_to_vswr(s11_db):
        """Convert S11 in dB to VSWR, clamped to avoid division by zero."""
        mag = np.power(10.0, s11_db / 20.0)
        mag = np.clip(mag, 0, 0.9999)
        return (1 + mag) / (1 - mag)

    def _on_simulation_finished(self, result):
        """Called when a new simulation result is ready."""
        freqs_hz = np.asarray(result['freqs'], dtype=float)
        s11_db = np.asarray(result['s11_db'], dtype=float)
        s11_complex = np.asarray(result.get('s11_complex', []), dtype=complex)

        self._freqs_mhz = freqs_hz / 1e6
        self._s11_db = s11_db
        self._s11_complex = s11_complex if len(s11_complex) else None

        vswr = self._s11_db_to_vswr(s11_db)

        # --- Update S11 subplot ---
        if self._line_s11_bare is not None:
            self._line_s11_bare.remove()
        self._line_s11_bare, = self._ax_s11.plot(
            self._freqs_mhz, s11_db, color='#1f77b4', lw=1.5, label='Bare')

        # --- Update VSWR subplot ---
        if self._line_vswr_bare is not None:
            self._line_vswr_bare.remove()
        self._line_vswr_bare, = self._ax_vswr.plot(
            self._freqs_mhz, vswr, color='#1f77b4', lw=1.5, label='Bare')

        # Clear any previous matched overlay
        self._clear_matched_lines()

        # Rescale and redraw
        self._ax_s11.relim()
        self._ax_s11.autoscale_view()
        self._ax_vswr.set_ylim(1, 6)
        self._update_legends()
        self._canvas.draw_idle()

    def set_matched_s11(self, freqs_hz, s11_db_matched):
        """Overlay matched S11 as a red dashed line on both subplots.

        Parameters
        ----------
        freqs_hz : array-like
            Frequency vector in Hz.
        s11_db_matched : array-like
            Matched S11 magnitude in dB.
        """
        freqs_mhz = np.asarray(freqs_hz, dtype=float) / 1e6
        s11_db_m = np.asarray(s11_db_matched, dtype=float)
        vswr_m = self._s11_db_to_vswr(s11_db_m)

        self._clear_matched_lines()

        self._line_s11_matched, = self._ax_s11.plot(
            freqs_mhz, s11_db_m, color='red', ls='--', lw=1.3, label='Matched')
        self._line_vswr_matched, = self._ax_vswr.plot(
            freqs_mhz, vswr_m, color='red', ls='--', lw=1.3, label='Matched')

        self._ax_s11.relim()
        self._ax_s11.autoscale_view()
        self._ax_vswr.set_ylim(1, 6)
        self._update_legends()
        self._canvas.draw_idle()

    def _clear_matched_lines(self):
        if self._line_s11_matched is not None:
            self._line_s11_matched.remove()
            self._line_s11_matched = None
        if self._line_vswr_matched is not None:
            self._line_vswr_matched.remove()
            self._line_vswr_matched = None

    def _update_legends(self):
        for ax in (self._ax_s11, self._ax_vswr):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
