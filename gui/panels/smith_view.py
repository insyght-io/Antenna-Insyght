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

"""Smith chart plotting panel for the antenna designer GUI."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle
import matplotlib.colors as mcolors


Z0 = 50.0  # Reference impedance (ohm)


class SmithView(QWidget):
    """Smith chart panel showing Z11 reflection coefficient."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state

        self._fig = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(1, 1, 1, aspect='equal')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._scatter = None
        self._colorbar = None

        self._draw_grid()
        self._state.simulation_finished.connect(self._on_simulation_finished)

    # ------------------------------------------------------------------
    # Smith chart grid
    # ------------------------------------------------------------------

    def _draw_grid(self):
        ax = self._ax
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(r'Re($\Gamma$)')
        ax.set_ylabel(r'Im($\Gamma$)')
        ax.set_title('Smith Chart')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        grid_color = '#cccccc'
        grid_lw = 0.5

        # Unit circle (boundary)
        unit = Circle((0, 0), 1, fill=False, ec='black', lw=1.0, zorder=1)
        ax.add_patch(unit)

        # Constant-resistance circles: r_n = R / Z0
        # Centre at (r_n/(1+r_n), 0), radius 1/(1+r_n)
        r_vals = [0, 0.2, 0.5, 1, 2, 5]
        for r in r_vals:
            cx = r / (1 + r)
            cr = 1 / (1 + r)
            circ = Circle((cx, 0), cr, fill=False, ec=grid_color, lw=grid_lw,
                          clip_on=True, zorder=0)
            ax.add_patch(circ)

        # Constant-reactance arcs: x_n = X / Z0
        # Centre at (1, 1/x_n), radius 1/|x_n|
        x_vals = [0.2, 0.5, 1, 2, 5]
        for x in x_vals:
            for sign in (+1, -1):
                cx = 1.0
                cy = sign / x
                cr = 1.0 / x
                # We need the arc that lies inside the unit circle.
                # Parametrise the circle and find the angular range inside
                # |gamma| <= 1.
                arc_pts = self._reactance_arc_points(cx, cy, cr)
                if arc_pts is not None:
                    ax.plot(arc_pts[:, 0], arc_pts[:, 1], color=grid_color,
                            lw=grid_lw, zorder=0)

        # Horizontal axis (real axis)
        ax.plot([-1, 1], [0, 0], color=grid_color, lw=grid_lw, zorder=0)

        self._canvas.draw_idle()

    @staticmethod
    def _reactance_arc_points(cx, cy, cr, n_pts=200):
        """Return points of a reactance arc clipped to the unit circle."""
        theta = np.linspace(0, 2 * np.pi, n_pts)
        pts_x = cx + cr * np.cos(theta)
        pts_y = cy + cr * np.sin(theta)
        r_sq = pts_x ** 2 + pts_y ** 2
        mask = r_sq <= 1.001  # small tolerance
        if not np.any(mask):
            return None
        # Keep only inside-unit-circle points
        return np.column_stack([pts_x[mask], pts_y[mask]])

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------

    def _on_simulation_finished(self, result):
        freqs_hz = np.asarray(result['freqs'], dtype=float)
        z_complex = np.asarray(result['z_complex'], dtype=complex)

        freqs_mhz = freqs_hz / 1e6

        # Reflection coefficient
        gamma = (z_complex - Z0) / (z_complex + Z0)
        gr = np.real(gamma)
        gi = np.imag(gamma)

        # Band edges for edge highlighting
        fb = self._state.freq_bands
        lb_min, lb_max = fb['LB_MIN'] / 1e6, fb['LB_MAX'] / 1e6
        hb_min, hb_max = fb['HB_MIN'] / 1e6, fb['HB_MAX'] / 1e6

        in_lb = (freqs_mhz >= lb_min) & (freqs_mhz <= lb_max)
        in_hb = (freqs_mhz >= hb_min) & (freqs_mhz <= hb_max)

        # Edge colors: blue for LB, red for HB, none for out-of-band
        edge_colors = np.full((len(freqs_mhz), 4), 0.0)  # RGBA, transparent
        edge_colors[in_lb] = mcolors.to_rgba('blue')
        edge_colors[in_hb] = mcolors.to_rgba('red')
        # Out-of-band points get a thin gray edge
        oob = ~(in_lb | in_hb)
        edge_colors[oob] = (0.6, 0.6, 0.6, 0.4)

        edge_widths = np.where(in_lb | in_hb, 1.2, 0.3)

        # Clear previous scatter and colorbar
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._scatter = self._ax.scatter(
            gr, gi,
            c=freqs_mhz,
            cmap='rainbow',
            s=18,
            edgecolors=edge_colors,
            linewidths=edge_widths,
            zorder=5,
        )
        self._colorbar = self._fig.colorbar(
            self._scatter, ax=self._ax, shrink=0.7, pad=0.05,
            label='Frequency (MHz)')

        self._canvas.draw_idle()
