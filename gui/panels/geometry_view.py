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

"""Geometry view panel: matplotlib canvas showing 2D top-down antenna layout."""

import sys
import os
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add project root to path so antenna_model imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antenna_model import BranchedIFA


class GeometryView(QWidget):
    """2D top-down view of the branched IFA antenna on its half-circle board."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        self._build_ui()

        # Redraw whenever parameters change
        self.state.params_changed.connect(self._redraw)

        # Initial draw
        self._redraw()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.canvas)

    def _redraw(self):
        """Regenerate antenna geometry from current state and redraw."""
        ax = self.ax
        ax.clear()

        radius = self.state.params.get('BOARD_RADIUS', 31.0)
        gnd_clearance = self.state.params.get('GND_CLEARANCE', 10.0)

        # --- Build antenna geometry ---
        try:
            antenna = BranchedIFA(self.state.antenna_params_dict())
            antenna.generate_geometry()
            lb_traces = antenna.lb_trace_2d
            hb_traces = antenna.hb_trace_2d
            feed_pt = antenna.feed_point
            short_pt = antenna.short_point
        except Exception:
            # If geometry generation fails, show empty board
            lb_traces = []
            hb_traces = []
            feed_pt = (0, 0)
            short_pt = (0, 0)

        # --- 1. Half-circle board outline ---
        theta = np.linspace(0, np.pi, 200)
        arc_x = radius * np.cos(theta)
        arc_y = radius * np.sin(theta)
        # Close with flat edge
        board_x = np.concatenate([[-radius], arc_x, [radius, -radius]])
        board_y = np.concatenate([[0], arc_y, [0, 0]])
        ax.plot(board_x, board_y, 'k-', linewidth=1.5, label='Board')

        # --- 2. Ground plane (filled gray rectangle below flat edge) ---
        gnd_depth = 15.0  # visual depth below flat edge
        gnd_rect_x = [-radius, radius, radius, -radius, -radius]
        gnd_rect_y = [0, 0, -gnd_depth, -gnd_depth, 0]
        ax.fill(gnd_rect_x, gnd_rect_y, color='#d0d0d0', alpha=0.5)
        ax.plot(gnd_rect_x, gnd_rect_y, 'k-', linewidth=1.0)

        # Ground clearance line (dashed)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

        # --- 3. LB arm traces (blue) ---
        for trace in lb_traces:
            if isinstance(trace, np.ndarray):
                xs = trace[:, 0]
                ys = trace[:, 1]
            else:
                xs = [p[0] for p in trace]
                ys = [p[1] for p in trace]
            ax.plot(xs, ys, 'b-', linewidth=2.0, solid_capstyle='round')
        # Add single label entry for legend
        if lb_traces:
            ax.plot([], [], 'b-', linewidth=2.0, label='LB arm')

        # --- 4. HB branch traces (red) ---
        for trace in hb_traces:
            if isinstance(trace, np.ndarray):
                xs = trace[:, 0]
                ys = trace[:, 1]
            else:
                xs = [p[0] for p in trace]
                ys = [p[1] for p in trace]
            ax.plot(xs, ys, 'r-', linewidth=2.0, solid_capstyle='round')
        if hb_traces:
            ax.plot([], [], 'r-', linewidth=2.0, label='HB branch')

        # --- 5. Feed point (green dot) ---
        ax.plot(feed_pt[0], feed_pt[1], 'o', color='green', markersize=8,
                zorder=5, label='Feed')

        # --- 6. Short point (orange dot) ---
        ax.plot(short_pt[0], short_pt[1], 's', color='orange', markersize=8,
                zorder=5, label='Short')

        # --- 7. Labels ---
        ax.annotate('Feed', xy=feed_pt, xytext=(feed_pt[0] + 2, feed_pt[1] - 3),
                     fontsize=8, color='green',
                     arrowprops=dict(arrowstyle='->', color='green', lw=0.8))
        ax.annotate('Short', xy=short_pt, xytext=(short_pt[0] - 8, short_pt[1] - 3),
                     fontsize=8, color='orange',
                     arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))

        # --- Axes formatting ---
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Antenna Geometry (top view)')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.3)

        margin = 5.0
        ax.set_xlim(-radius - margin, radius + margin)
        ax.set_ylim(-gnd_depth - margin, radius + margin)

        self.fig.tight_layout()
        self.canvas.draw_idle()
