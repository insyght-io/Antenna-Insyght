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

"""
DXF export for KiCad import.

Exports the optimized antenna design as a DXF file with:
  - F.Cu layer: top copper antenna traces
  - B.Cu layer: bottom copper ground plane
  - Edge.Cuts layer: half-circle board outline
  - F.Fab layer: labels, dimensions, feed/short markers

Coordinates in mm, origin at center of the flat edge of the half-circle.
"""

import os
import json
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment

from config import (
    BOARD_RADIUS, GND_EXTENSION, SUBSTRATE_THICKNESS,
    MOUNTING_HOLES, COPPER_THICKNESS,
)
from antenna_model import DualBandIFA, BranchedIFA


def segment_to_polygon(x1, y1, x2, y2, width):
    """Convert a line segment (centerline) to a closed rectangle polygon.

    Returns 4 corner points offset ±width/2 perpendicular to the segment.
    For zero-length segments, returns a small square.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    hw = width / 2.0

    if length < 1e-9:
        # Degenerate segment — small square
        return [
            (x1 - hw, y1 - hw),
            (x1 + hw, y1 - hw),
            (x1 + hw, y1 + hw),
            (x1 - hw, y1 + hw),
        ]

    # Unit normal perpendicular to segment
    nx = -dy / length
    ny = dx / length

    return [
        (x1 + nx * hw, y1 + ny * hw),
        (x2 + nx * hw, y2 + ny * hw),
        (x2 - nx * hw, y2 - ny * hw),
        (x1 - nx * hw, y1 - ny * hw),
    ]


def _add_trace_polygons(msp, trace_segments, trace_width, layer):
    """Add closed polygon outlines for each segment in a polyline trace.

    For each consecutive pair of points in each trace segment, creates a
    closed rectangle polygon with the given width. This replaces
    LWPOLYLINE const_width which KiCad ignores on import.
    """
    for trace_seg in trace_segments:
        if len(trace_seg) < 2:
            continue
        points = [(p[0], p[1]) for p in trace_seg]
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            corners = segment_to_polygon(x1, y1, x2, y2, trace_width)
            msp.add_lwpolyline(corners, close=True,
                               dxfattribs={'layer': layer})


def create_dxf(antenna, output_path='export/antenna_traces.dxf'):
    """Create a DXF file with the antenna design for KiCad import.

    Args:
        antenna: DualBandIFA instance (geometry already generated)
        output_path: path for the DXF file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Create layers matching KiCad conventions
    doc.layers.add('F.Cu', color=1)       # Red - top copper (antenna traces)
    doc.layers.add('B.Cu', color=4)       # Cyan - bottom copper (ground plane)
    doc.layers.add('Edge.Cuts', color=6)  # Yellow - board outline
    doc.layers.add('F.Fab', color=3)      # Green - fabrication marks

    # =========================================================
    # Edge.Cuts — Board outline (half-circle + straight edge)
    # =========================================================
    r = BOARD_RADIUS

    # Half-circle arc: from (-r, 0) to (r, 0) going through (0, r)
    # ezdxf arc: center, radius, start_angle, end_angle (degrees, CCW)
    msp.add_arc(
        center=(0, 0), radius=r,
        start_angle=0, end_angle=180,
        dxfattribs={'layer': 'Edge.Cuts'}
    )
    # Straight edge (flat bottom)
    msp.add_line((-r, 0), (r, 0), dxfattribs={'layer': 'Edge.Cuts'})

    # =========================================================
    # F.Cu — Antenna traces (top copper, closed polygon outlines)
    # =========================================================
    _add_trace_polygons(msp, antenna.lowband.trace_2d,
                        antenna.lowband.trace_width, 'F.Cu')
    _add_trace_polygons(msp, antenna.highband.trace_2d,
                        antenna.highband.trace_width, 'F.Cu')

    # =========================================================
    # B.Cu — Ground plane (bottom copper)
    # =========================================================
    # The ground plane covers:
    # 1. The full area below the flat edge (the rest of the PCB)
    # 2. The bottom layer under the half-circle antenna area

    # Ground plane below flat edge (rectangular)
    gnd_rect = [
        (-r, 0),
        (r, 0),
        (r, -GND_EXTENSION),
        (-r, -GND_EXTENSION),
        (-r, 0),  # close the polygon
    ]
    msp.add_lwpolyline(gnd_rect, close=True, dxfattribs={'layer': 'B.Cu'})

    # Ground plane under half-circle (bottom layer pour)
    # This is a filled region matching the half-circle shape
    n_arc = 72
    arc_angles = np.linspace(0, np.pi, n_arc)
    arc_points = [(r * np.cos(a), r * np.sin(a)) for a in arc_angles]
    arc_points.append(arc_points[0])  # close
    msp.add_lwpolyline(arc_points, close=True, dxfattribs={'layer': 'B.Cu'})

    # =========================================================
    # F.Fab — Labels, markers, dimensions
    # =========================================================

    # Feed and short markers for low band
    lb_feed = antenna.lowband.feed_point
    lb_short = antenna.lowband.short_point
    _add_marker(msp, lb_feed[0], lb_feed[1], 'LB_FEED', 'F.Fab')
    _add_marker(msp, lb_short[0], lb_short[1], 'LB_SHORT', 'F.Fab')

    # Feed and short markers for high band
    hb_feed = antenna.highband.feed_point
    hb_short = antenna.highband.short_point
    _add_marker(msp, hb_feed[0], hb_feed[1], 'HB_FEED', 'F.Fab')
    _add_marker(msp, hb_short[0], hb_short[1], 'HB_SHORT', 'F.Fab')

    # Mounting holes
    for mx, my, mr in MOUNTING_HOLES:
        msp.add_circle(
            center=(mx, my), radius=mr,
            dxfattribs={'layer': 'F.Fab'}
        )
        msp.add_text(
            f'MTG {2*mr:.1f}mm',
            height=1.0,
            dxfattribs={'layer': 'F.Fab'}
        ).set_placement((mx + mr + 0.5, my), align=TextEntityAlignment.LEFT)

    # Title and dimensions
    msp.add_text(
        'Dual-Band IFA Antenna',
        height=1.5,
        dxfattribs={'layer': 'F.Fab'}
    ).set_placement((0, r + 3), align=TextEntityAlignment.CENTER)

    msp.add_text(
        f'Board: {2*r:.0f}mm x {r:.0f}mm half-circle',
        height=1.0,
        dxfattribs={'layer': 'F.Fab'}
    ).set_placement((0, r + 1), align=TextEntityAlignment.CENTER)

    # Antenna length annotations
    msp.add_text(
        f'LB: {antenna.lowband.actual_length:.1f}mm trace',
        height=0.8,
        dxfattribs={'layer': 'F.Fab'}
    ).set_placement((-r + 2, -3), align=TextEntityAlignment.LEFT)

    msp.add_text(
        f'HB: {antenna.highband.actual_length:.1f}mm trace',
        height=0.8,
        dxfattribs={'layer': 'F.Fab'}
    ).set_placement((5, -3), align=TextEntityAlignment.LEFT)

    doc.saveas(output_path)
    print(f"DXF saved to {output_path}")
    return output_path


def _add_marker(msp, x, y, label, layer):
    """Add a crosshair marker with label at (x, y)."""
    size = 1.0
    msp.add_line((x - size, y), (x + size, y), dxfattribs={'layer': layer})
    msp.add_line((x, y - size), (x, y + size), dxfattribs={'layer': layer})
    msp.add_circle(center=(x, y), radius=0.3, dxfattribs={'layer': layer})
    msp.add_text(
        label, height=0.7,
        dxfattribs={'layer': layer}
    ).set_placement((x + size + 0.3, y), align=TextEntityAlignment.LEFT)


def export_board_outline(output_path='export/board_outline.dxf'):
    """Export just the board outline as a separate DXF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.add('Edge.Cuts', color=6)

    r = BOARD_RADIUS
    msp.add_arc(
        center=(0, 0), radius=r,
        start_angle=0, end_angle=180,
        dxfattribs={'layer': 'Edge.Cuts'}
    )
    msp.add_line((-r, 0), (r, 0), dxfattribs={'layer': 'Edge.Cuts'})

    # Mounting holes
    for mx, my, mr in MOUNTING_HOLES:
        msp.add_circle(
            center=(mx, my), radius=mr,
            dxfattribs={'layer': 'Edge.Cuts'}
        )

    doc.saveas(output_path)
    print(f"Board outline DXF saved to {output_path}")
    return output_path


def create_dxf_branched(antenna, output_path='export/antenna_traces_branched.dxf'):
    """Create a DXF file for the branched IFA design.

    Args:
        antenna: BranchedIFA instance (geometry already generated)
        output_path: path for the DXF file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    doc.layers.add('F.Cu', color=1)
    doc.layers.add('B.Cu', color=4)
    doc.layers.add('Edge.Cuts', color=6)
    doc.layers.add('F.Fab', color=3)

    r = BOARD_RADIUS

    # Edge.Cuts — Board outline
    msp.add_arc(center=(0, 0), radius=r, start_angle=0, end_angle=180,
                dxfattribs={'layer': 'Edge.Cuts'})
    msp.add_line((-r, 0), (r, 0), dxfattribs={'layer': 'Edge.Cuts'})

    # F.Cu — All antenna traces (closed polygon outlines)
    _add_trace_polygons(msp, antenna.trace_2d, antenna.trace_width, 'F.Cu')

    # Capacitive load rectangle (already a closed polygon, no const_width needed)
    if antenna.cap_load_rect is not None:
        cx, cy, w, h = antenna.cap_load_rect
        cap_pts = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        msp.add_lwpolyline(cap_pts, close=True,
                            dxfattribs={'layer': 'F.Cu'})

    # B.Cu — Ground plane
    gnd_rect = [(-r, 0), (r, 0), (r, -GND_EXTENSION), (-r, -GND_EXTENSION), (-r, 0)]
    msp.add_lwpolyline(gnd_rect, close=True, dxfattribs={'layer': 'B.Cu'})

    n_arc = 72
    arc_angles = np.linspace(0, np.pi, n_arc)
    arc_points = [(r * np.cos(a), r * np.sin(a)) for a in arc_angles]
    arc_points.append(arc_points[0])
    msp.add_lwpolyline(arc_points, close=True, dxfattribs={'layer': 'B.Cu'})

    # F.Fab — Labels and markers
    feed = antenna.feed_point
    short = antenna.short_point
    _add_marker(msp, feed[0], feed[1], 'FEED', 'F.Fab')
    _add_marker(msp, short[0], short[1], 'SHORT', 'F.Fab')

    for mx, my, mr in MOUNTING_HOLES:
        msp.add_circle(center=(mx, my), radius=mr, dxfattribs={'layer': 'F.Fab'})
        msp.add_text(f'MTG {2*mr:.1f}mm', height=1.0,
                     dxfattribs={'layer': 'F.Fab'}
                     ).set_placement((mx + mr + 0.5, my), align=TextEntityAlignment.LEFT)

    msp.add_text('Branched IFA Antenna', height=1.5,
                 dxfattribs={'layer': 'F.Fab'}
                 ).set_placement((0, r + 3), align=TextEntityAlignment.CENTER)

    msp.add_text(f'LB: {antenna.actual_lb_length:.1f}mm, HB: {antenna.actual_hb_length:.1f}mm',
                 height=0.8, dxfattribs={'layer': 'F.Fab'}
                 ).set_placement((-r + 2, -3), align=TextEntityAlignment.LEFT)

    doc.saveas(output_path)
    print(f"DXF saved to {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse
    from config import ANTENNA_TOPOLOGY

    parser = argparse.ArgumentParser(description='Export antenna design as DXF')
    parser.add_argument('--params', default=None,
                        help='Path to best_params.json from optimization')
    parser.add_argument('--output', default='export/antenna_traces.dxf',
                        help='Output DXF file path')
    parser.add_argument('--topology', choices=['dual_ifa', 'branched'],
                        default=None, help='Antenna topology')
    args = parser.parse_args()

    topology = args.topology or ANTENNA_TOPOLOGY

    params = None
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        print(f"Loaded parameters from {args.params}")

    if topology == 'branched':
        antenna = BranchedIFA(params)
        antenna.generate_geometry()
        create_dxf_branched(antenna, args.output)
    else:
        antenna = DualBandIFA(params)
        geom = antenna.generate()
        create_dxf(antenna, args.output)

    export_board_outline('export/board_outline.dxf')

    print("\nDXF files ready for KiCad import.")
    print("Import instructions:")
    print("  1. Open KiCad PCB editor")
    print("  2. File -> Import -> Import Graphics")
    print("  3. Select the DXF file")
    print("  4. Set layer mapping: F.Cu -> F.Cu, Edge.Cuts -> Edge.Cuts")
    print("  5. Place at desired location")
