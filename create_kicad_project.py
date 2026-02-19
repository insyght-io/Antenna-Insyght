#!/usr/bin/env python3

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
Generate a complete KiCad 9 project for the branched IFA antenna test board.

Creates:
  - kicad_project/antenna_test.kicad_pro  (project config)
  - kicad_project/antenna_test.kicad_sch  (schematic)
  - kicad_project/antenna_test.kicad_pcb  (PCB layout)

The PCB outline comes from the product DXF (PCB big_V4.dxf).
Antenna traces are generated from the BranchedIFA model in antenna_model.py.
Components include U.FL connector, matching network (8-element LCLCLCLC, 0603),
solder jumper for bypass, and mounting holes.

Usage:
  cd antenna-design
  python create_kicad_project.py
"""

import os
import sys
import math
import json
import uuid as uuid_mod
import numpy as np

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BOARD_RADIUS, BRANCHED_SHORT_X, BRANCHED_FEED_OFFSET,
    BRANCHED_ELEM_HEIGHT, BRANCHED_TRACE_WIDTH, SUBSTRATE_THICKNESS,
)
from antenna_model import BranchedIFA

# ============================================================
# Constants
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_DXF = os.path.join(SCRIPT_DIR, '..', 'PCB big_V4.dxf')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'kicad_project')

# Net definitions: (id, name)
# Simplified: no SMA connector, U.FL connects directly to RF_IN
# 8-element LCLCLCLC matching: 4 series inductors, 4 shunt caps, 3 internal nodes
NETS = [
    (0, ''),
    (1, 'GND'),
    (2, 'RF_IN'),
    (3, 'N1'),
    (4, 'N2'),
    (5, 'N3'),
    (6, 'ANT'),
]

# Matching network values: 8-element LCLCLCLC for R=31 groot board (Design B)
# E24-optimized, 0603 package
# Physical order from RF_IN to ANT: L1,C1,L2,C2,L3,C3,L4,C4
# Each: (ref, type, value, pad1_net_id, pad2_net_id, lcsc, mpn, manufacturer)
MATCHING_COMPONENTS = [
    ('L1', 'L', '3.0nH',  2, 3, 'TBD', 'TBD', 'TBD'),   # RF_IN → N1
    ('C1', 'C', '1.8pF',  3, 1, 'TBD', 'TBD', 'TBD'),   # N1 → GND
    ('L2', 'L', '7.5nH',  3, 4, 'TBD', 'TBD', 'TBD'),   # N1 → N2
    ('C2', 'C', '3.0pF',  4, 1, 'TBD', 'TBD', 'TBD'),   # N2 → GND
    ('L3', 'L', '8.2nH',  4, 5, 'TBD', 'TBD', 'TBD'),   # N2 → N3
    ('C3', 'C', '2.7pF',  5, 1, 'TBD', 'TBD', 'TBD'),   # N3 → GND
    ('L4', 'L', '6.8nH',  5, 6, 'TBD', 'TBD', 'TBD'),   # N3 → ANT
    ('C4', 'C', '1.8pF',  6, 1, 'TBD', 'TBD', 'TBD'),   # ANT → GND
]

# Connector part data (SMA removed, U.FL only)
CONNECTOR_PARTS = {
    'J1': {'lcsc': 'C88374', 'mpn': 'U.FL-R-SMT-1(80)', 'manufacturer': 'HRS (Hirose)'},
}

TRACE_WIDTH_ANT = BRANCHED_TRACE_WIDTH   # 1.5 mm antenna traces
TRACE_WIDTH_RF  = 0.5                     # 0.5 mm RF signal traces
TRACE_WIDTH_GND = 0.5                     # 0.5 mm GND traces
VIA_SIZE = 0.8
VIA_DRILL = 0.4


def new_uuid():
    """Generate a new UUID string."""
    return str(uuid_mod.uuid4())


# ============================================================
# Coordinate Transformation
# ============================================================

class CoordTransform:
    """Transform from antenna space to KiCad PCB space.

    Antenna space: origin at center of flat edge, Y up into semicircle.
    KiCad space: Y down. No rotation — user will position manually.
    """

    def __init__(self):
        # No rotation or scaling — straight orientation
        # Place antenna origin at (50, 50) in KiCad space
        self.tx = 50.0
        self.ty = 50.0

        # Antenna center in KiCad coords
        self.center_kicad = self.antenna_to_kicad(0, 0)

    def antenna_to_kicad(self, x_ant, y_ant):
        """Transform point from antenna space to KiCad space."""
        return (x_ant + self.tx, -y_ant + self.ty)


# ============================================================
# DXF Outline Parser
# ============================================================

def parse_groot_outline():
    """Parse the groot DXF and return board outline as gr_line segments.

    Returns list of ((x1,y1), (x2,y2)) in KiCad coordinates (Y negated).
    """
    import ezdxf

    doc = ezdxf.readfile(BOARD_DXF)
    msp = doc.modelspace()

    segments = []

    for entity in msp:
        etype = entity.dxftype()

        if etype == 'LINE':
            s = entity.dxf.start
            e = entity.dxf.end
            segments.append(((s.x, -s.y), (e.x, -e.y)))

        elif etype == 'SPLINE':
            try:
                points = list(entity.flattening(0.1))
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    segments.append(((p1[0], -p1[1]), (p2[0], -p2[1])))
            except Exception as ex:
                print(f"  Warning: Could not flatten spline: {ex}")

        elif etype == 'ARC':
            c = entity.dxf.center
            r = entity.dxf.radius
            sa = math.radians(entity.dxf.start_angle)
            ea = math.radians(entity.dxf.end_angle)
            if ea < sa:
                ea += 2 * math.pi
            n_pts = max(int((ea - sa) / (math.pi / 36)), 8)
            angles = np.linspace(sa, ea, n_pts + 1)
            for i in range(len(angles) - 1):
                x1 = c.x + r * math.cos(angles[i])
                y1 = -(c.y + r * math.sin(angles[i]))
                x2 = c.x + r * math.cos(angles[i + 1])
                y2 = -(c.y + r * math.sin(angles[i + 1]))
                segments.append(((x1, y1), (x2, y2)))
        # Skip CIRCLE entities (mounting holes handled separately)

    return segments


def parse_groot_mounting_holes():
    """Extract mounting hole positions from groot DXF.

    Returns list of (x, y, radius) in KiCad coordinates.
    """
    import ezdxf

    doc = ezdxf.readfile(BOARD_DXF)
    msp = doc.modelspace()

    holes = []
    for entity in msp:
        if entity.dxftype() == 'CIRCLE':
            c = entity.dxf.center
            r = entity.dxf.radius
            holes.append((c.x, -c.y, r))
    return holes


# ============================================================
# Antenna Geometry
# ============================================================

def generate_antenna_traces(transform):
    """Generate antenna trace segments in KiCad coordinates.

    Returns:
        traces: list of ((x1,y1), (x2,y2)) segments
        feed_kicad: (x, y) feed point in KiCad coords
        short_kicad: (x, y) short point in KiCad coords
    """
    antenna = BranchedIFA()
    segments, feed_pt, short_pt = antenna.generate_geometry()

    traces = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        kx1, ky1 = transform.antenna_to_kicad(x1, y1)
        kx2, ky2 = transform.antenna_to_kicad(x2, y2)
        traces.append(((kx1, ky1), (kx2, ky2)))

    feed_kicad = transform.antenna_to_kicad(feed_pt[0], feed_pt[1])
    short_kicad = transform.antenna_to_kicad(short_pt[0], short_pt[1])

    print(f"  Antenna: {len(traces)} trace segments")
    print(f"  Feed point (KiCad): ({feed_kicad[0]:.2f}, {feed_kicad[1]:.2f})")
    print(f"  Short point (KiCad): ({short_kicad[0]:.2f}, {short_kicad[1]:.2f})")

    return traces, feed_kicad, short_kicad


# ============================================================
# Via Stitching
# ============================================================

def generate_via_stitching(transform, pitch=4.0):
    """Generate via positions along the semicircle flat edge.

    Returns list of (x, y) in KiCad coordinates.
    """
    # Flat edge in antenna space: from (-30, 0) to (30, 0)
    # Offset into ground plane (y = -2mm in antenna space) for edge clearance
    edge_len = 2 * BOARD_RADIUS
    n_vias = int(edge_len / pitch)
    vias = []
    for i in range(n_vias + 1):
        x_ant = -BOARD_RADIUS + 2.0 + i * pitch  # inset 2mm from edges
        if x_ant > BOARD_RADIUS - 2.0:
            break
        y_ant = -2.0  # 2mm below flat edge (in ground plane area)
        vx, vy = transform.antenna_to_kicad(x_ant, y_ant)
        vias.append((vx, vy))

    return vias


# ============================================================
# Footprint Generators
# ============================================================

def _fp_header(lib_name, ref, value, x, y, rot, attrs='smd'):
    """Generate footprint header (KiCad 9 order: layer, uuid, at)."""
    uid = new_uuid()
    lines = [
        f'  (footprint "{lib_name}"',
        f'    (layer "F.Cu")',
        f'    (uuid "{uid}")',
        f'    (at {x:.4f} {y:.4f} {rot:.1f})',
        f'    (property "Reference" "{ref}"',
        f'      (at 0 -2 0)',
        f'      (layer "F.SilkS")',
        f'      (uuid "{new_uuid()}")',
        f'      (effects (font (size 0.8 0.8) (thickness 0.12)))',
        f'    )',
        f'    (property "Value" "{value}"',
        f'      (at 0 2 0)',
        f'      (layer "F.Fab")',
        f'      (uuid "{new_uuid()}")',
        f'      (effects (font (size 0.8 0.8) (thickness 0.12)))',
        f'    )',
    ]
    if attrs:
        lines.append(f'    (attr {attrs})')
    return '\n'.join(lines)


def _pad_smd_roundrect(num, x, y, w, h, net_id, net_name, layers='"F.Cu" "F.Mask" "F.Paste"', rratio=0.25, rot=0):
    rot_str = f' {rot:.0f}' if rot else ''
    return (
        f'    (pad "{num}" smd roundrect\n'
        f'      (at {x:.4f} {y:.4f}{rot_str})\n'
        f'      (size {w:.3f} {h:.3f})\n'
        f'      (layers {layers})\n'
        f'      (roundrect_rratio {rratio})\n'
        f'      (net {net_id} "{net_name}")\n'
        f'      (uuid "{new_uuid()}")\n'
        f'    )'
    )


def _pad_smd_rect(num, x, y, w, h, net_id, net_name, layers='"F.Cu" "F.Mask" "F.Paste"', rot=0):
    rot_str = f' {rot:.0f}' if rot else ''
    return (
        f'    (pad "{num}" smd rect\n'
        f'      (at {x:.4f} {y:.4f}{rot_str})\n'
        f'      (size {w:.3f} {h:.3f})\n'
        f'      (layers {layers})\n'
        f'      (net {net_id} "{net_name}")\n'
        f'      (uuid "{new_uuid()}")\n'
        f'    )'
    )


def _pad_thru_circle(num, x, y, pad_size, drill, net_id, net_name):
    return (
        f'    (pad "{num}" thru_hole circle\n'
        f'      (at {x:.4f} {y:.4f})\n'
        f'      (size {pad_size:.3f} {pad_size:.3f})\n'
        f'      (drill {drill:.3f})\n'
        f'      (layers "*.Cu" "*.Mask")\n'
        f'      (net {net_id} "{net_name}")\n'
        f'      (uuid "{new_uuid()}")\n'
        f'    )'
    )


def _fp_rect(x1, y1, x2, y2, layer, width=0.1):
    return (
        f'    (fp_rect\n'
        f'      (start {x1:.3f} {y1:.3f})\n'
        f'      (end {x2:.3f} {y2:.3f})\n'
        f'      (stroke (width {width}) (type solid))\n'
        f'      (fill no)\n'
        f'      (layer "{layer}")\n'
        f'      (uuid "{new_uuid()}")\n'
        f'    )'
    )


def fp_0603_inductor(ref, value, x, y, rot, pad1_net, pad2_net):
    """Generate 0603 inductor footprint (1608 Metric)."""
    net_name = {n[0]: n[1] for n in NETS}
    parts = [
        _fp_header('Inductor_SMD:L_0603_1608Metric', ref, value, x, y, rot),
        _fp_rect(-0.8, -0.4, 0.8, 0.4, 'F.Fab'),
        _fp_rect(-1.48, -0.73, 1.48, 0.73, 'F.CrtYd', 0.05),
        _pad_smd_roundrect('1', -0.775, 0, 0.9, 0.95, pad1_net, net_name[pad1_net]),
        _pad_smd_roundrect('2', 0.775, 0, 0.9, 0.95, pad2_net, net_name[pad2_net]),
        '  )',
    ]
    return '\n'.join(parts)


def fp_0603_capacitor(ref, value, x, y, rot, pad1_net, pad2_net):
    """Generate 0603 capacitor footprint (1608 Metric)."""
    net_name = {n[0]: n[1] for n in NETS}
    parts = [
        _fp_header('Capacitor_SMD:C_0603_1608Metric', ref, value, x, y, rot),
        _fp_rect(-0.8, -0.4, 0.8, 0.4, 'F.Fab'),
        _fp_rect(-1.48, -0.73, 1.48, 0.73, 'F.CrtYd', 0.05),
        _pad_smd_roundrect('1', -0.775, 0, 0.9, 0.95, pad1_net, net_name[pad1_net]),
        _pad_smd_roundrect('2', 0.775, 0, 0.9, 0.95, pad2_net, net_name[pad2_net]),
        '  )',
    ]
    return '\n'.join(parts)


def fp_sma_edge_mount(ref, x, y, rot, sig_net, gnd_net):
    """Generate SMA edge-mount connector footprint (Amphenol 132289)."""
    net_name = {n[0]: n[1] for n in NETS}
    sn, gn = net_name[sig_net], net_name[gnd_net]
    parts = [
        _fp_header('Connector_Coaxial:SMA_Amphenol_132289_EdgeMount', ref, 'SMA', x, y, rot),
        # Fab outline (connector body)
        f'    (fp_line (start -1.91 -5.08) (end 4.445 -5.08) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start -1.91 -5.08) (end -1.91 -3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start -1.91 -3.81) (end 2.54 -3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 2.54 -3.81) (end 2.54 3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 2.54 3.81) (end -1.91 3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start -1.91 3.81) (end -1.91 5.08) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start -1.91 5.08) (end 4.445 5.08) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 4.445 -5.08) (end 4.445 -3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 4.445 -3.81) (end 13.97 -3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 13.97 -3.81) (end 13.97 3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 13.97 3.81) (end 4.445 3.81) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 4.445 3.81) (end 4.445 5.08) (stroke (width 0.1) (type solid)) (layer "F.Fab") (uuid "{new_uuid()}"))',
        _fp_rect(-3.04, -5.58, 14.47, 5.58, 'F.CrtYd', 0.05),
        # Pads: signal center, GND at ±4.25
        _pad_smd_rect('1', 0, 0, 1.5, 5.08, sig_net, sn, rot=90),
        _pad_smd_rect('2', 0, -4.25, 1.5, 5.08, gnd_net, gn, layers='"F.Cu" "F.Mask" "F.Paste"', rot=90),
        _pad_smd_rect('2', 0, -4.25, 1.5, 5.08, gnd_net, gn, layers='"B.Cu" "B.Mask" "B.Paste"', rot=90),
        _pad_smd_rect('2', 0, 4.25, 1.5, 5.08, gnd_net, gn, layers='"F.Cu" "F.Mask" "F.Paste"', rot=90),
        _pad_smd_rect('2', 0, 4.25, 1.5, 5.08, gnd_net, gn, layers='"B.Cu" "B.Mask" "B.Paste"', rot=90),
        '  )',
    ]
    return '\n'.join(parts)


def fp_ufl(ref, x, y, rot, sig_net, gnd_net):
    """Generate U.FL connector footprint (Hirose U.FL-R-SMT-1)."""
    net_name = {n[0]: n[1] for n in NETS}
    sn, gn = net_name[sig_net], net_name[gnd_net]
    parts = [
        _fp_header('Connector_Coaxial:U.FL_Hirose_U.FL-R-SMT-1_Vertical', ref, 'U.FL', x, y, rot),
        # Simplified fab outline
        _fp_rect(-0.425, -1.5, 1.375, 1.5, 'F.Fab'),
        _fp_rect(-2.02, -2.5, 2.28, 2.5, 'F.CrtYd', 0.05),
        # Pads
        _pad_smd_roundrect('1', -1.05, 0, 1.05, 1.0, sig_net, sn),
        _pad_smd_roundrect('2', 0.475, -1.475, 2.2, 1.05, gnd_net, gn, rratio=0.238),
        _pad_smd_roundrect('2', 0.475, 1.475, 2.2, 1.05, gnd_net, gn, rratio=0.238),
        '  )',
    ]
    return '\n'.join(parts)


def fp_solder_jumper_3(ref, x, y, rot, pad1_net, pad2_net, pad3_net):
    """Generate 3-pad solder jumper footprint."""
    net_name = {n[0]: n[1] for n in NETS}
    parts = [
        _fp_header('Jumper:SolderJumper-3_P1.3mm_Open_RoundedPad1.0x1.5mm_NumberLabels',
                    ref, 'JP_SMA_UFL', x, y, rot,
                    attrs='exclude_from_pos_files exclude_from_bom allow_soldermask_bridges'),
        _fp_rect(-2.3, -1.25, 2.3, 1.25, 'F.CrtYd', 0.05),
        # Silkscreen outline
        f'    (fp_line (start -1.4 -1) (end 1.4 -1) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 1.4 1) (end -1.4 1) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{new_uuid()}"))',
        # Number labels
        f'    (fp_text user "1" (at -2.6 0 0) (layer "F.SilkS") (uuid "{new_uuid()}") (effects (font (size 0.8 0.8) (thickness 0.12))))',
        f'    (fp_text user "3" (at 2.6 0 0) (layer "F.SilkS") (uuid "{new_uuid()}") (effects (font (size 0.8 0.8) (thickness 0.12))))',
        # Pads (simplified to rect instead of custom rounded)
        _pad_smd_rect('1', -1.3, 0, 1.0, 1.5, pad1_net, net_name[pad1_net]),
        _pad_smd_rect('2', 0, 0, 1.0, 1.5, pad2_net, net_name[pad2_net]),
        _pad_smd_rect('3', 1.3, 0, 1.0, 1.5, pad3_net, net_name[pad3_net]),
        '  )',
    ]
    return '\n'.join(parts)


def fp_solder_jumper_2(ref, x, y, rot, pad1_net, pad2_net):
    """Generate 2-pad solder jumper footprint (bypass)."""
    net_name = {n[0]: n[1] for n in NETS}
    parts = [
        _fp_header('Jumper:SolderJumper-2_P1.3mm_Open_RoundedPad1.0x1.5mm',
                    ref, 'Bypass', x, y, rot,
                    attrs='exclude_from_pos_files exclude_from_bom allow_soldermask_bridges'),
        _fp_rect(-1.65, -1.25, 1.65, 1.25, 'F.CrtYd', 0.05),
        f'    (fp_line (start -0.7 -1) (end 0.7 -1) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{new_uuid()}"))',
        f'    (fp_line (start 0.7 1) (end -0.7 1) (stroke (width 0.12) (type solid)) (layer "F.SilkS") (uuid "{new_uuid()}"))',
        _pad_smd_rect('1', -0.65, 0, 1.0, 1.5, pad1_net, net_name[pad1_net]),
        _pad_smd_rect('2', 0.65, 0, 1.0, 1.5, pad2_net, net_name[pad2_net]),
        '  )',
    ]
    return '\n'.join(parts)


def fp_mounting_hole(ref, x, y, gnd_net):
    """Generate mounting hole footprint (M2.5, grounded pad)."""
    net_name = {n[0]: n[1] for n in NETS}
    parts = [
        _fp_header('MountingHole:MountingHole_2.7mm_M2.5_DIN965_Pad',
                    ref, 'MountingHole', x, y, 0.0, attrs='exclude_from_pos_files exclude_from_bom'),
        f'    (fp_circle (center 0 0) (end 2.45 0) (stroke (width 0.15) (type solid)) (fill no) (layer "Dwgs.User") (uuid "{new_uuid()}"))',
        _pad_thru_circle('1', 0, 0, 5.0, 2.7, gnd_net, net_name[gnd_net]),
        '  )',
    ]
    return '\n'.join(parts)


# ============================================================
# KiCad Project File (.kicad_pro)
# ============================================================

def generate_kicad_pro():
    """Generate minimal KiCad 9 project file."""
    return json.dumps({
        "board": {
            "3dviewports": [],
            "design_settings": {
                "defaults": {
                    "apply_defaults_to_fp_fields": False,
                    "apply_defaults_to_fp_shapes": False,
                    "apply_defaults_to_fp_text": False,
                    "board_outline_line_width": 0.05,
                    "copper_line_width": 0.2,
                    "copper_text_italic": False,
                    "copper_text_size_h": 1.5,
                    "copper_text_size_v": 1.5,
                    "copper_text_thickness": 0.3,
                    "copper_text_upright": False,
                    "other_line_width": 0.1,
                    "silk_line_width": 0.1,
                    "silk_text_italic": False,
                    "silk_text_size_h": 1.0,
                    "silk_text_size_v": 1.0,
                    "silk_text_thickness": 0.1,
                    "silk_text_upright": False,
                    "zones": { "min_clearance": 0.3 }
                },
                "diff_pair_dimensions": [],
                "drc_exclusions": [],
                "rules": {
                    "max_error": 0.005,
                    "min_clearance": 0.2,
                    "min_connection": 0.0,
                    "min_copper_edge_clearance": 0.0,
                    "min_hole_clearance": 0.25,
                    "min_hole_to_hole": 0.25,
                    "min_microvia_diameter": 0.2,
                    "min_microvia_drill": 0.1,
                    "min_resolved_spokes": 2,
                    "min_silk_clearance": 0.0,
                    "min_text_height": 0.8,
                    "min_text_thickness": 0.08,
                    "min_through_hole_diameter": 0.3,
                    "min_track_width": 0.1,
                    "min_via_annular_width": 0.1,
                    "min_via_diameter": 0.5,
                    "solder_mask_to_copper_clearance": 0.0,
                    "use_height_for_length_calcs": True
                },
                "teardrop_options": [
                    { "td_onpadsmd": True, "td_onroundshapesonly": False,
                      "td_ontrackend": False, "td_onviapad": True }
                ],
                "teardrop_parameters": [
                    { "td_allow_use_two_tracks": True, "td_curve_segcount": 0,
                      "td_height_ratio": 1.0, "td_length_ratio": 0.5,
                      "td_maxheight": 2.0, "td_maxlen": 1.0,
                      "td_on_pad_in_zone": False, "td_target_name": "td_round_shape",
                      "td_width_to_size_filter_ratio": 0.9 },
                    { "td_allow_use_two_tracks": True, "td_curve_segcount": 0,
                      "td_height_ratio": 1.0, "td_length_ratio": 0.5,
                      "td_maxheight": 2.0, "td_maxlen": 1.0,
                      "td_on_pad_in_zone": False, "td_target_name": "td_rect_shape",
                      "td_width_to_size_filter_ratio": 0.9 },
                    { "td_allow_use_two_tracks": True, "td_curve_segcount": 0,
                      "td_height_ratio": 1.0, "td_length_ratio": 0.5,
                      "td_maxheight": 2.0, "td_maxlen": 1.0,
                      "td_on_pad_in_zone": False, "td_target_name": "td_track_end",
                      "td_width_to_size_filter_ratio": 0.9 }
                ],
                "track_widths": [0.0, 0.2, 0.5, 1.0, 1.5],
                "via_dimensions": [{ "diameter": 0.0, "drill": 0.0 }],
                "zones_allow_external_fillets": False
            },
            "ipc2581": { "dist": "", "mfg": "", "assembly_variant_default": "" },
            "layer_presets": [],
            "viewports": []
        },
        "boards": [],
        "cvpcb": { "equivalence_files": [] },
        "libraries": {
            "pinned_footprint_libs": [],
            "pinned_symbol_libs": []
        },
        "meta": {
            "filename": "antenna_test.kicad_pro",
            "version": 2
        },
        "net_settings": {
            "classes": [
                {
                    "bus_width": 12,
                    "clearance": 0.2,
                    "diff_pair_gap": 0.25,
                    "diff_pair_via_gap": 0.25,
                    "diff_pair_width": 0.2,
                    "line_style": 0,
                    "microvia_diameter": 0.3,
                    "microvia_drill": 0.1,
                    "name": "Default",
                    "pcb_color": "rgba(0, 0, 0, 0.000)",
                    "schematic_color": "rgba(0, 0, 0, 0.000)",
                    "track_width": 0.2,
                    "via_diameter": 0.6,
                    "via_drill": 0.3,
                    "wire_width": 6
                }
            ],
            "meta": { "version": 3 },
            "net_colors": None,
            "netclass_assignments": None,
            "netclass_patterns": []
        },
        "pcbnew": {
            "last_paths": { "gencad": "", "idf": "", "netlist": "", "plot": "", "pos_files": "", "specctra_dsn": "", "step": "", "vrml": "" },
            "page_layout_descr_file": ""
        },
        "schematic": {
            "legacy_lib_dir": "",
            "legacy_lib_list": []
        },
        "sheets": [],
        "text_variables": {}
    }, indent=2)


# ============================================================
# KiCad Schematic File (.kicad_sch)
# ============================================================

def _sym_lib_capacitor():
    """Embedded library symbol for capacitor."""
    return f"""    (symbol "Device:C"
      (pin_numbers hide)
      (pin_names (offset 0.254) hide)
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "C" (at 0.635 2.54 0) (effects (font (size 1.27 1.27)) (justify left)))
      (property "Value" "C" (at 0.635 -2.54 0) (effects (font (size 1.27 1.27)) (justify left)))
      (property "Footprint" "" (at 0.9652 -3.81 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "Unpolarized capacitor" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "C_0_1"
        (polyline (pts (xy -2.032 -0.762) (xy 2.032 -0.762)) (stroke (width 0.508) (type default)) (fill (type none)))
        (polyline (pts (xy -2.032 0.762) (xy 2.032 0.762)) (stroke (width 0.508) (type default)) (fill (type none)))
      )
      (symbol "C_1_1"
        (pin passive line (at 0 3.81 270) (length 2.794) (name "~" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 0 -3.81 90) (length 2.794) (name "~" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_inductor():
    """Embedded library symbol for inductor."""
    return f"""    (symbol "Device:L"
      (pin_numbers hide)
      (pin_names (offset 1.016) hide)
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "L" (at -1.016 0 90) (effects (font (size 1.27 1.27))))
      (property "Value" "L" (at 1.524 0 90) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "Inductor" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "L_0_1"
        (arc (start 0 -2.54) (mid 0.6323 -1.905) (end 0 -1.27) (stroke (width 0) (type default)) (fill (type none)))
        (arc (start 0 -1.27) (mid 0.6323 -0.635) (end 0 0) (stroke (width 0) (type default)) (fill (type none)))
        (arc (start 0 0) (mid 0.6323 0.635) (end 0 1.27) (stroke (width 0) (type default)) (fill (type none)))
        (arc (start 0 1.27) (mid 0.6323 1.905) (end 0 2.54) (stroke (width 0) (type default)) (fill (type none)))
      )
      (symbol "L_1_1"
        (pin passive line (at 0 3.81 270) (length 1.27) (name "~" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 0 -3.81 90) (length 1.27) (name "~" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_conn_coaxial():
    """Embedded library symbol for coaxial connector."""
    return f"""    (symbol "Connector:Conn_Coaxial"
      (pin_names (offset 1.016) hide)
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "J" (at 0.254 3.048 0) (effects (font (size 1.27 1.27))))
      (property "Value" "Conn_Coaxial" (at 2.286 -2.286 0) (effects (font (size 1.27 1.27)) (justify left)))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "Coaxial connector" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "Conn_Coaxial_0_1"
        (arc (start -1.778 -0.508) (mid 0.2311 -1.8066) (end 1.778 0) (stroke (width 0.254) (type default)) (fill (type none)))
        (arc (start 1.778 0) (mid 0.2311 1.8066) (end -1.778 0.508) (stroke (width 0.254) (type default)) (fill (type none)))
      )
      (symbol "Conn_Coaxial_1_1"
        (pin passive line (at -3.81 0 0) (length 2.032) (name "In" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 0 -3.81 90) (length 2.54) (name "Ext" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_solder_jumper_3():
    """Embedded library symbol for 3-pad solder jumper."""
    return f"""    (symbol "Jumper:SolderJumper_3_Open"
      (pin_names (offset 0) hide)
      (exclude_from_sim yes)
      (in_bom no)
      (on_board yes)
      (property "Reference" "JP" (at -2.54 -2.54 0) (effects (font (size 1.27 1.27)) (justify left)))
      (property "Value" "SolderJumper_3_Open" (at 0 2.794 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "3-pad solder jumper, open" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "SolderJumper_3_Open_0_1"
        (polyline (pts (xy -2.54 0) (xy -0.635 0)) (stroke (width 0) (type default)) (fill (type none)))
        (polyline (pts (xy 0.635 0) (xy 2.54 0)) (stroke (width 0) (type default)) (fill (type none)))
        (polyline (pts (xy 0 0.508) (xy 0 -0.508)) (stroke (width 0.508) (type default)) (fill (type none)))
        (rectangle (start -0.508 0.762) (end 0.508 -0.762) (stroke (width 0) (type default)) (fill (type outline)))
      )
      (symbol "SolderJumper_3_Open_1_1"
        (pin passive line (at -5.08 0 0) (length 2.54) (name "A" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin input line (at 0 -3.81 90) (length 3.302) (name "C" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 5.08 0 180) (length 2.54) (name "B" (effects (font (size 1.27 1.27)))) (number "3" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_solder_jumper_2():
    """Embedded library symbol for 2-pad solder jumper."""
    return f"""    (symbol "Jumper:SolderJumper_2_Open"
      (pin_names (offset 0) hide)
      (exclude_from_sim yes)
      (in_bom no)
      (on_board yes)
      (property "Reference" "JP" (at -2.54 -2.54 0) (effects (font (size 1.27 1.27)) (justify left)))
      (property "Value" "SolderJumper_2_Open" (at 0 2.794 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "2-pad solder jumper, open" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "SolderJumper_2_Open_0_1"
        (polyline (pts (xy -2.54 0) (xy -0.635 0)) (stroke (width 0) (type default)) (fill (type none)))
        (polyline (pts (xy 0.635 0) (xy 2.54 0)) (stroke (width 0) (type default)) (fill (type none)))
      )
      (symbol "SolderJumper_2_Open_1_1"
        (pin passive line (at -5.08 0 0) (length 2.54) (name "A" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 5.08 0 180) (length 2.54) (name "B" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_gnd():
    """Embedded library symbol for GND power symbol."""
    return f"""    (symbol "power:GND"
      (power)
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "#PWR" (at 0 -6.35 0) (effects (font (size 1.27 1.27)) hide))
      (property "Value" "GND" (at 0 -3.81 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "Power symbol, ground" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "GND_0_1"
        (polyline (pts (xy 0 0) (xy 0 -1.27) (xy 1.27 -1.27) (xy 0 -2.54) (xy -1.27 -1.27) (xy 0 -1.27)) (stroke (width 0) (type default)) (fill (type none)))
      )
      (symbol "GND_1_1"
        (pin power_in line (at 0 0 270) (length 0) (name "~" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sym_lib_pwr_flag():
    """Embedded library symbol for PWR_FLAG."""
    return f"""    (symbol "power:PWR_FLAG"
      (power)
      (pin_numbers hide)
      (pin_names (offset 0) hide)
      (exclude_from_sim no)
      (in_bom yes)
      (on_board yes)
      (property "Reference" "#FLG" (at 0 1.905 0) (effects (font (size 1.27 1.27)) hide))
      (property "Value" "PWR_FLAG" (at 0 3.81 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Description" "Special symbol for power flag" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "PWR_FLAG_0_1"
        (pin power_out line (at 0 0 90) (length 0) (name "pwr" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      )
    )"""


def _sch_symbol(lib_id, ref, value, x, y, rot_code=0, mirror=False, unit=1,
                footprint='', extra_props=None, pin_uuids=None):
    """Generate a placed schematic symbol.

    rot_code: 0=0°, 1=90°, 2=180°, 3=270°
    """
    uid = new_uuid()
    # Build transform matrix based on rotation
    transforms = {
        0: '(1 0 0 1)',     # 0°
        1: '(0 1 -1 0)',    # 90°
        2: '(-1 0 0 -1)',   # 180°
        3: '(0 -1 1 0)',    # 270°
    }
    mx = '-1 0 0 1' if mirror else '1 0 0 1'
    if rot_code == 0:
        mx = '-1 0 0 1' if mirror else '1 0 0 1'
    elif rot_code == 1:
        mx = '0 -1 -1 0' if mirror else '0 1 -1 0'
    elif rot_code == 2:
        mx = '1 0 0 -1' if mirror else '-1 0 0 -1'
    elif rot_code == 3:
        mx = '0 1 1 0' if mirror else '0 -1 1 0'

    lines = [
        f'  (symbol',
        f'    (lib_id "{lib_id}")',
        f'    (at {x:.2f} {y:.2f} {rot_code * 90})',
        f'    (unit {unit})',
        f'    (exclude_from_sim no)',
        f'    (in_bom {"no" if "Jumper" in lib_id or "power" in lib_id else "yes"})',
        f'    (on_board yes)',
        f'    (dnp no)',
        f'    (uuid "{uid}")',
        f'    (property "Reference" "{ref}" (at 0 -3.81 0) (effects (font (size 1.27 1.27))))',
        f'    (property "Value" "{value}" (at 0 3.81 0) (effects (font (size 1.27 1.27))))',
    ]
    if footprint:
        lines.append(f'    (property "Footprint" "{footprint}" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    if extra_props:
        for k, v in extra_props.items():
            lines.append(f'    (property "{k}" "{v}" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')

    # Pin UUIDs
    if pin_uuids:
        for pin_num, pin_uuid in pin_uuids.items():
            lines.append(f'    (pin "{pin_num}" (uuid "{pin_uuid}"))')
    else:
        lines.append(f'    (pin "1" (uuid "{new_uuid()}"))')
        lines.append(f'    (pin "2" (uuid "{new_uuid()}"))')

    lines.append(f'    (instances')
    lines.append(f'      (project "antenna_test"')
    lines.append(f'        (path "/{new_uuid()}" (reference "{ref}") (unit {unit}))')
    lines.append(f'      )')
    lines.append(f'    )')
    lines.append(f'  )')
    return '\n'.join(lines)


def _sch_wire(x1, y1, x2, y2):
    return f'  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f})) (uuid "{new_uuid()}"))'


def _sch_label(text, x, y, angle=0):
    return (
        f'  (label "{text}" (at {x:.2f} {y:.2f} {angle})\n'
        f'    (effects (font (size 1.27 1.27)))\n'
        f'    (uuid "{new_uuid()}")\n'
        f'  )'
    )


def generate_kicad_sch():
    """Generate KiCad 9 schematic file content.

    Simplified: U.FL (J1) connects directly to RF_IN.
    JP1 bypass jumper shorts RF_IN to ANT (bypasses matching network).
    """
    sch_uuid = new_uuid()

    # U.FL connector (left side)
    j1_x, j1_y = 25.4, 63.5       # U.FL

    # Matching network (middle) — horizontal chain
    mn_x_start = 63.5
    mn_spacing = 20.32  # 8 grid units between inductor centers
    mn_y = 63.5

    # JP1 bypass (above matching network)
    jp1_x, jp1_y = 101.6, 50.8

    parts = []
    parts.append(f'(kicad_sch')
    parts.append(f'  (version 20231120)')
    parts.append(f'  (generator "antenna_test_generator")')
    parts.append(f'  (generator_version "2.0")')
    parts.append(f'  (uuid "{sch_uuid}")')
    parts.append(f'  (paper "A3")')

    # Title block
    parts.append(f'  (title_block')
    parts.append(f'    (title "Branched IFA Antenna Test Board")')
    parts.append(f'    (date "2026-02-19")')
    parts.append(f'    (rev "2.0")')
    parts.append(f'    (comment 1 "8-element LCLCLCLC matching network, 0603")')
    parts.append(f'    (comment 2 "R=31 groot board, U.FL only")')
    parts.append(f'  )')

    # Embedded library symbols
    parts.append(f'  (lib_symbols')
    parts.append(_sym_lib_capacitor())
    parts.append(_sym_lib_inductor())
    parts.append(_sym_lib_conn_coaxial())
    parts.append(_sym_lib_solder_jumper_2())
    parts.append(_sym_lib_gnd())
    parts.append(_sym_lib_pwr_flag())
    parts.append(f'  )')

    # --- Place symbols ---
    # J1 U.FL (was J2, now the only connector)
    j1_props = CONNECTOR_PARTS['J1']
    parts.append(_sch_symbol('Connector:Conn_Coaxial', 'J1', 'U.FL', j1_x, j1_y,
                             footprint='Connector_Coaxial:U.FL_Hirose_U.FL-R-SMT-1_Vertical',
                             extra_props={'LCSC': j1_props['lcsc'], 'MPN': j1_props['mpn'],
                                          'Manufacturer': j1_props['manufacturer']}))

    # Matching network L1-L4, C1-C4 (0603 footprints)
    # Inductors: rotated 90° (horizontal), Capacitors: rotation 0° (vertical shunt)
    for i, (ref, ctype, value, net1, net2, lcsc, mpn, mfr) in enumerate(MATCHING_COMPONENTS):
        comp_props = {'LCSC': lcsc, 'MPN': mpn, 'Manufacturer': mfr}
        if ctype == 'L':
            col = i // 2
            sx = mn_x_start + col * mn_spacing
            parts.append(_sch_symbol('Device:L', ref, value, sx, mn_y, rot_code=1,
                                     footprint='Inductor_SMD:L_0603_1608Metric',
                                     extra_props=comp_props))
        else:
            col = i // 2
            cx = mn_x_start + col * mn_spacing + 3.81
            parts.append(_sch_symbol('Device:C', ref, value, cx, mn_y + 15.24, rot_code=0,
                                     footprint='Capacitor_SMD:C_0603_1608Metric',
                                     extra_props=comp_props))

    # JP1 bypass jumper (shorts RF_IN to ANT, bypassing matching network)
    parts.append(_sch_symbol('Jumper:SolderJumper_2_Open', 'JP1', 'Bypass',
                             jp1_x, jp1_y,
                             footprint='Jumper:SolderJumper-2_P1.3mm_Open_RoundedPad1.0x1.5mm'))

    # GND power symbols for each shunt cap
    for i in range(4):
        gx = mn_x_start + i * mn_spacing + 3.81
        gy = mn_y + 15.24 + 7.62
        parts.append(_sch_symbol('power:GND', f'#PWR0{i+1:02d}', 'GND', gx, gy,
                                 pin_uuids={'1': new_uuid()}))

    # J1 GND symbol
    parts.append(_sch_symbol('power:GND', '#PWR005', 'GND', j1_x, j1_y + 7.62,
                             pin_uuids={'1': new_uuid()}))

    # PWR_FLAG on GND net
    parts.append(_sch_symbol('power:PWR_FLAG', '#FLG01', 'PWR_FLAG',
                             j1_x + 7.62, j1_y + 7.62,
                             pin_uuids={'1': new_uuid()}))

    # Net labels
    parts.append(_sch_label('RF_IN', j1_x + 5.08, j1_y, 0))
    parts.append(_sch_label('ANT', mn_x_start + 3 * mn_spacing + 7.62, mn_y, 0))

    for i, name in enumerate(['N1', 'N2', 'N3']):
        lx = mn_x_start + i * mn_spacing + mn_spacing / 2
        parts.append(_sch_label(name, lx, mn_y, 0))

    # =========================================================
    # Wires — complete connectivity
    # =========================================================
    cap_y = mn_y + 15.24

    # --- J1 U.FL wiring ---
    # J1 signal (pin1, left) → RF_IN → L1.pin1 (left)
    parts.append(_sch_wire(j1_x - 3.81, j1_y, mn_x_start - 3.81, mn_y))

    # J1 GND (pin2, bottom) → GND symbol
    parts.append(_sch_wire(j1_x, j1_y + 3.81, j1_x, j1_y + 7.62))

    # PWR_FLAG → GND wire
    parts.append(_sch_wire(j1_x + 7.62, j1_y + 7.62, j1_x, j1_y + 7.62))

    # --- Series inductor chain ---
    for col in range(3):  # 3 connections between 4 inductors
        l_cur_x = mn_x_start + col * mn_spacing
        l_next_x = mn_x_start + (col + 1) * mn_spacing
        l_pin2_x = l_cur_x + 3.81
        l_pin1_x = l_next_x - 3.81
        parts.append(_sch_wire(l_pin2_x, mn_y, l_pin1_x, mn_y))

    # --- Shunt capacitors to junctions ---
    for col in range(4):  # 4 shunt caps
        cx = mn_x_start + col * mn_spacing + 3.81
        parts.append(_sch_wire(cx, mn_y, cx, cap_y - 3.81))
        parts.append(_sch_wire(cx, cap_y + 3.81, cx, cap_y + 7.62))

    # --- L4.pin2 → ANT label ---
    l4_pin2_x = mn_x_start + 3 * mn_spacing + 3.81
    ant_label_x = mn_x_start + 3 * mn_spacing + 7.62
    parts.append(_sch_wire(l4_pin2_x, mn_y, ant_label_x, mn_y))

    # --- JP1 bypass: RF_IN to ANT ---
    # JP1 pin1 (left) connects to RF_IN net, pin2 (right) to ANT net
    # Horizontal wire from RF_IN area to JP1.pin1
    rfin_x = j1_x + 5.08  # RF_IN label position
    parts.append(_sch_wire(rfin_x, j1_y, rfin_x, jp1_y))
    parts.append(_sch_wire(rfin_x, jp1_y, jp1_x - 5.08, jp1_y))
    # JP1.pin2 to ANT
    parts.append(_sch_wire(jp1_x + 5.08, jp1_y, ant_label_x, jp1_y))
    parts.append(_sch_wire(ant_label_x, jp1_y, ant_label_x, mn_y))

    # Note about the circuit
    parts.append(f'  (text "8-element LCLCLCLC matching network (0603)\\n'
                 f'R=31 groot board, U.FL input only\\n'
                 f'Solder JP1 to bypass matching network"')
    parts.append(f'    (exclude_from_sim no)')
    parts.append(f'    (at 25.4 25.4 0)')
    parts.append(f'    (effects (font (size 1.27 1.27)) (justify left))')
    parts.append(f'    (uuid "{new_uuid()}")')
    parts.append(f'  )')

    parts.append(f')')
    return '\n'.join(parts)


# ============================================================
# KiCad PCB File (.kicad_pcb)
# ============================================================

def generate_kicad_pcb(transform):
    """Generate the complete KiCad 9 PCB file."""

    print("Generating PCB...")

    # --- Gather geometry ---
    print("  Parsing board outline from DXF...")
    outline_segs = parse_groot_outline()
    print(f"  Board outline: {len(outline_segs)} segments")

    print("  Generating antenna traces...")
    ant_traces, feed_kicad, short_kicad = generate_antenna_traces(transform)

    print("  Generating via stitching...")
    via_positions = generate_via_stitching(transform, pitch=4.0)
    print(f"  Via stitching: {len(via_positions)} vias")

    # --- Component positions (KiCad coordinates) ---
    # Board bounds (approx): X: 21-79, Y: 21-125
    # Flat edge (antenna/body boundary): Y ≈ 50
    # Components go in body area (Y > 55)
    # Layout: J1 U.FL near matching input, matching chain horizontal, JP1 bypass

    # J1 - U.FL connector, body area below matching network
    j1_x, j1_y, j1_rot = 35.0, 80.0, 0.0

    # Matching network: horizontal chain near flat edge (8-element, 4L + 4C)
    mn_y_l = 64.0       # Y for series inductors
    mn_y_c = 70.0       # Y for shunt capacitors (6mm below inductors)
    mn_x0 = 37.5        # Starting X for L1
    mn_dx = 5.5          # Spacing between L centers

    # L positions (4 series inductors, going right)
    l_positions = [(mn_x0 + i * mn_dx, mn_y_l) for i in range(4)]
    # C positions (4 shunt caps, at node junctions: C1-C3 between inductors, C4 after L4)
    c_positions = [
        (mn_x0 + i * mn_dx + mn_dx / 2, mn_y_c) for i in range(4)
    ]

    # JP1 (bypass) - left of matching network
    jp1_x, jp1_y, jp1_rot = 35.5, 60.5, 0.0

    # --- Build PCB content ---
    pcb = []

    # Header
    pcb.append('(kicad_pcb')
    pcb.append('  (version 20241229)')
    pcb.append('  (generator "pcbnew")')
    pcb.append('  (generator_version "9.0")')
    pcb.append(f'  (general')
    pcb.append(f'    (thickness {SUBSTRATE_THICKNESS})')
    pcb.append(f'    (legacy_teardrops no)')
    pcb.append(f'  )')
    pcb.append(f'  (paper "A4")')

    # Layers (2-layer board, KiCad 9 numbering)
    pcb.append('  (layers')
    pcb.append('    (0 "F.Cu" signal)')
    pcb.append('    (2 "B.Cu" signal)')
    pcb.append('    (9 "F.Adhes" user "F.Adhesive")')
    pcb.append('    (11 "B.Adhes" user "B.Adhesive")')
    pcb.append('    (13 "F.Paste" user)')
    pcb.append('    (15 "B.Paste" user)')
    pcb.append('    (5 "F.SilkS" user "F.Silkscreen")')
    pcb.append('    (7 "B.SilkS" user "B.Silkscreen")')
    pcb.append('    (1 "F.Mask" user)')
    pcb.append('    (3 "B.Mask" user)')
    pcb.append('    (17 "Dwgs.User" user "User.Drawings")')
    pcb.append('    (19 "Cmts.User" user "User.Comments")')
    pcb.append('    (21 "Eco1.User" user "User.Eco1")')
    pcb.append('    (23 "Eco2.User" user "User.Eco2")')
    pcb.append('    (25 "Edge.Cuts" user)')
    pcb.append('    (27 "Margin" user)')
    pcb.append('    (31 "F.CrtYd" user "F.Courtyard")')
    pcb.append('    (29 "B.CrtYd" user "B.Courtyard")')
    pcb.append('    (35 "F.Fab" user)')
    pcb.append('    (33 "B.Fab" user)')
    pcb.append('  )')

    # Setup with stackup (required for KiCad 9)
    pcb.append('  (setup')
    pcb.append('    (stackup')
    pcb.append('      (layer "F.SilkS" (type "Top Silk Screen"))')
    pcb.append('      (layer "F.Paste" (type "Top Solder Paste"))')
    pcb.append('      (layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))')
    pcb.append('      (layer "F.Cu" (type "copper") (thickness 0.035))')
    pcb.append('      (layer "dielectric 1" (type "core") (thickness 1.51) (material "FR4") (epsilon_r 4.5) (loss_tangent 0.02))')
    pcb.append('      (layer "B.Cu" (type "copper") (thickness 0.035))')
    pcb.append('      (layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.01))')
    pcb.append('      (layer "B.Paste" (type "Bottom Solder Paste"))')
    pcb.append('      (layer "B.SilkS" (type "Bottom Silk Screen"))')
    pcb.append('      (copper_finish "None")')
    pcb.append('      (dielectric_constraints no)')
    pcb.append('    )')
    pcb.append('    (pad_to_mask_clearance 0)')
    pcb.append('    (allow_soldermask_bridges_in_footprints yes)')
    pcb.append('    (tenting front back)')
    pcb.append('    (pcbplotparams')
    pcb.append('      (layerselection 0x00000000_00000000_00000000_000000a5)')
    pcb.append('      (plot_on_all_layers_selection 0x00000000_00000000_00000000_00000000)')
    pcb.append('      (disableapertmacros no)')
    pcb.append('      (usegerberextensions no)')
    pcb.append('      (usegerberattributes yes)')
    pcb.append('      (usegerberadvancedattributes yes)')
    pcb.append('      (creategerberjobfile yes)')
    pcb.append('      (dashed_line_dash_ratio 12.000000)')
    pcb.append('      (dashed_line_gap_ratio 3.000000)')
    pcb.append('      (svgprecision 6)')
    pcb.append('      (plotframeref no)')
    pcb.append('      (mode 1)')
    pcb.append('      (useauxorigin no)')
    pcb.append('      (hpglpennumber 1)')
    pcb.append('      (hpglpenspeed 20)')
    pcb.append('      (hpglpendiameter 15.000000)')
    pcb.append('      (pdf_front_fp_property_popups yes)')
    pcb.append('      (pdf_back_fp_property_popups yes)')
    pcb.append('      (pdf_metadata yes)')
    pcb.append('      (pdf_single_document no)')
    pcb.append('      (dxfpolygonmode yes)')
    pcb.append('      (dxfimperialunits yes)')
    pcb.append('      (dxfusepcbnewfont yes)')
    pcb.append('      (psnegative no)')
    pcb.append('      (psa4output no)')
    pcb.append('      (plot_black_and_white yes)')
    pcb.append('      (plotinvisibletext no)')
    pcb.append('      (sketchpadsonfab no)')
    pcb.append('      (plotpadnumbers no)')
    pcb.append('      (hidednponfab no)')
    pcb.append('      (sketchdnponfab yes)')
    pcb.append('      (crossoutdnponfab yes)')
    pcb.append('      (subtractmaskfromsilk no)')
    pcb.append('      (outputformat 1)')
    pcb.append('      (mirror no)')
    pcb.append('      (drillshape 1)')
    pcb.append('      (scaleselection 1)')
    pcb.append('      (outputdirectory "")')
    pcb.append('    )')
    pcb.append('  )')

    # Nets
    for net_id, net_name in NETS:
        pcb.append(f'  (net {net_id} "{net_name}")')

    # =========================================================
    # Board Outline (Edge.Cuts)
    # =========================================================
    pcb.append('')
    for (x1, y1), (x2, y2) in outline_segs:
        pcb.append(f'  (gr_line')
        pcb.append(f'    (start {x1:.4f} {y1:.4f})')
        pcb.append(f'    (end {x2:.4f} {y2:.4f})')
        pcb.append(f'    (stroke (width 0.05) (type solid))')
        pcb.append(f'    (layer "Edge.Cuts")')
        pcb.append(f'    (uuid "{new_uuid()}")')
        pcb.append(f'  )')

    # =========================================================
    # Footprints
    # =========================================================
    pcb.append('')

    # J1 - U.FL connector (only connector, connects to RF_IN)
    pcb.append(fp_ufl('J1', j1_x, j1_y, j1_rot, sig_net=2, gnd_net=1))

    # JP1 - 2-pad solder jumper (bypass matching network: RF_IN → ANT)
    pcb.append(fp_solder_jumper_2('JP1', jp1_x, jp1_y, jp1_rot,
                                  pad1_net=2, pad2_net=6))

    # Matching network components
    net_name_map = {n[0]: n[1] for n in NETS}
    for ref, ctype, value, net1, net2, _lcsc, _mpn, _mfr in MATCHING_COMPONENTS:
        idx = int(ref[1]) - 1  # L1→0, C1→0, L2→1, etc.
        if ctype == 'L':
            lx, ly = l_positions[idx]
            pcb.append(fp_0603_inductor(ref, value, lx, ly, 0.0, net1, net2))
        else:
            cx, cy = c_positions[idx]
            pcb.append(fp_0603_capacitor(ref, value, cx, cy, 270.0, net1, net2))

    # =========================================================
    # Antenna Traces (F.Cu segments)
    # =========================================================
    pcb.append('')
    for (x1, y1), (x2, y2) in ant_traces:
        pcb.append(
            f'  (segment (start {x1:.4f} {y1:.4f}) (end {x2:.4f} {y2:.4f})'
            f' (width {TRACE_WIDTH_ANT}) (layer "F.Cu") (net 6)'
            f' (uuid "{new_uuid()}"))'
        )

    # Short stub via (antenna short point connects F.Cu to B.Cu GND)
    sx, sy = short_kicad
    pcb.append(
        f'  (via (at {sx:.4f} {sy:.4f}) (size {VIA_SIZE}) (drill {VIA_DRILL})'
        f' (layers "F.Cu" "B.Cu") (net 1) (uuid "{new_uuid()}"))'
    )

    # =========================================================
    # Matching Network Traces (F.Cu segments)
    # =========================================================
    pcb.append('')
    fx, fy = feed_kicad

    # Connect series inductors in chain: L1-pad2 → node → L2-pad1
    # 0603 pad centers at ±0.775mm from component center
    pad_offset = 0.775
    for i in range(3):  # 3 junctions between 4 inductors
        lx1, ly1 = l_positions[i]
        lx2, ly2 = l_positions[i + 1]
        cx_node = c_positions[i][0]  # cap X is the node junction
        sx1 = lx1 + pad_offset   # right pad edge of current inductor
        sx2 = lx2 - pad_offset   # left pad edge of next inductor
        net_id = 3 + i  # N1=3, N2=4, N3=5
        pcb.append(
            f'  (segment (start {sx1:.4f} {ly1:.4f}) (end {cx_node:.4f} {ly1:.4f})'
            f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net {net_id})'
            f' (uuid "{new_uuid()}"))'
        )
        pcb.append(
            f'  (segment (start {cx_node:.4f} {ly1:.4f}) (end {sx2:.4f} {ly2:.4f})'
            f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net {net_id})'
            f' (uuid "{new_uuid()}"))'
        )

    # Connect shunt capacitors to node junctions (vertical traces)
    # 0603 cap at 270° rotation: pad1 at (cx, cy-0.775), pad2 at (cx, cy+0.775)
    # Pad size 0.9×0.95, half-height=0.475 → pad1 top edge at cy-0.775-0.475=cy-1.25
    # Trace endpoints at pad EDGES to prevent round end caps from extending
    for i in range(4):  # 4 shunt caps
        cx, cy = c_positions[i]
        node_net = 3 + i if i < 3 else 6  # N1, N2, N3, ANT
        cap_pad1_edge_y = cy - 1.25  # top edge of pad1
        pcb.append(
            f'  (segment (start {cx:.4f} {cap_pad1_edge_y:.4f}) (end {cx:.4f} {mn_y_l:.4f})'
            f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net {node_net})'
            f' (uuid "{new_uuid()}"))'
        )
        # GND via below cap pad2, with trace from pad2 bottom edge
        cap_pad2_edge_y = cy + 1.25  # bottom edge of pad2
        via_y = cap_pad2_edge_y + 1.0
        pcb.append(
            f'  (via (at {cx:.4f} {via_y:.4f}) (size {VIA_SIZE}) (drill {VIA_DRILL})'
            f' (layers "F.Cu" "B.Cu") (net 1) (uuid "{new_uuid()}"))'
        )
        pcb.append(
            f'  (segment (start {cx:.4f} {cap_pad2_edge_y:.4f}) (end {cx:.4f} {via_y:.4f})'
            f' (width {TRACE_WIDTH_GND}) (layer "F.Cu") (net 1)'
            f' (uuid "{new_uuid()}"))'
        )

    # =========================================================
    # RF_IN bus (net 2): J1 → L1-pad1, JP1-pad1
    # =========================================================
    l1_pad1_x = l_positions[0][0] - pad_offset
    l1_pad1_y = l_positions[0][1]

    # J1 U.FL signal pad at (j1_x - 1.05, j1_y)
    j1_sig_x = j1_x - 1.05

    # Route: J1 signal → up to L1 Y level → right to L1 pad1
    bus_x = j1_sig_x  # vertical bus at J1 signal X
    pcb.append(
        f'  (segment (start {j1_sig_x:.4f} {j1_y:.4f}) (end {bus_x:.4f} {l1_pad1_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 2)'
        f' (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (segment (start {bus_x:.4f} {l1_pad1_y:.4f}) (end {l1_pad1_x:.4f} {l1_pad1_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 2)'
        f' (uuid "{new_uuid()}"))'
    )

    # JP1 bypass pad1 (RF_IN, net 2) connects to RF_IN bus
    jp1_pad1_x = jp1_x - 0.65
    pcb.append(
        f'  (segment (start {jp1_pad1_x:.4f} {jp1_y:.4f}) (end {jp1_pad1_x:.4f} {l1_pad1_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 2)'
        f' (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (segment (start {jp1_pad1_x:.4f} {l1_pad1_y:.4f}) (end {l1_pad1_x:.4f} {l1_pad1_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 2)'
        f' (uuid "{new_uuid()}"))'
    )

    # JP1 pad2 (ANT, net 7) — leave unrouted for manual routing

    # =========================================================
    # ANT route (net 6): L4-pad2 → C4 junction → above matching → feed
    # =========================================================
    l4_pad2_x = l_positions[3][0] + pad_offset
    l4_pad2_y = l_positions[3][1]
    c4_x = c_positions[3][0]
    # L4 pad2 → right to C4 junction
    pcb.append(
        f'  (segment (start {l4_pad2_x:.4f} {l4_pad2_y:.4f}) (end {c4_x:.4f} {l4_pad2_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 6)'
        f' (uuid "{new_uuid()}"))'
    )
    # Route UP (lower Y) from C4 junction, above matching network, left to feed
    ant_route_y = mn_y_l - 4.0  # above inductors
    pcb.append(
        f'  (segment (start {c4_x:.4f} {l4_pad2_y:.4f}) (end {c4_x:.4f} {ant_route_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 6)'
        f' (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (segment (start {c4_x:.4f} {ant_route_y:.4f}) (end {fx:.4f} {ant_route_y:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 6)'
        f' (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (segment (start {fx:.4f} {ant_route_y:.4f}) (end {fx:.4f} {fy:.4f})'
        f' (width {TRACE_WIDTH_RF}) (layer "F.Cu") (net 6)'
        f' (uuid "{new_uuid()}"))'
    )

    # =========================================================
    # Via Stitching (along flat edge)
    # =========================================================
    pcb.append('')
    for vx, vy in via_positions:
        pcb.append(
            f'  (via (at {vx:.4f} {vy:.4f}) (size {VIA_SIZE}) (drill {VIA_DRILL})'
            f' (layers "F.Cu" "B.Cu") (net 1) (uuid "{new_uuid()}"))'
        )

    # =========================================================
    # Ground Plane Zone (B.Cu)
    # =========================================================
    pcb.append('')

    # Large rectangle covering entire board, KiCad auto-clips to Edge.Cuts
    pcb.append(f'  (zone')
    pcb.append(f'    (net 1)')
    pcb.append(f'    (net_name "GND")')
    pcb.append(f'    (layer "B.Cu")')
    pcb.append(f'    (uuid "{new_uuid()}")')
    pcb.append(f'    (hatch edge 0.5)')
    pcb.append(f'    (connect_pads')
    pcb.append(f'      (clearance 0.3)')
    pcb.append(f'    )')
    pcb.append(f'    (min_thickness 0.25)')
    pcb.append(f'    (filled_areas_thickness no)')
    pcb.append(f'    (fill')
    pcb.append(f'      (thermal_gap 0.5)')
    pcb.append(f'      (thermal_bridge_width 0.5)')
    pcb.append(f'    )')
    pcb.append(f'    (polygon')
    pcb.append(f'      (pts')
    pcb.append(f'        (xy 25 40) (xy 95 40) (xy 95 140) (xy 25 140)')
    pcb.append(f'      )')
    pcb.append(f'    )')
    pcb.append(f'  )')

    # Ground zone on F.Cu (body area — keepout prevents antenna area fill)
    pcb.append(f'  (zone')
    pcb.append(f'    (net 1)')
    pcb.append(f'    (net_name "GND")')
    pcb.append(f'    (layer "F.Cu")')
    pcb.append(f'    (uuid "{new_uuid()}")')
    pcb.append(f'    (hatch edge 0.5)')
    pcb.append(f'    (connect_pads')
    pcb.append(f'      (clearance 0.3)')
    pcb.append(f'    )')
    pcb.append(f'    (min_thickness 0.25)')
    pcb.append(f'    (filled_areas_thickness no)')
    pcb.append(f'    (fill')
    pcb.append(f'      (thermal_gap 0.5)')
    pcb.append(f'      (thermal_bridge_width 0.5)')
    pcb.append(f'    )')
    pcb.append(f'    (polygon')
    pcb.append(f'      (pts')
    pcb.append(f'        (xy 25 40) (xy 95 40) (xy 95 140) (xy 25 140)')
    pcb.append(f'      )')
    pcb.append(f'    )')
    pcb.append(f'  )')

    # =========================================================
    # Keepout Zone over semicircle on F.Cu
    # (Prevent copper pour from interfering with antenna traces)
    # =========================================================
    pcb.append('')

    # Generate semicircle polygon in KiCad coords
    n_arc = 72
    arc_angles = np.linspace(0, math.pi, n_arc + 1)
    semi_pts = []
    for a in arc_angles:
        ax = BOARD_RADIUS * math.cos(a)
        ay = BOARD_RADIUS * math.sin(a)
        kx, ky = transform.antenna_to_kicad(ax, ay)
        semi_pts.append((kx, ky))
    # Close with flat edge
    # (The first and last points are already the flat edge endpoints)

    pcb.append(f'  (zone')
    pcb.append(f'    (net 0)')
    pcb.append(f'    (net_name "")')
    pcb.append(f'    (layer "F.Cu")')
    pcb.append(f'    (uuid "{new_uuid()}")')
    pcb.append(f'    (hatch edge 0.5)')
    pcb.append(f'    (connect_pads')
    pcb.append(f'      (clearance 0)')
    pcb.append(f'    )')
    pcb.append(f'    (min_thickness 0.25)')
    pcb.append(f'    (filled_areas_thickness no)')
    pcb.append(f'    (keepout')
    pcb.append(f'      (tracks allowed)')
    pcb.append(f'      (vias allowed)')
    pcb.append(f'      (pads allowed)')
    pcb.append(f'      (copperpour not_allowed)')
    pcb.append(f'      (footprints allowed)')
    pcb.append(f'    )')
    pcb.append(f'    (fill')
    pcb.append(f'      (thermal_gap 0.5)')
    pcb.append(f'      (thermal_bridge_width 0.5)')
    pcb.append(f'    )')
    pcb.append(f'    (polygon')
    pcb.append(f'      (pts')
    pts_str = ' '.join(f'(xy {px:.4f} {py:.4f})' for px, py in semi_pts)
    pcb.append(f'        {pts_str}')
    pcb.append(f'      )')
    pcb.append(f'    )')
    pcb.append(f'  )')

    # =========================================================
    # Annotations on F.Fab
    # =========================================================
    pcb.append('')
    cx, cy = transform.center_kicad
    pcb.append(
        f'  (gr_text "Branched IFA Antenna Test Board\\n'
        f'R=31 groot board, U.FL only, 0603 matching" (at {cx:.2f} {cy + 55:.2f})'
        f' (layer "F.SilkS") (uuid "{new_uuid()}")'
        f' (effects (font (size 1.2 1.2) (thickness 0.12))))'
    )
    # Feed point marker
    fx, fy = feed_kicad
    pcb.append(
        f'  (gr_circle (center {fx:.4f} {fy:.4f}) (end {fx + 1:.4f} {fy:.4f})'
        f' (stroke (width 0.15) (type solid)) (fill no) (layer "F.SilkS") (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (gr_text "FEED" (at {fx + 2:.2f} {fy:.2f})'
        f' (layer "F.SilkS") (uuid "{new_uuid()}")'
        f' (effects (font (size 0.8 0.8) (thickness 0.12))))'
    )
    # Short point marker
    sx, sy = short_kicad
    pcb.append(
        f'  (gr_circle (center {sx:.4f} {sy:.4f}) (end {sx + 1:.4f} {sy:.4f})'
        f' (stroke (width 0.15) (type solid)) (fill no) (layer "F.SilkS") (uuid "{new_uuid()}"))'
    )
    pcb.append(
        f'  (gr_text "SHORT" (at {sx + 2:.2f} {sy:.2f})'
        f' (layer "F.SilkS") (uuid "{new_uuid()}")'
        f' (effects (font (size 0.8 0.8) (thickness 0.12))))'
    )

    pcb.append(')')
    return '\n'.join(pcb)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("KiCad Antenna Test Board Generator")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Initialize coordinate transform
    print("\nCoordinate transform:")
    transform = CoordTransform()
    cx, cy = transform.center_kicad
    print(f"  Antenna center (KiCad): ({cx:.2f}, {cy:.2f})")
    print(f"  Orientation: straight (no rotation)")

    # Generate project file
    print("\nGenerating project file...")
    pro_path = os.path.join(OUTPUT_DIR, 'antenna_test.kicad_pro')
    with open(pro_path, 'w') as f:
        f.write(generate_kicad_pro())
    print(f"  Written: {pro_path}")

    # Generate schematic
    print("\nGenerating schematic...")
    sch_path = os.path.join(OUTPUT_DIR, 'antenna_test.kicad_sch')
    with open(sch_path, 'w') as f:
        f.write(generate_kicad_sch())
    print(f"  Written: {sch_path}")

    # Generate PCB
    print("\nGenerating PCB layout...")
    pcb_path = os.path.join(OUTPUT_DIR, 'antenna_test.kicad_pcb')
    with open(pcb_path, 'w') as f:
        f.write(generate_kicad_pcb(transform))
    print(f"  Written: {pcb_path}")

    # Generate custom DRC rules
    print("\nGenerating custom DRC rules...")
    dru_path = os.path.join(OUTPUT_DIR, 'antenna_test.kicad_dru')
    dru_content = """(version 1)

(rule "Antenna traces at board edge"
  (condition "A.NetName == 'ANT'")
  (constraint edge_clearance (min 0mm))
)

(rule "Via stitching near board edge"
  (condition "A.Type == 'Via' && A.NetName == 'GND'")
  (constraint edge_clearance (min 0mm))
)
"""
    with open(dru_path, 'w') as f:
        f.write(dru_content)
    print(f"  Written: {dru_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DONE! KiCad project created at:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    antenna_test.kicad_pro")
    print(f"    antenna_test.kicad_sch")
    print(f"    antenna_test.kicad_pcb")
    print(f"    antenna_test.kicad_dru")
    print()
    print("Next steps:")
    print("  1. Open antenna_test.kicad_pcb in KiCad PCB Editor")
    print("  2. Run Edit → Fill All Zones (resolves GND unconnected items)")
    print("  3. Run DRC — expect 0 shorts, some edge clearance warnings")
    print("  4. Connect U.FL cable to J1 for testing")
    print("  5. Populate matching network L1-L4, C1-C4 (0603)")
    print("  6. Solder JP1 to bypass matching network for bare antenna test")
    print("=" * 60)


if __name__ == '__main__':
    main()
