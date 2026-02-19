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
PDF report generator for the dual-band IFA antenna design.

Generates a comprehensive report using reportlab with:
- Title page with board dimensions
- Design overview and antenna geometry
- Optimized parameter tables
- S11, VSWR, Smith chart plots
- Radiation pattern plots
- Performance summary with pass/fail
- Simulation notes for RF reviewer (backend-aware: NEC2 or openEMS)
"""

import os
import json
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from config import (
    BOARD_RADIUS, BOARD_WIDTH, BOARD_HEIGHT, GND_EXTENSION, GND_HALF_WIDTH,
    SUBSTRATE_THICKNESS, SUBSTRATE_ER, SUBSTRATE_TAND, COPPER_THICKNESS,
    Z0, TARGETS, NEC2_TARGETS, NEC2_FREQ_SCALE, SUBSTRATE_ER_EFF,
    GND_GRID_WIRE_RADIUS, SIM_BACKEND,
    LB_SHORT_POS_X, LB_FEED_OFFSET, LB_ELEMENT_HEIGHT, LB_TRACE_WIDTH,
    LB_TOTAL_LENGTH, LB_MEANDER_SPACING, LB_NUM_MEANDERS,
    HB_SHORT_POS_X, HB_FEED_OFFSET, HB_ELEMENT_HEIGHT, HB_TRACE_WIDTH,
    HB_TOTAL_LENGTH, HB_MEANDER_SPACING, HB_NUM_MEANDERS,
)


PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def _styles():
    """Build paragraph styles for the report."""
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        'ReportTitle', parent=ss['Title'], fontSize=22, spaceAfter=6 * mm,
        alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        'ReportSubtitle', parent=ss['Normal'], fontSize=13,
        alignment=TA_CENTER, spaceAfter=4 * mm, textColor=colors.grey,
    ))
    ss.add(ParagraphStyle(
        'SectionHead', parent=ss['Heading1'], fontSize=15, spaceBefore=8 * mm,
        spaceAfter=3 * mm, textColor=colors.HexColor('#1a5276'),
    ))
    ss.add(ParagraphStyle(
        'SubSection', parent=ss['Heading2'], fontSize=12, spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    ))
    ss.add(ParagraphStyle(
        'BodyText2', parent=ss['BodyText'], fontSize=10, leading=14,
    ))
    ss.add(ParagraphStyle(
        'SmallNote', parent=ss['Normal'], fontSize=8, leading=10,
        textColor=colors.grey,
    ))
    return ss


def _table_style():
    """Standard table style for data tables."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])


def _fit_image(path, max_w=160 * mm, max_h=110 * mm):
    """Return an Image flowable that fits within max dimensions, or None."""
    if not os.path.isfile(path):
        return None
    img = Image(path)
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return None
    scale = min(max_w / iw, max_h / ih, 1.0)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = 'CENTER'
    return img


def _load_band_data(results_dir, band):
    """Load saved .npz data for a band. Returns dict or None."""
    path = os.path.join(results_dir, band, 'data.npz')
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_params(results_dir):
    """Load best_params.json if available."""
    path = os.path.join(results_dir, 'best_params.json')
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _pass_fail(val, threshold, lower_better=True):
    """Return 'PASS' or 'FAIL' string."""
    if lower_better:
        return 'PASS' if val <= threshold else 'FAIL'
    return 'PASS' if val >= threshold else 'FAIL'


def generate_report(results_dir='results', output_path=None, params=None):
    """Generate the PDF antenna design report.

    Args:
        results_dir: directory containing simulation output (plots, data.npz)
        output_path: path for the PDF (default: results_dir/antenna_report.pdf)
        params: optional dict of antenna parameters (overrides best_params.json)
    """
    if output_path is None:
        output_path = os.path.join(results_dir, 'antenna_report.pdf')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    styles = _styles()
    story = []

    # Load data
    saved_params = _load_params(results_dir) or {}
    if params:
        saved_params.update(params)
    lb_data = _load_band_data(results_dir, 'lowband')
    hb_data = _load_band_data(results_dir, 'highband')

    # ================================================================
    # TITLE PAGE
    # ================================================================
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph('Dual-Band IFA Antenna Design Report', styles['ReportTitle']))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        styles['ReportSubtitle'],
    ))
    story.append(Paragraph(
        f'Board: {2*GND_HALF_WIDTH:.0f} mm x {GND_EXTENSION:.0f} mm rectangular body '
        f'+ {BOARD_WIDTH:.0f} mm semicircle antenna area',
        styles['ReportSubtitle'],
    ))
    story.append(Paragraph(
        f'Substrate: FR4 ({SUBSTRATE_THICKNESS} mm, er={SUBSTRATE_ER}, '
        f'tan d={SUBSTRATE_TAND})',
        styles['ReportSubtitle'],
    ))
    story.append(Spacer(1, 15 * mm))

    # Board summary table
    board_data = [
        ['Parameter', 'Value'],
        ['Antenna type', 'Inverted-F Antenna (IFA), conformal arc meander'],
        ['PCB body', f'{2*GND_HALF_WIDTH:.0f} x {GND_EXTENSION:.0f} mm'],
        ['Antenna area', f'{BOARD_WIDTH:.0f} mm diameter semicircle'],
        ['Substrate', f'FR4, {SUBSTRATE_THICKNESS} mm, er={SUBSTRATE_ER}'],
        ['Copper', f'{COPPER_THICKNESS*1000:.0f} um ({COPPER_THICKNESS/0.035:.0f} oz)'],
        ['Low band target', '700 - 960 MHz (LTE B5/B8/B20)'],
        ['High band target', '1710 - 2170 MHz (LTE B1/B3/B7)'],
        ['Feed impedance', f'{Z0:.0f} ohm'],
    ]
    t = Table(board_data, colWidths=[55 * mm, 105 * mm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(PageBreak())

    # ================================================================
    # DESIGN OVERVIEW
    # ================================================================
    story.append(Paragraph('1. Design Overview', styles['SectionHead']))
    story.append(Paragraph(
        'This report documents a dual-band Inverted-F Antenna (IFA) design for '
        'cellular IoT applications. Two independent IFAs are placed in a semicircular '
        'antenna clearance area at the top of the PCB, with a shared rectangular '
        'ground plane forming the main PCB body below. The low-band IFA (left half) '
        'uses conformal arc meanders to achieve the long trace length needed for '
        'sub-GHz resonance. The high-band IFA (right half) uses a shorter trace. '
        'Both antennas use shorting stubs and offset feed points for impedance matching.',
        styles['BodyText2'],
    ))
    story.append(Spacer(1, 4 * mm))

    # ================================================================
    # ANTENNA GEOMETRY
    # ================================================================
    story.append(Paragraph('2. Antenna Geometry', styles['SectionHead']))

    layout_img = _fit_image(
        os.path.join(results_dir, 'antenna_layout.png'), max_h=130 * mm)
    if layout_img:
        story.append(layout_img)
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            'Figure 1: Antenna layout showing both IFAs in the semicircular clearance '
            'area and the rectangular PCB ground plane below.',
            styles['SmallNote'],
        ))
    story.append(Spacer(1, 4 * mm))

    # ================================================================
    # OPTIMIZED PARAMETERS
    # ================================================================
    story.append(Paragraph('3. Antenna Parameters', styles['SectionHead']))

    def _pval(key, default, fmt='.1f'):
        v = saved_params.get(key, default)
        return f'{v:{fmt}}'

    param_rows = [
        ['Parameter', 'Low Band', 'High Band', 'Unit'],
        ['Total trace length',
         _pval('LB_TOTAL_LENGTH', LB_TOTAL_LENGTH),
         _pval('HB_TOTAL_LENGTH', HB_TOTAL_LENGTH), 'mm'],
        ['Feed offset',
         _pval('LB_FEED_OFFSET', LB_FEED_OFFSET),
         _pval('HB_FEED_OFFSET', HB_FEED_OFFSET), 'mm'],
        ['Element height',
         _pval('LB_ELEMENT_HEIGHT', LB_ELEMENT_HEIGHT),
         _pval('HB_ELEMENT_HEIGHT', HB_ELEMENT_HEIGHT), 'mm'],
        ['Meander spacing',
         _pval('LB_MEANDER_SPACING', LB_MEANDER_SPACING),
         _pval('HB_MEANDER_SPACING', HB_MEANDER_SPACING), 'mm'],
        ['Trace width',
         f'{LB_TRACE_WIDTH:.1f}', f'{HB_TRACE_WIDTH:.1f}', 'mm'],
        ['Short position (from left)',
         f'{LB_SHORT_POS_X:.1f}', f'{HB_SHORT_POS_X:.1f}', 'mm'],
    ]
    t = Table(param_rows, colWidths=[45 * mm, 30 * mm, 30 * mm, 15 * mm])
    t.setStyle(_table_style())
    story.append(t)

    if saved_params:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            'Values shown reflect optimizer output (best_params.json) where available, '
            'otherwise config.py defaults.',
            styles['SmallNote'],
        ))
    story.append(PageBreak())

    # ================================================================
    # S11 / VSWR PERFORMANCE
    # ================================================================
    story.append(Paragraph('4. S11 and VSWR Performance', styles['SectionHead']))

    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        story.append(Paragraph(f'4.{1 if band=="lowband" else 2}. {label}', styles['SubSection']))
        band_dir = os.path.join(results_dir, band)

        s11_img = _fit_image(os.path.join(band_dir, 's11.png'))
        if s11_img:
            story.append(s11_img)
            story.append(Spacer(1, 2 * mm))

        vswr_img = _fit_image(os.path.join(band_dir, 'vswr.png'))
        if vswr_img:
            story.append(vswr_img)
            story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    # ================================================================
    # SMITH CHARTS
    # ================================================================
    story.append(Paragraph('5. Smith Charts', styles['SectionHead']))

    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        smith_img = _fit_image(
            os.path.join(results_dir, band, 'smith.png'), max_h=100 * mm)
        if smith_img:
            story.append(Paragraph(label, styles['SubSection']))
            story.append(smith_img)
            story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    # ================================================================
    # RADIATION PATTERNS
    # ================================================================
    story.append(Paragraph('6. Radiation Patterns', styles['SectionHead']))

    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        story.append(Paragraph(label, styles['SubSection']))
        band_dir = os.path.join(results_dir, band)

        e_img = _fit_image(os.path.join(band_dir, 'pattern_e.png'), max_h=90 * mm)
        h_img = _fit_image(os.path.join(band_dir, 'pattern_h.png'), max_h=90 * mm)

        if e_img:
            story.append(e_img)
            story.append(Spacer(1, 2 * mm))
        if h_img:
            story.append(h_img)
            story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    # ================================================================
    # PERFORMANCE SUMMARY TABLE
    # ================================================================
    story.append(Paragraph('7. Performance Summary', styles['SectionHead']))

    perf_header = ['Metric', 'Low Band', 'LB Target', 'LB Status',
                   'High Band', 'HB Target', 'HB Status']
    perf_rows = [perf_header]

    for band, data, col_offset in [('lowband', lb_data, 1), ('highband', hb_data, 4)]:
        targets = TARGETS[band]
        if data is None:
            continue

        s11_vals = data.get('s11_db', np.array([0]))
        vswr_vals = data.get('vswr', np.array([0]))
        freqs = data.get('freqs', np.array([0]))
        gain_max = float(data.get('gain_max', 0))
        efficiency = float(data.get('efficiency', 0))

        # Find in-band metrics
        band_mask = (freqs >= targets['freq_min']) & (freqs <= targets['freq_max'])
        if np.any(band_mask):
            worst_s11 = float(np.max(s11_vals[band_mask]))
            best_s11 = float(np.min(s11_vals[band_mask]))
            worst_vswr = float(np.max(vswr_vals[band_mask]))
        else:
            worst_s11 = float(np.max(s11_vals))
            best_s11 = float(np.min(s11_vals))
            worst_vswr = float(np.max(vswr_vals))

        # Find resonance frequency
        min_idx = np.argmin(s11_vals)
        res_freq = float(freqs[min_idx]) / 1e6

        if band == 'lowband':
            perf_rows.append(['Worst S11 (dB)',
                              f'{worst_s11:.1f}', f'{targets["s11_db"]:.0f}',
                              _pass_fail(worst_s11, targets['s11_db']),
                              '', '', ''])
            perf_rows.append(['Best S11 (dB)',
                              f'{best_s11:.1f}', '-', '-',
                              '', '', ''])
            perf_rows.append(['Worst VSWR',
                              f'{worst_vswr:.1f}', f'{targets["vswr_max"]:.1f}',
                              _pass_fail(worst_vswr, targets['vswr_max']),
                              '', '', ''])
            perf_rows.append(['Resonance (MHz)',
                              f'{res_freq:.0f}', '-', '-',
                              '', '', ''])
            perf_rows.append(['Peak Gain (dBi)',
                              f'{gain_max:.1f}',
                              f'{targets.get("gain_min_dbi", -99):.0f}',
                              _pass_fail(gain_max, targets.get('gain_min_dbi', -99), False),
                              '', '', ''])
            perf_rows.append(['Efficiency (%)',
                              f'{efficiency*100:.0f}',
                              f'{targets.get("efficiency_min", 0)*100:.0f}',
                              _pass_fail(efficiency, targets.get('efficiency_min', 0), False),
                              '', '', ''])
        else:
            # Fill in high band columns for existing rows
            for row_idx, (ws11, bs11, wv, rf, gm, eff) in enumerate([
                (worst_s11, best_s11, worst_vswr, res_freq, gain_max, efficiency)
            ]):
                if len(perf_rows) > 1:
                    perf_rows[1][4] = f'{ws11:.1f}'
                    perf_rows[1][5] = f'{targets["s11_db"]:.0f}'
                    perf_rows[1][6] = _pass_fail(ws11, targets['s11_db'])
                if len(perf_rows) > 2:
                    perf_rows[2][4] = f'{bs11:.1f}'
                    perf_rows[2][5] = '-'
                    perf_rows[2][6] = '-'
                if len(perf_rows) > 3:
                    perf_rows[3][4] = f'{wv:.1f}'
                    perf_rows[3][5] = f'{targets["vswr_max"]:.1f}'
                    perf_rows[3][6] = _pass_fail(wv, targets['vswr_max'])
                if len(perf_rows) > 4:
                    perf_rows[4][4] = f'{rf:.0f}'
                    perf_rows[4][5] = '-'
                    perf_rows[4][6] = '-'
                if len(perf_rows) > 5:
                    perf_rows[5][4] = f'{gm:.1f}'
                    perf_rows[5][5] = f'{targets.get("gain_min_dbi", -99):.0f}'
                    perf_rows[5][6] = _pass_fail(gm, targets.get('gain_min_dbi', -99), False)
                if len(perf_rows) > 6:
                    perf_rows[6][4] = f'{eff*100:.0f}'
                    perf_rows[6][5] = f'{targets.get("efficiency_min", 0)*100:.0f}'
                    perf_rows[6][6] = _pass_fail(eff, targets.get('efficiency_min', 0), False)

    if len(perf_rows) > 1:
        col_widths = [30 * mm, 20 * mm, 20 * mm, 15 * mm, 20 * mm, 20 * mm, 15 * mm]
        t = Table(perf_rows, colWidths=col_widths)
        ts = _table_style()
        # Color pass/fail cells
        for row_idx in range(1, len(perf_rows)):
            for col_idx in [3, 6]:
                cell_val = perf_rows[row_idx][col_idx]
                if cell_val == 'PASS':
                    ts.add('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx),
                           colors.HexColor('#27ae60'))
                elif cell_val == 'FAIL':
                    ts.add('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx),
                           colors.HexColor('#e74c3c'))
                    ts.add('FONTNAME', (col_idx, row_idx), (col_idx, row_idx),
                           'Helvetica-Bold')
        t.setStyle(ts)
        story.append(t)
    else:
        story.append(Paragraph(
            'No simulation data found. Run the simulation first.',
            styles['BodyText2'],
        ))

    story.append(PageBreak())

    # ================================================================
    # SIMULATION NOTES
    # ================================================================
    # Determine which backend was used from optimization log
    opt_log_path = os.path.join(results_dir, 'optimization_log.json')
    sim_backend = SIM_BACKEND
    if os.path.isfile(opt_log_path):
        with open(opt_log_path) as f:
            _opt_log = json.load(f)
        sim_backend = _opt_log.get('backend', SIM_BACKEND)

    story.append(Paragraph('8. Simulation Notes', styles['SectionHead']))
    story.append(Paragraph(
        'The following notes describe the simulation methodology, assumptions, and '
        'limitations. This information is provided for RF expert review.',
        styles['BodyText2'],
    ))
    story.append(Spacer(1, 3 * mm))

    if sim_backend in ('openems', 'hybrid'):
        story.append(Paragraph('FDTD Simulation Method (openEMS)', styles['SubSection']))
        story.append(Paragraph(
            'The antenna was simulated using the openEMS FDTD (Finite-Difference '
            'Time-Domain) solver. Unlike method-of-moments solvers, FDTD natively '
            'models the FR4 substrate, copper ground planes, and dielectric losses. '
            'No frequency correction factor is needed. The simulation domain includes '
            'the full PCB stackup: FR4 substrate (er=%.1f, tan d=%.3f, %.1f mm thick), '
            'top and bottom copper layers, antenna traces, lumped feed port through the '
            'substrate, and shorting via. Absorbing boundary conditions (MUR/PML) '
            'terminate the simulation domain.' % (
                SUBSTRATE_ER, SUBSTRATE_TAND, SUBSTRATE_THICKNESS),
            styles['BodyText2'],
        ))
        story.append(Spacer(1, 3 * mm))

        if sim_backend == 'hybrid':
            story.append(Paragraph('Hybrid Optimization Strategy', styles['SubSection']))
            story.append(Paragraph(
                'Optimization used a hybrid approach: Phases 1-2 (coarse arm length and '
                'feed offset sweeps) were run using NEC2 for speed, then Phases 3-4 '
                '(Bayesian refinement and verification) used the openEMS FDTD solver '
                'for accuracy. This balances fast exploration with precise final results.',
                styles['BodyText2'],
            ))
            story.append(Spacer(1, 3 * mm))

    if sim_backend in ('nec2', 'hybrid'):
        story.append(Paragraph('NEC2 Ground Plane Model', styles['SubSection']))
        story.append(Paragraph(
            f'The PCB ground plane is modeled as a wire grid ({2*GND_HALF_WIDTH:.0f} mm wide x '
            f'{GND_EXTENSION:.0f} mm deep) using NEC2 wire elements. The grid uses 7 vertical '
            f'x-positions with ~11 mm spacing and 9 horizontal z-levels (0 to -{GND_EXTENSION:.0f} mm '
            f'in ~11 mm steps). All wires at grid intersections share exact endpoints for proper '
            f'galvanic connection. Wire radius: {GND_GRID_WIRE_RADIUS} mm.',
            styles['BodyText2'],
        ))
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph('NEC2 Substrate Correction Factor', styles['SubSection']))
        story.append(Paragraph(
            f'NEC2 simulates in free space and cannot directly model dielectric substrates. '
            f'To account for the FR4 PCB substrate (er={SUBSTRATE_ER}), a frequency scaling '
            f'factor of sqrt(er_eff) = sqrt({SUBSTRATE_ER_EFF:.1f}) = {NEC2_FREQ_SCALE:.3f} is '
            f'applied. The effective er of {SUBSTRATE_ER_EFF:.1f} accounts for the elevated '
            f'IFA geometry where fields extend into air above the trace.',
            styles['BodyText2'],
        ))
        story.append(Spacer(1, 3 * mm))

    story.append(Paragraph('Assumptions and Limitations', styles['SubSection']))

    # Common notes for all backends
    notes = [
        'Component loading (nRF9151, passives, connectors) on the PCB is not modeled. '
        'These may detune the antenna slightly.',
        'The semicircular antenna clearance area has no ground copper underneath. '
        'The actual keepout area should extend at least 1 mm beyond trace edges.',
        'Mounting holes and edge effects are not modeled.',
        'Mutual coupling between the two IFAs is accounted for in the simulation '
        '(both share the ground plane), but each is simulated independently.',
    ]

    if sim_backend == 'nec2':
        notes = [
            'PCB traces are modeled as round wires with equivalent radius '
            '(trace_width / 4), a standard NEC2 approximation for microstrip.',
            'The ground plane wire grid is a finite approximation. Real PCB ground '
            'is a continuous copper pour with higher conductivity than the wire model.',
            'Dielectric losses are estimated analytically, not modeled in NEC2. '
            'Real FR4 losses at RF will reduce efficiency and broaden bandwidth.',
        ] + notes

    for note in notes:
        story.append(Paragraph(f'- {note}', styles['BodyText2']))
        story.append(Spacer(1, 1.5 * mm))

    # Optimization log info (opt_log_path already resolved above)
    if os.path.isfile(opt_log_path):
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph('Optimization Log', styles['SubSection']))
        with open(opt_log_path) as f:
            opt_log = json.load(f)
        elapsed = opt_log.get('elapsed_seconds', 0)
        phases = opt_log.get('phases', {})
        story.append(Paragraph(
            f'Total optimization time: {elapsed:.0f} s ({elapsed/60:.1f} min). '
            f'Phases completed: {", ".join(sorted(phases.keys()))}.',
            styles['BodyText2'],
        ))
        if 'phase3' in phases:
            p3 = phases['phase3']
            story.append(Paragraph(
                f'Bayesian optimization: {p3.get("n_evals", "?")} iterations, '
                f'best cost = {p3.get("best_cost", "?"):.2f}.',
                styles['BodyText2'],
            ))

    # ================================================================
    # BUILD PDF
    # ================================================================
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='Dual-Band IFA Antenna Design Report',
        author='Antenna Design Tool',
    )
    doc.build(story)
    print(f"Report saved to {output_path}")
    return output_path


def generate_report_branched(results_dir='results', output_path=None, params=None):
    """Generate PDF report for branched IFA topology.

    Args:
        results_dir: directory containing simulation output
        output_path: path for PDF
        params: optional dict of antenna parameters
    """
    from config import (
        BRANCHED_SHORT_X, BRANCHED_FEED_OFFSET, BRANCHED_ELEM_HEIGHT,
        BRANCHED_LB_LENGTH, BRANCHED_LB_SPACING,
        BRANCHED_LB_CAP_W, BRANCHED_LB_CAP_L,
        BRANCHED_HB_LENGTH, BRANCHED_HB_ANGLE, BRANCHED_TRACE_WIDTH,
    )

    if output_path is None:
        output_path = os.path.join(results_dir, 'antenna_report.pdf')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    styles = _styles()
    story = []

    saved_params = _load_params(results_dir) or {}
    if params:
        saved_params.update(params)
    lb_data = _load_band_data(results_dir, 'lowband')
    hb_data = _load_band_data(results_dir, 'highband')

    # TITLE PAGE
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph('Branched IFA Antenna Design Report', styles['ReportTitle']))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        styles['ReportSubtitle'],
    ))
    story.append(Paragraph(
        f'Board: {2*GND_HALF_WIDTH:.0f} mm x {GND_EXTENSION:.0f} mm body '
        f'+ {BOARD_WIDTH:.0f} mm semicircle antenna area',
        styles['ReportSubtitle'],
    ))
    story.append(Paragraph(
        f'Substrate: FR4 ({SUBSTRATE_THICKNESS} mm, er={SUBSTRATE_ER}, '
        f'tan d={SUBSTRATE_TAND})',
        styles['ReportSubtitle'],
    ))
    story.append(Spacer(1, 15 * mm))

    board_data = [
        ['Parameter', 'Value'],
        ['Antenna type', 'Branched IFA (single-feed, dual-band)'],
        ['PCB body', f'{2*GND_HALF_WIDTH:.0f} x {GND_EXTENSION:.0f} mm'],
        ['Antenna area', f'{BOARD_WIDTH:.0f} mm diameter semicircle (full)'],
        ['Substrate', f'FR4, {SUBSTRATE_THICKNESS} mm, er={SUBSTRATE_ER}'],
        ['Copper', f'{COPPER_THICKNESS*1000:.0f} um ({COPPER_THICKNESS/0.035:.0f} oz)'],
        ['Low band target', '700 - 960 MHz (LTE B5/B8/B20)'],
        ['High band target', '1710 - 2170 MHz (LTE B1/B3/B7)'],
        ['Feed impedance', f'{Z0:.0f} ohm'],
    ]
    t = Table(board_data, colWidths=[55 * mm, 105 * mm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(PageBreak())

    # DESIGN OVERVIEW
    story.append(Paragraph('1. Design Overview', styles['SectionHead']))
    story.append(Paragraph(
        'This report documents a single-feed branched Inverted-F Antenna (IFA) design '
        'for cellular IoT applications. Unlike a dual-IFA approach, the branched IFA uses '
        'the ENTIRE semicircular clearance area for one antenna structure. A long meandered '
        'LB arm provides sub-GHz resonance, while a shorter HB branch extending in the '
        'opposite direction creates the high-band resonance. Both bands share a single feed '
        'point and shorting stub, enabling dual-band operation from one simulation run.',
        styles['BodyText2'],
    ))
    story.append(Spacer(1, 4 * mm))

    # ANTENNA GEOMETRY
    story.append(Paragraph('2. Antenna Geometry', styles['SectionHead']))
    layout_img = _fit_image(
        os.path.join(results_dir, 'antenna_layout.png'), max_h=130 * mm)
    if layout_img:
        story.append(layout_img)
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            'Figure 1: Branched IFA layout showing LB arm (blue), HB branch (red), '
            'and capacitive tip load (green) in the semicircular clearance area.',
            styles['SmallNote'],
        ))
    story.append(Spacer(1, 4 * mm))

    # PARAMETERS
    story.append(Paragraph('3. Antenna Parameters', styles['SectionHead']))

    def _pval(key, default, fmt='.1f'):
        v = saved_params.get(key, default)
        return f'{v:{fmt}}'

    param_rows = [
        ['Parameter', 'Value', 'Unit'],
        ['Short position (X)', _pval('SHORT_X', BRANCHED_SHORT_X), 'mm'],
        ['Feed offset', _pval('FEED_OFFSET', BRANCHED_FEED_OFFSET), 'mm'],
        ['Element height', _pval('ELEM_HEIGHT', BRANCHED_ELEM_HEIGHT), 'mm'],
        ['LB arm length', _pval('LB_LENGTH', BRANCHED_LB_LENGTH), 'mm'],
        ['LB meander spacing', _pval('LB_SPACING', BRANCHED_LB_SPACING), 'mm'],
        ['LB cap load width', _pval('LB_CAP_W', BRANCHED_LB_CAP_W), 'mm'],
        ['LB cap load length', _pval('LB_CAP_L', BRANCHED_LB_CAP_L), 'mm'],
        ['HB branch length', _pval('HB_LENGTH', BRANCHED_HB_LENGTH), 'mm'],
        ['HB branch angle', _pval('HB_ANGLE', BRANCHED_HB_ANGLE), 'deg'],
        ['Trace width', f'{BRANCHED_TRACE_WIDTH:.1f}', 'mm'],
    ]
    t = Table(param_rows, colWidths=[50 * mm, 30 * mm, 15 * mm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(PageBreak())

    # COMBINED S11
    story.append(Paragraph('4. S11 Performance (Combined)', styles['SectionHead']))
    combined_img = _fit_image(os.path.join(results_dir, 's11_combined.png'))
    if combined_img:
        story.append(combined_img)
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            'Figure 2: S11 across full frequency range showing both LB and HB bands.',
            styles['SmallNote'],
        ))
    story.append(Spacer(1, 4 * mm))

    # Per-band S11/VSWR
    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        story.append(Paragraph(f'4.{1 if band=="lowband" else 2}. {label}', styles['SubSection']))
        band_dir = os.path.join(results_dir, band)

        s11_img = _fit_image(os.path.join(band_dir, 's11.png'))
        if s11_img:
            story.append(s11_img)
            story.append(Spacer(1, 2 * mm))

        vswr_img = _fit_image(os.path.join(band_dir, 'vswr.png'))
        if vswr_img:
            story.append(vswr_img)
            story.append(Spacer(1, 4 * mm))

    story.append(PageBreak())

    # SMITH CHARTS
    story.append(Paragraph('5. Smith Charts', styles['SectionHead']))
    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        smith_img = _fit_image(
            os.path.join(results_dir, band, 'smith.png'), max_h=100 * mm)
        if smith_img:
            story.append(Paragraph(label, styles['SubSection']))
            story.append(smith_img)
            story.append(Spacer(1, 4 * mm))
    story.append(PageBreak())

    # RADIATION PATTERNS
    story.append(Paragraph('6. Radiation Patterns', styles['SectionHead']))
    for band, label in [('lowband', 'Low Band'), ('highband', 'High Band')]:
        story.append(Paragraph(label, styles['SubSection']))
        band_dir = os.path.join(results_dir, band)
        e_img = _fit_image(os.path.join(band_dir, 'pattern_e.png'), max_h=90 * mm)
        h_img = _fit_image(os.path.join(band_dir, 'pattern_h.png'), max_h=90 * mm)
        if e_img:
            story.append(e_img)
            story.append(Spacer(1, 2 * mm))
        if h_img:
            story.append(h_img)
            story.append(Spacer(1, 4 * mm))
    story.append(PageBreak())

    # PERFORMANCE SUMMARY
    story.append(Paragraph('7. Performance Summary', styles['SectionHead']))

    perf_header = ['Metric', 'Low Band', 'LB Target', 'LB Status',
                   'High Band', 'HB Target', 'HB Status']
    perf_rows = [perf_header]

    for band, data, col_offset in [('lowband', lb_data, 1), ('highband', hb_data, 4)]:
        targets = TARGETS[band]
        if data is None:
            continue

        s11_vals = data.get('s11_db', np.array([0]))
        vswr_vals = data.get('vswr', np.array([0]))
        freqs = data.get('freqs', np.array([0]))
        gain_max = float(data.get('gain_max', 0))
        efficiency = float(data.get('efficiency', 0))

        band_mask = (freqs >= targets['freq_min']) & (freqs <= targets['freq_max'])
        if np.any(band_mask):
            worst_s11 = float(np.max(s11_vals[band_mask]))
            best_s11 = float(np.min(s11_vals[band_mask]))
            worst_vswr = float(np.max(vswr_vals[band_mask]))
        else:
            worst_s11 = float(np.max(s11_vals))
            best_s11 = float(np.min(s11_vals))
            worst_vswr = float(np.max(vswr_vals))

        min_idx = np.argmin(s11_vals)
        res_freq = float(freqs[min_idx]) / 1e6

        if band == 'lowband':
            perf_rows.append(['Worst S11 (dB)',
                              f'{worst_s11:.1f}', f'{targets["s11_db"]:.0f}',
                              _pass_fail(worst_s11, targets['s11_db']),
                              '', '', ''])
            perf_rows.append(['Best S11 (dB)',
                              f'{best_s11:.1f}', '-', '-', '', '', ''])
            perf_rows.append(['Worst VSWR',
                              f'{worst_vswr:.1f}', f'{targets["vswr_max"]:.1f}',
                              _pass_fail(worst_vswr, targets['vswr_max']),
                              '', '', ''])
            perf_rows.append(['Resonance (MHz)',
                              f'{res_freq:.0f}', '-', '-', '', '', ''])
            perf_rows.append(['Peak Gain (dBi)',
                              f'{gain_max:.1f}',
                              f'{targets.get("gain_min_dbi", -99):.0f}',
                              _pass_fail(gain_max, targets.get('gain_min_dbi', -99), False),
                              '', '', ''])
            perf_rows.append(['Efficiency (%)',
                              f'{efficiency*100:.0f}',
                              f'{targets.get("efficiency_min", 0)*100:.0f}',
                              _pass_fail(efficiency, targets.get('efficiency_min', 0), False),
                              '', '', ''])
        else:
            if len(perf_rows) > 1:
                perf_rows[1][4] = f'{worst_s11:.1f}'
                perf_rows[1][5] = f'{targets["s11_db"]:.0f}'
                perf_rows[1][6] = _pass_fail(worst_s11, targets['s11_db'])
            if len(perf_rows) > 2:
                perf_rows[2][4] = f'{best_s11:.1f}'
                perf_rows[2][5] = '-'
                perf_rows[2][6] = '-'
            if len(perf_rows) > 3:
                perf_rows[3][4] = f'{worst_vswr:.1f}'
                perf_rows[3][5] = f'{targets["vswr_max"]:.1f}'
                perf_rows[3][6] = _pass_fail(worst_vswr, targets['vswr_max'])
            if len(perf_rows) > 4:
                perf_rows[4][4] = f'{res_freq:.0f}'
                perf_rows[4][5] = '-'
                perf_rows[4][6] = '-'
            if len(perf_rows) > 5:
                perf_rows[5][4] = f'{gain_max:.1f}'
                perf_rows[5][5] = f'{targets.get("gain_min_dbi", -99):.0f}'
                perf_rows[5][6] = _pass_fail(gain_max, targets.get('gain_min_dbi', -99), False)
            if len(perf_rows) > 6:
                perf_rows[6][4] = f'{efficiency*100:.0f}'
                perf_rows[6][5] = f'{targets.get("efficiency_min", 0)*100:.0f}'
                perf_rows[6][6] = _pass_fail(efficiency, targets.get('efficiency_min', 0), False)

    if len(perf_rows) > 1:
        col_widths = [30 * mm, 20 * mm, 20 * mm, 15 * mm, 20 * mm, 20 * mm, 15 * mm]
        t = Table(perf_rows, colWidths=col_widths)
        ts = _table_style()
        for row_idx in range(1, len(perf_rows)):
            for col_idx in [3, 6]:
                cell_val = perf_rows[row_idx][col_idx]
                if cell_val == 'PASS':
                    ts.add('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx),
                           colors.HexColor('#27ae60'))
                elif cell_val == 'FAIL':
                    ts.add('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx),
                           colors.HexColor('#e74c3c'))
                    ts.add('FONTNAME', (col_idx, row_idx), (col_idx, row_idx),
                           'Helvetica-Bold')
        t.setStyle(ts)
        story.append(t)

    story.append(PageBreak())

    # SIMULATION NOTES
    story.append(Paragraph('8. Simulation Notes', styles['SectionHead']))
    story.append(Paragraph(
        'The branched IFA was simulated using the openEMS FDTD solver. A single '
        'broadband FDTD run covers both LB and HB bands simultaneously, extracting '
        'S11 via Fourier transform of time-domain port signals. The simulation domain '
        'includes FR4 substrate (er=%.1f, tan d=%.3f, %.1f mm thick), top and bottom '
        'copper layers, all antenna traces, lumped feed port, and shorting via.' % (
            SUBSTRATE_ER, SUBSTRATE_TAND, SUBSTRATE_THICKNESS),
        styles['BodyText2'],
    ))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph('Assumptions and Limitations', styles['SubSection']))
    notes = [
        'Single feed point serves both bands; no external diplexer needed.',
        'Capacitive tip load modeled as zero-thickness PEC sheet at z=sub_t.',
        'Component loading (nRF9151, passives, connectors) not modeled.',
        'Mounting holes and edge effects not modeled.',
        'LB arm uses full semicircle for maximum electrical size (ka ~ 1.3 at 830 MHz).',
    ]
    for note in notes:
        story.append(Paragraph(f'- {note}', styles['BodyText2']))
        story.append(Spacer(1, 1.5 * mm))

    # BUILD PDF
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='Branched IFA Antenna Design Report',
        author='Antenna Design Tool',
    )
    doc.build(story)
    print(f"Report saved to {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate antenna design PDF report')
    parser.add_argument('--results', default='results', help='Results directory')
    parser.add_argument('--output', default=None, help='Output PDF path')
    parser.add_argument('--topology', choices=['dual_ifa', 'branched'],
                        default=None, help='Antenna topology')
    args = parser.parse_args()

    from config import ANTENNA_TOPOLOGY
    topology = args.topology or ANTENNA_TOPOLOGY

    if topology == 'branched':
        generate_report_branched(results_dir=args.results, output_path=args.output)
    else:
        generate_report(results_dir=args.results, output_path=args.output)
