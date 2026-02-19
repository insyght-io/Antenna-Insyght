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
Central configuration for the dual-band PCB IFA antenna system.
All dimensions in mm, frequencies in Hz.
"""

import numpy as np

# --- Board Geometry ---
BOARD_WIDTH = 62.0           # mm, half-circle diameter (groot board)
BOARD_HEIGHT = 31.0          # mm, half-circle radius
BOARD_RADIUS = 31.0          # mm, equals BOARD_HEIGHT (groot board R=31)
SUBSTRATE_THICKNESS = 1.6    # mm, FR4
SUBSTRATE_ER = 4.4           # FR4 relative permittivity
SUBSTRATE_TAND = 0.02        # FR4 loss tangent
COPPER_THICKNESS = 0.035     # mm, 1oz copper

# --- Ground Plane ---
GND_EXTENSION = 75.0         # mm below the flat edge (V4 rectangular body depth)
GND_HALF_WIDTH = 31.0        # mm, half-width of rectangular PCB body (62mm total)
GND_GRID_WIRE_RADIUS = 0.3   # mm, wire radius for ground mesh wires

# --- Low Band IFA Parameters (LEFT half, 700-960 MHz) ---
LB_SHORT_POS_X = 21.0        # mm from left edge of half-circle
LB_FEED_OFFSET = 4.0         # mm, short-to-feed offset (impedance control)
LB_ELEMENT_HEIGHT = 4.0      # mm above ground plane edge
LB_TRACE_WIDTH = 1.5         # mm
LB_TOTAL_LENGTH = 97.0       # mm target (will be meandered)
LB_MEANDER_SPACING = 4.5     # mm between parallel runs
LB_NUM_MEANDERS = 3          # number of horizontal runs

# --- High Band IFA Parameters (RIGHT half, 1710-2170 MHz) ---
HB_SHORT_POS_X = 37.0        # mm from left edge (in right half)
HB_FEED_OFFSET = 3.0         # mm
HB_ELEMENT_HEIGHT = 4.0      # mm above ground plane edge
HB_TRACE_WIDTH = 1.0         # mm
HB_TOTAL_LENGTH = 42.0       # mm target
HB_MEANDER_SPACING = 3.5     # mm between parallel runs
HB_NUM_MEANDERS = 2          # number of horizontal runs

# --- Feed ---
Z0 = 50.0                    # ohm, feed impedance

# --- Frequency Bands (NB-IoT / LTE-M: US + EU) ---
# LB: B20 (791-862 EU) + B8 (880-960 EU), stretch B12/B13 (699-787 US)
# HB: B3 (1710-1880 EU) + B2 (1850-1990 US), stretch B4 (1710-2155 US)
LB_FREQ_MIN = 791e6          # Hz (B20 low edge)
LB_FREQ_MAX = 960e6          # Hz (B8 high edge)
LB_FREQ_CENTER = 875e6       # Hz
HB_FREQ_MIN = 1710e6         # Hz (B3/B4 low edge)
HB_FREQ_MAX = 1990e6         # Hz (B2 high edge)
HB_FREQ_CENTER = 1850e6      # Hz

# --- Simulation Settings ---
SIM_FREQ_MIN = 500e6         # Hz, simulation sweep start
SIM_FREQ_MAX = 2500e6        # Hz, simulation sweep stop
SIM_NUM_FREQ = 501           # number of frequency points
SIM_FREQUENCIES = np.linspace(SIM_FREQ_MIN, SIM_FREQ_MAX, SIM_NUM_FREQ)

# NEC2 segmentation
NEC_SEGMENTS_PER_LAMBDA = 20  # segments per wavelength (minimum 10)
NEC_GND_SEGMENTS = 15         # segments for ground plane wires

# --- Performance Targets ---
TARGETS = {
    'lowband': {
        's11_db': -6.0,        # dB, maximum S11 across band
        'vswr_max': 3.0,       # maximum VSWR
        'gain_min_dbi': -2.0,  # dBi, minimum peak gain
        'efficiency_min': 0.20, # minimum radiation efficiency
        'freq_min': LB_FREQ_MIN,
        'freq_max': LB_FREQ_MAX,
        'freq_center': LB_FREQ_CENTER,
    },
    'highband': {
        's11_db': -6.0,
        'vswr_max': 2.5,
        'gain_min_dbi': 0.0,
        'efficiency_min': 0.35,
        'freq_min': HB_FREQ_MIN,
        'freq_max': HB_FREQ_MAX,
        'freq_center': HB_FREQ_CENTER,
    },
}

# --- Optimization Bounds ---
OPT_BOUNDS = {
    'LB_TOTAL_LENGTH':   (85.0, 280.0),
    'LB_FEED_OFFSET':    (2.0, 10.0),
    'LB_ELEMENT_HEIGHT': (3.0, 15.0),
    'LB_MEANDER_SPACING':(3.0, 6.0),
    'HB_TOTAL_LENGTH':   (35.0, 90.0),
    'HB_FEED_OFFSET':    (1.5, 5.0),
    'HB_ELEMENT_HEIGHT': (3.0, 15.0),
    'HB_MEANDER_SPACING':(2.5, 5.0),
}

# --- Mounting Holes ---
MOUNTING_HOLES = [
    (-20.0, -2.0, 1.5),  # (x, y, radius) mm - left hole
    (20.0, -2.0, 1.5),   # right hole
]

# --- Substrate Correction for NEC2 ---
# NEC2 simulates in free space. Real PCB antennas on FR4 resonate lower
# by approximately sqrt(εr_eff). For an IFA elevated 4-6mm above ground
# on 1.6mm FR4, the effective εr is ~2.0 (less than full microstrip εr
# because fields extend into air above the trace).
SUBSTRATE_ER_EFF = 2.0           # effective εr for NEC2 → PCB frequency mapping
NEC2_FREQ_SCALE = np.sqrt(SUBSTRATE_ER_EFF)  # multiply target freq by this for NEC2

# NEC2 evaluation frequencies: where the antenna should resonate in NEC2
# to achieve the PCB target band
NEC2_LB_FREQ_MIN = LB_FREQ_MIN * NEC2_FREQ_SCALE
NEC2_LB_FREQ_MAX = LB_FREQ_MAX * NEC2_FREQ_SCALE
NEC2_LB_FREQ_CENTER = LB_FREQ_CENTER * NEC2_FREQ_SCALE
NEC2_HB_FREQ_MIN = HB_FREQ_MIN * NEC2_FREQ_SCALE
NEC2_HB_FREQ_MAX = HB_FREQ_MAX * NEC2_FREQ_SCALE
NEC2_HB_FREQ_CENTER = HB_FREQ_CENTER * NEC2_FREQ_SCALE

# NEC2-corrected targets for optimization
NEC2_TARGETS = {
    'lowband': {
        's11_db': -6.0,
        'vswr_max': 3.0,
        'freq_min': NEC2_LB_FREQ_MIN,
        'freq_max': NEC2_LB_FREQ_MAX,
        'freq_center': NEC2_LB_FREQ_CENTER,
    },
    'highband': {
        's11_db': -6.0,
        'vswr_max': 2.5,
        'freq_min': NEC2_HB_FREQ_MIN,
        'freq_max': NEC2_HB_FREQ_MAX,
        'freq_center': NEC2_HB_FREQ_CENTER,
    },
}

# --- Antenna Topology ---
ANTENNA_TOPOLOGY = 'branched'  # 'dual_ifa' or 'branched'

# --- Branched IFA Parameters (single-feed, dual-band) ---
# Left-side short/feed design: LB arm spans full diameter, HB branches off
BRANCHED_SHORT_X = -25.0        # mm, short position (R=31 Design B: left-shifted)
BRANCHED_FEED_OFFSET = 7.0      # mm, feed-to-short distance (feed at SHORT_X + offset)
BRANCHED_ELEM_HEIGHT = 14.0     # mm, height above ground plane edge
BRANCHED_LB_LENGTH = 200.0      # mm, LB arm target (clipped to semicircle boundary)
BRANCHED_LB_SPACING = 10.0      # mm, HB vertical branch height
BRANCHED_LB_CAP_W = 0.0         # mm, capacitive tip-load width (disabled)
BRANCHED_LB_CAP_L = 0.0         # mm, capacitive tip-load length (disabled)
BRANCHED_HB_LENGTH = 30.0       # mm, HB branch total length (vertical + horizontal)
BRANCHED_HB_ANGLE = 8.0         # mm, HB branch offset from feed along LB arm
BRANCHED_TRACE_WIDTH = 1.5      # mm

BRANCHED_OPT_BOUNDS = {
    'SHORT_X': (-27.0, -20.0),  # left-shifted for R=31 groot board
    'FEED_OFFSET': (3.0, 8.0),
    'ELEM_HEIGHT': (6.0, 15.0),
    'LB_LENGTH': (100.0, 200.0),
    'LB_SPACING': (6.0, 15.0),
    'LB_CAP_W': (0.0, 0.0),
    'LB_CAP_L': (0.0, 0.0),
    'HB_LENGTH': (15.0, 35.0),
    'HB_ANGLE': (5.0, 15.0),    # HB branch offset from feed (mm)
    'GND_CLEARANCE': (5.0, 15.0),
    'TRACE_WIDTH': (0.8, 2.0),
}

# --- Simulation Backend ---
SIM_BACKEND = 'openems'  # 'nec2', 'openems', or 'hybrid'

# --- openEMS FDTD Settings ---
OPENEMS_END_CRITERIA = 1e-4        # convergence threshold for FDTD (-40 dB energy decay)
OPENEMS_MAX_TIMESTEPS = 200000     # max FDTD timesteps (need ~3x excitation length)
OPENEMS_MESH_RES = 20              # cells per wavelength at max freq
OPENEMS_BC = ['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8']  # PML on all faces
OPENEMS_FAST_TIMESTEPS = 80000     # reduced timesteps for optimization (faster, ~-15dB convergence)
OPENEMS_EXPLORE_TIMESTEPS = 30000  # minimal timesteps for parameter exploration (~-10dB, rough S11)
OPENEMS_GND_CLEARANCE = 10.0       # mm, top copper ground clearance from antenna area edge
                                    # Reference designs use 10-15mm for proper IFA operation

# --- Physical Constants ---
C0 = 299792458.0  # m/s, speed of light


def wavelength(freq_hz):
    """Wavelength in mm for a given frequency in Hz."""
    return C0 / freq_hz * 1000.0


def wavelength_eff(freq_hz, er=SUBSTRATE_ER):
    """Effective wavelength in mm accounting for substrate."""
    return wavelength(freq_hz) / np.sqrt((er + 1) / 2)


def point_in_halfcircle(x, y, cx=0.0, cy=0.0, r=BOARD_RADIUS):
    """Check if a point (x,y) is inside the half-circle.
    Origin at center of flat edge. Half-circle is upper half (y >= 0).
    """
    if y < 0:
        return False
    return (x - cx)**2 + (y - cy)**2 <= r**2


def halfcircle_x_at_y(y, r=BOARD_RADIUS):
    """Return the max x extent of the half-circle at height y.
    Returns (x_left, x_right) or None if y is outside the circle.
    """
    if y < 0 or y > r:
        return None
    x = np.sqrt(r**2 - y**2)
    return (-x, x)
