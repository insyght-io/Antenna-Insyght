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
Parametric dual-band IFA antenna model for NEC2 simulation.

Constructs meandered Inverted-F Antenna geometry within a half-circle
boundary. Each IFA consists of:
  - A shorting stub (connects radiating element to ground plane)
  - A feed point (offset from short, lumped port to ground)
  - A horizontal radiating element that meanders upward within the boundary
  - Open-ended termination

All coordinates use origin at center of flat edge of the half-circle.
NEC2 works in meters; this module handles mm->m conversion at the boundary.
"""

import numpy as np
from config import (
    BOARD_RADIUS, GND_EXTENSION, SUBSTRATE_ER, SUBSTRATE_THICKNESS,
    COPPER_THICKNESS, Z0, C0, NEC_SEGMENTS_PER_LAMBDA,
    point_in_halfcircle, halfcircle_x_at_y,
)


MM_TO_M = 1e-3
# Wire radius approximation for PCB trace: width / 4 is a common approximation
# for the equivalent wire radius of a microstrip trace
TRACE_TO_WIRE_FACTOR = 0.25


def trace_width_to_wire_radius(trace_width_mm):
    """Convert PCB trace width to equivalent NEC2 wire radius (in mm)."""
    return trace_width_mm * TRACE_TO_WIRE_FACTOR


def segments_for_length(length_mm, freq_hz, min_seg=3, wire_radius_mm=0.5):
    """Calculate number of NEC2 segments for a given wire length and frequency.

    Enforces NEC2 thin-wire constraint: segment length must be >= 4x wire diameter.
    """
    wavelength_mm = C0 / freq_hz * 1000.0
    seg_length = wavelength_mm / NEC_SEGMENTS_PER_LAMBDA
    n = max(min_seg, int(np.ceil(length_mm / seg_length)))
    # NEC2 thin-wire constraint: seg_length >= 4 * diameter = 8 * radius
    min_seg_length = 8 * wire_radius_mm
    max_segs = max(1, int(length_mm / min_seg_length))
    return min(n, max_segs)


class IFAAntenna:
    """Parametric Inverted-F Antenna within a half-circle boundary."""

    def __init__(self, band='lowband',
                 short_pos_x=21.0, feed_offset=4.0, element_height=4.0,
                 trace_width=1.5, total_length=97.0, meander_spacing=4.5,
                 num_meanders=3, direction='left'):
        """
        Args:
            band: 'lowband' or 'highband'
            short_pos_x: x position of shorting stub (mm from left edge, so actual x = short_pos_x - BOARD_RADIUS)
            feed_offset: distance from short to feed (mm)
            element_height: height of first horizontal run above ground edge (mm)
            trace_width: trace width in mm
            total_length: target total radiating length in mm
            meander_spacing: vertical spacing between meander runs (mm)
            num_meanders: number of horizontal meander runs
            direction: 'left' (element extends left, for lowband) or 'right' (for highband)
        """
        self.band = band
        self.direction = direction
        self.trace_width = trace_width
        self.wire_radius = trace_width_to_wire_radius(trace_width)
        self.element_height = element_height
        self.meander_spacing = meander_spacing
        self.num_meanders = num_meanders
        self.total_length = total_length

        # Convert short_pos_x from "mm from left edge" to centered coordinates
        self.short_x = short_pos_x - BOARD_RADIUS
        if direction == 'left':
            self.feed_x = self.short_x + feed_offset
        else:
            self.feed_x = self.short_x - feed_offset

        self.feed_offset = feed_offset
        self._segments = []   # List of wire segments: [(x1,y1,z1, x2,y2,z2), ...]
        self._trace_2d = []   # 2D trace for DXF export: [[(x,y), (x,y), ...], ...]
        self._feed_segment_info = None  # (tag, segment_index, total_segments)

    def _clip_x_to_boundary(self, x, y, margin=1.0):
        """Clip an x coordinate to stay within the half-circle and this antenna's half."""
        bounds = halfcircle_x_at_y(y)
        if bounds is None:
            return 0.0
        x_min, x_max = bounds
        # Enforce left/right half: low band stays left (x < center_gap),
        # high band stays right (x > -center_gap)
        center_gap = 2.5  # mm from center to enforce 5mm gap
        if self.direction == 'left':
            x_max = min(x_max, -center_gap)
        else:
            x_min = max(x_min, center_gap)
        return np.clip(x, x_min + margin, x_max - margin)

    def _max_x_extent(self, y, margin=1.0):
        """Get the maximum horizontal extent at height y, restricted to this antenna's half."""
        bounds = halfcircle_x_at_y(y)
        if bounds is None:
            return (0, 0)
        x_left = bounds[0] + margin
        x_right = bounds[1] - margin
        center_gap = 2.5
        if self.direction == 'left':
            x_right = min(x_right, -center_gap - margin)
        else:
            x_left = max(x_left, center_gap + margin)
        return (x_left, x_right)

    def _arc_angular_bounds(self, r, margin=1.0):
        """Compute angular extent of an arc at radius r in this antenna's half.

        Returns (theta_min, theta_max) or None if the arc is too small.
        For left half:  theta_min = center gap side (~95°), theta_max = edge side (~170°)
        For right half: theta_min = edge side (~10°), theta_max = center gap side (~85°)

        Arcs stay above element_height + margin to avoid crossing the
        first horizontal run (at y=element_height), which would create
        mid-wire junctions that NEC2 cannot solve.
        """
        center_gap = 2.5  # mm from center
        y_margin = max(margin, self.element_height + margin)  # stay above horizontal run

        if r <= center_gap + 0.5 or r <= y_margin:
            return None

        sin_val = np.clip(y_margin / r, -1.0, 1.0)

        if self.direction == 'left':
            cos_val = np.clip(-center_gap / r, -1.0, 1.0)
            theta_min = np.arccos(cos_val)       # center gap side
            theta_max = np.pi - np.arcsin(sin_val)  # edge side (near y=margin)
        else:
            cos_val = np.clip(center_gap / r, -1.0, 1.0)
            theta_min = np.arcsin(sin_val)       # edge side (near y=margin)
            theta_max = np.arccos(cos_val)       # center gap side

        if theta_max <= theta_min + 0.01:
            return None
        return (theta_min, theta_max)

    def _discretize_arc(self, r, theta_start, theta_end, max_seg_length=4.0):
        """Convert arc to polyline points for NEC2 compatibility.

        Each segment respects the NEC2 thin-wire constraint:
        segment length >= 8 * wire_radius.
        """
        arc_length = abs(r * (theta_end - theta_start))
        if arc_length < 0.01:
            return [(r * np.cos(theta_start), r * np.sin(theta_start))]

        min_seg_length = max(8 * self.wire_radius + 0.05, 1.0)
        # Target number of segments for smooth arcs
        n_segs = max(1, int(np.ceil(arc_length / max_seg_length)))
        # Cap at maximum that keeps each segment above NEC2 minimum
        max_segs = max(1, int(arc_length / min_seg_length))
        n_segs = min(n_segs, max_segs)
        n_pts = n_segs + 1

        thetas = np.linspace(theta_start, theta_end, n_pts)
        points = [(r * np.cos(t), r * np.sin(t)) for t in thetas]
        return points

    def generate_geometry(self):
        """Generate the IFA wire geometry.

        Returns:
            segments: list of (x1,y1, x2,y2) in mm (2D, z=0 plane for top copper)
            feed_point: (x, y) of feed location
            short_point: (x, y) of short location
        """
        self._segments = []
        self._trace_2d = []
        margin = 1.0  # mm margin from half-circle edge

        z_top = SUBSTRATE_THICKNESS / 2.0  # top copper layer z
        z_bot = -SUBSTRATE_THICKNESS / 2.0  # bottom copper / ground z

        # --- Shorting stub: vertical from ground plane to element height ---
        short_seg = (self.short_x, 0, self.short_x, self.element_height)
        self._segments.append(short_seg)
        self._trace_2d.append([(self.short_x, 0), (self.short_x, self.element_height)])

        # --- Feed stub: vertical from ground plane to element height ---
        feed_seg = (self.feed_x, 0, self.feed_x, self.element_height)
        self._segments.append(feed_seg)
        self._trace_2d.append([(self.feed_x, 0), (self.feed_x, self.element_height)])

        # --- Horizontal connection from short to feed at element_height ---
        if self.direction == 'left':
            conn_seg = (self.short_x, self.element_height,
                        self.feed_x, self.element_height)
        else:
            conn_seg = (self.feed_x, self.element_height,
                        self.short_x, self.element_height)
        self._segments.append(conn_seg)
        self._trace_2d.append([
            (conn_seg[0], conn_seg[1]), (conn_seg[2], conn_seg[3])
        ])

        # --- Conformal arc meander radiating element ---
        # Concentric arcs following the half-circle boundary replace
        # horizontal meander runs, achieving much longer trace lengths.

        accumulated_length = abs(self.feed_offset)  # short-to-feed connection
        current_x = self.short_x
        current_y = self.element_height

        # First horizontal run from short position to boundary
        x_left, x_right = self._max_x_extent(current_y, margin)
        if self.direction == 'left':
            target_x = x_left
        else:
            target_x = x_right

        run_length = abs(target_x - current_x)
        remaining = self.total_length - accumulated_length
        if run_length > remaining:
            if self.direction == 'left':
                target_x = current_x - remaining
            else:
                target_x = current_x + remaining
            run_length = remaining

        if run_length > 0.5:
            self._segments.append((current_x, current_y, target_x, current_y))
            self._trace_2d.append([(current_x, current_y), (target_x, current_y)])
            accumulated_length += run_length
            current_x = target_x

        # Connect to first conformal arc and begin arc meander loop
        if accumulated_length < self.total_length - 0.5:
            first_arc_r = BOARD_RADIUS - margin
            endpoint_theta = np.arctan2(current_y, current_x)

            # Compute the actual arc start angle (clipped to angular bounds)
            first_bounds = self._arc_angular_bounds(first_arc_r, margin)
            if first_bounds is not None:
                actual_start_theta = np.clip(
                    endpoint_theta, first_bounds[0], first_bounds[1])
            else:
                actual_start_theta = endpoint_theta

            # Arc meander loop: alternating arcs at decreasing radii
            # Radial transitions are merged into the next arc's polyline
            # to avoid short segments that violate NEC2 thin-wire constraint.
            # The connection from horizontal run to the first arc is also merged
            # into the first arc's polyline (prevents short NEC2 segments).
            current_r = first_arc_r
            current_theta = actual_start_theta
            center_gap = 2.5
            prev_arc_end = (current_x, current_y)  # merge horizontal→arc connection
            min_nec_seg = 8 * self.wire_radius  # NEC2 minimum segment length

            max_arcs = 50  # safety limit
            for arc_idx in range(max_arcs):
                remaining = self.total_length - accumulated_length
                if remaining < 0.5:
                    break

                bounds = self._arc_angular_bounds(current_r, margin)
                if bounds is None:
                    break
                theta_min, theta_max = bounds

                # Even arcs: edge → gap; odd arcs: gap → edge
                # Left:  edge = theta_max, gap = theta_min
                # Right: edge = theta_min, gap = theta_max
                if self.direction == 'left':
                    sweep_end = theta_min if arc_idx % 2 == 0 else theta_max
                else:
                    sweep_end = theta_max if arc_idx % 2 == 0 else theta_min

                sweep_start = np.clip(current_theta, theta_min, theta_max)
                arc_angle = abs(sweep_end - sweep_start)
                if arc_angle < 0.01:
                    break

                arc_length = current_r * arc_angle

                # Account for transition distance from previous arc
                trans_dist = 0.0
                if prev_arc_end is not None:
                    arc_start_pt = (current_r * np.cos(sweep_start),
                                    current_r * np.sin(sweep_start))
                    trans_dist = np.sqrt(
                        (prev_arc_end[0] - arc_start_pt[0])**2 +
                        (prev_arc_end[1] - arc_start_pt[1])**2)

                # Truncate to remaining length if needed
                if trans_dist + arc_length > remaining:
                    arc_remaining = remaining - trans_dist
                    if arc_remaining < 0.5:
                        break
                    partial_angle = arc_remaining / current_r
                    if sweep_end > sweep_start:
                        sweep_end = sweep_start + partial_angle
                    else:
                        sweep_end = sweep_start - partial_angle

                # Discretize the arc
                points = self._discretize_arc(current_r, sweep_start, sweep_end)

                # Prepend transition point from previous arc end
                if prev_arc_end is not None:
                    p0 = points[0]
                    td = np.sqrt((prev_arc_end[0] - p0[0])**2 +
                                 (prev_arc_end[1] - p0[1])**2)
                    if td > 0.1:
                        points = [prev_arc_end] + points

                # Emit NEC2-safe segments (combine any that are too short)
                arc_actual = 0.0
                seg_start = 0
                for i in range(1, len(points)):
                    p1 = points[seg_start]
                    p2 = points[i]
                    seg_len = np.sqrt((p2[0] - p1[0])**2 +
                                      (p2[1] - p1[1])**2)
                    if seg_len >= min_nec_seg:
                        self._segments.append(
                            (p1[0], p1[1], p2[0], p2[1]))
                        seg_start = i
                    elif i == len(points) - 1 and seg_len > 0.01:
                        # Last segment too short: merge with previous
                        if self._segments and seg_start > 0:
                            prev = self._segments[-1]
                            self._segments[-1] = (
                                prev[0], prev[1], p2[0], p2[1])
                        else:
                            self._segments.append(
                                (p1[0], p1[1], p2[0], p2[1]))

                # Total polyline distance for length tracking
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    arc_actual += np.sqrt((p2[0] - p1[0])**2 +
                                          (p2[1] - p1[1])**2)
                self._trace_2d.append(points)
                accumulated_length += arc_actual

                if accumulated_length >= self.total_length - 0.5:
                    break

                # Prepare transition to next arc (merged into next iteration)
                next_r = current_r - self.meander_spacing
                if next_r < center_gap + margin:
                    break

                prev_arc_end = points[-1]
                current_r = next_r
                current_theta = sweep_end

        self._actual_length = accumulated_length
        self._feed_point = (self.feed_x, 0)
        self._short_point = (self.short_x, 0)

        return self._segments, self._feed_point, self._short_point

    @property
    def actual_length(self):
        return getattr(self, '_actual_length', 0)

    @property
    def feed_point(self):
        return getattr(self, '_feed_point', (0, 0))

    @property
    def short_point(self):
        return getattr(self, '_short_point', (0, 0))

    @property
    def trace_2d(self):
        return self._trace_2d

    def validate_geometry(self):
        """Check that all segments are within the half-circle boundary."""
        violations = []
        for seg in self._segments:
            x1, y1, x2, y2 = seg
            # Check endpoints
            for x, y in [(x1, y1), (x2, y2)]:
                if y >= 0 and not point_in_halfcircle(x, y, r=BOARD_RADIUS + 0.1):
                    violations.append((x, y))
        return violations


class BranchedIFA:
    """Single-feed branched IFA using the entire semicircle.

    Components:
    1. Shorting stub: (short_x, 0) -> (short_x, elem_height)
    2. Feed stub: (feed_x, 0) -> (feed_x, elem_height)
    3. Junction bar: horizontal at y=elem_height connecting short to feed
    4. LB arm: conformal arc meander across full semicircle width
    5. LB cap load: rectangular PEC pad at LB arm terminus (optional)
    6. HB branch: straight/curved trace extending from junction opposite to LB
    """

    def __init__(self, params=None):
        from config import (
            BRANCHED_SHORT_X, BRANCHED_FEED_OFFSET, BRANCHED_ELEM_HEIGHT,
            BRANCHED_LB_LENGTH, BRANCHED_LB_SPACING,
            BRANCHED_LB_CAP_W, BRANCHED_LB_CAP_L,
            BRANCHED_HB_LENGTH, BRANCHED_HB_ANGLE,
            BRANCHED_TRACE_WIDTH, OPENEMS_GND_CLEARANCE,
        )

        p = params or {}
        self.short_x = p.get('SHORT_X', BRANCHED_SHORT_X)
        self.feed_offset = p.get('FEED_OFFSET', BRANCHED_FEED_OFFSET)
        self.elem_height = p.get('ELEM_HEIGHT', BRANCHED_ELEM_HEIGHT)
        self.lb_length = p.get('LB_LENGTH', BRANCHED_LB_LENGTH)
        self.lb_spacing = p.get('LB_SPACING', BRANCHED_LB_SPACING)
        self.lb_cap_w = p.get('LB_CAP_W', BRANCHED_LB_CAP_W)
        self.lb_cap_l = p.get('LB_CAP_L', BRANCHED_LB_CAP_L)
        self.hb_length = p.get('HB_LENGTH', BRANCHED_HB_LENGTH)
        self.hb_angle = p.get('HB_ANGLE', BRANCHED_HB_ANGLE)
        self.trace_width = p.get('TRACE_WIDTH', BRANCHED_TRACE_WIDTH)
        self.gnd_clearance = p.get('GND_CLEARANCE', OPENEMS_GND_CLEARANCE)

        # Feed is to the right of short (LB arm extends left)
        self.feed_x = self.short_x + self.feed_offset

        self.wire_radius = trace_width_to_wire_radius(self.trace_width)
        self._segments = []
        self._lb_segments = []
        self._hb_segments = []
        self._trace_2d = []
        self._lb_trace_2d = []
        self._hb_trace_2d = []
        self._actual_lb_length = 0.0
        self._actual_hb_length = 0.0
        self._feed_point = (self.feed_x, 0)
        self._short_point = (self.short_x, 0)
        self.cap_load_rect = None  # (cx, cy, w, h) if cap load present

    def _add_lb_segment(self, seg):
        """Add a segment to overall, LB, and trace lists."""
        self._segments.append(seg)
        self._lb_segments.append(seg)
        self._trace_2d.append([(seg[0], seg[1]), (seg[2], seg[3])])
        self._lb_trace_2d.append([(seg[0], seg[1]), (seg[2], seg[3])])

    def _add_hb_segment(self, seg):
        """Add a segment to overall, HB, and trace lists."""
        self._segments.append(seg)
        self._hb_segments.append(seg)
        self._trace_2d.append([(seg[0], seg[1]), (seg[2], seg[3])])
        self._hb_trace_2d.append([(seg[0], seg[1]), (seg[2], seg[3])])

    def generate_geometry(self):
        """Generate branched IFA with left-side short/feed and full-span arms.

        Layout: Short and feed on the LEFT side of the semicircle.
        LB arm: straight horizontal extending RIGHT across the full diameter.
        HB arm: L-shaped branch off the LB arm going UP then RIGHT.

        This maximizes horizontal arm length for low-band resonance
        (IFA resonance depends on horizontal path length parallel to ground edge).

        Returns:
            segments: list of (x1,y1, x2,y2) in mm
            feed_point: (x, y) of feed location
            short_point: (x, y) of short location
        """
        self._segments = []
        self._lb_segments = []
        self._hb_segments = []
        self._trace_2d = []
        self._lb_trace_2d = []
        self._hb_trace_2d = []
        self.cap_load_rect = None
        margin = 1.0

        # --- Shorting stub ---
        short_seg = (self.short_x, 0, self.short_x, self.elem_height)
        self._segments.append(short_seg)
        self._trace_2d.append([(self.short_x, 0), (self.short_x, self.elem_height)])

        # --- Feed stub ---
        feed_seg = (self.feed_x, 0, self.feed_x, self.elem_height)
        self._segments.append(feed_seg)
        self._trace_2d.append([(self.feed_x, 0), (self.feed_x, self.elem_height)])

        # --- Junction bar: horizontal at elem_height from short to feed ---
        conn_seg = (self.short_x, self.elem_height,
                     self.feed_x, self.elem_height)
        self._segments.append(conn_seg)
        self._trace_2d.append([
            (self.short_x, self.elem_height),
            (self.feed_x, self.elem_height),
        ])

        # ============================
        # LB ARM: straight horizontal extending RIGHT from feed
        # ============================
        self._generate_lb_arm(margin)

        # ============================
        # HB ARM: L-shaped branch off the LB arm
        # ============================
        self._generate_hb_arm(margin)

        self._feed_point = (self.feed_x, 0)
        self._short_point = (self.short_x, 0)

        return self._segments, self._feed_point, self._short_point

    def _generate_lb_arm(self, margin=1.0):
        """Generate LB arm as a straight horizontal trace extending RIGHT.

        With the short/feed on the left side of the semicircle, the LB arm
        extends rightward across the full diameter for maximum horizontal
        length. IFA resonance depends on horizontal path length parallel
        to the ground edge.
        """
        start_x = self.feed_x
        start_y = self.elem_height

        bounds = halfcircle_x_at_y(start_y)
        if bounds is None:
            self._actual_lb_length = 0.0
            return

        x_right_max = bounds[1] - margin
        arm_length = min(self.lb_length, x_right_max - start_x)

        if arm_length < 1.0:
            self._actual_lb_length = 0.0
            return

        end_x = start_x + arm_length
        self._add_lb_segment((start_x, start_y, end_x, start_y))
        self._actual_lb_length = arm_length

    def _generate_hb_arm(self, margin=1.0):
        """Generate HB arm as an L-shaped branch off the LB arm.

        The HB arm branches off the LB arm at a point determined by
        hb_angle (used here as the branch offset from feed in mm).
        It goes UP by lb_spacing mm, then horizontally RIGHT toward
        the semicircle boundary. Going RIGHT (same direction as the LB arm)
        provides more horizontal room than going LEFT, which gets clipped
        by the semicircle at higher y values.

        Total path = vertical + horizontal = hb_length.
        """
        # Branch point along the LB arm
        branch_offset = self.hb_angle  # branch distance from feed along LB arm
        branch_x = self.feed_x + branch_offset
        branch_y = self.elem_height

        # Verify branch point is within semicircle
        R_eff = BOARD_RADIUS - margin
        if not point_in_halfcircle(branch_x, branch_y, r=R_eff):
            self._actual_hb_length = 0.0
            return

        # Vertical segment going UP
        vert_height = min(self.lb_spacing, self.hb_length)
        vert_top_y = branch_y + vert_height

        # Clip to semicircle
        while vert_top_y > branch_y + 0.5:
            if point_in_halfcircle(branch_x, vert_top_y, r=R_eff):
                break
            vert_top_y -= 0.5

        actual_vert = vert_top_y - branch_y
        if actual_vert < 0.5:
            self._actual_hb_length = 0.0
            return

        self._add_hb_segment((branch_x, branch_y, branch_x, vert_top_y))
        hb_accumulated = actual_vert

        # Horizontal segment going RIGHT (more room toward semicircle boundary)
        remaining = self.hb_length - hb_accumulated
        if remaining > 0.5:
            hb_end_x = branch_x + remaining
            bounds = halfcircle_x_at_y(vert_top_y)
            if bounds:
                hb_end_x = min(hb_end_x, bounds[1] - margin)
            horiz_len = hb_end_x - branch_x
            if horiz_len > 0.5:
                self._add_hb_segment(
                    (branch_x, vert_top_y, hb_end_x, vert_top_y))
                hb_accumulated += horiz_len

        self._actual_hb_length = hb_accumulated

    @property
    def actual_length(self):
        return self._actual_lb_length + self._actual_hb_length

    @property
    def actual_lb_length(self):
        return self._actual_lb_length

    @property
    def actual_hb_length(self):
        return self._actual_hb_length

    @property
    def feed_point(self):
        return self._feed_point

    @property
    def short_point(self):
        return self._short_point

    @property
    def trace_2d(self):
        return self._trace_2d

    @property
    def lb_trace_2d(self):
        return self._lb_trace_2d

    @property
    def hb_trace_2d(self):
        return self._hb_trace_2d

    def validate_geometry(self):
        """Check that all segments are within the half-circle boundary."""
        violations = []
        for seg in self._segments:
            x1, y1, x2, y2 = seg
            for x, y in [(x1, y1), (x2, y2)]:
                if y >= 0 and not point_in_halfcircle(x, y, r=BOARD_RADIUS + 0.1):
                    violations.append((x, y))
        return violations

    def check_gap(self):
        """Check minimum gap between LB and HB traces."""
        min_dist = float('inf')
        for lb_seg in self._lb_segments:
            lx1, ly1, lx2, ly2 = lb_seg
            for hb_seg in self._hb_segments:
                hx1, hy1, hx2, hy2 = hb_seg
                for lx, ly in [(lx1, ly1), (lx2, ly2), ((lx1+lx2)/2, (ly1+ly2)/2)]:
                    for hx, hy in [(hx1, hy1), (hx2, hy2), ((hx1+hx2)/2, (hy1+hy2)/2)]:
                        d = np.sqrt((lx - hx)**2 + (ly - hy)**2)
                        min_dist = min(min_dist, d)
        return min_dist


class DualBandIFA:
    """Manages both low-band and high-band IFA antennas."""

    def __init__(self, params=None):
        """
        Args:
            params: dict of parameter overrides. Keys match config parameter names.
        """
        from config import (
            LB_SHORT_POS_X, LB_FEED_OFFSET, LB_ELEMENT_HEIGHT,
            LB_TRACE_WIDTH, LB_TOTAL_LENGTH, LB_MEANDER_SPACING, LB_NUM_MEANDERS,
            HB_SHORT_POS_X, HB_FEED_OFFSET, HB_ELEMENT_HEIGHT,
            HB_TRACE_WIDTH, HB_TOTAL_LENGTH, HB_MEANDER_SPACING, HB_NUM_MEANDERS,
        )

        p = params or {}

        self.lowband = IFAAntenna(
            band='lowband',
            short_pos_x=p.get('LB_SHORT_POS_X', LB_SHORT_POS_X),
            feed_offset=p.get('LB_FEED_OFFSET', LB_FEED_OFFSET),
            element_height=p.get('LB_ELEMENT_HEIGHT', LB_ELEMENT_HEIGHT),
            trace_width=p.get('LB_TRACE_WIDTH', LB_TRACE_WIDTH),
            total_length=p.get('LB_TOTAL_LENGTH', LB_TOTAL_LENGTH),
            meander_spacing=p.get('LB_MEANDER_SPACING', LB_MEANDER_SPACING),
            num_meanders=p.get('LB_NUM_MEANDERS', LB_NUM_MEANDERS),
            direction='left',
        )

        self.highband = IFAAntenna(
            band='highband',
            short_pos_x=p.get('HB_SHORT_POS_X', HB_SHORT_POS_X),
            feed_offset=p.get('HB_FEED_OFFSET', HB_FEED_OFFSET),
            element_height=p.get('HB_ELEMENT_HEIGHT', HB_ELEMENT_HEIGHT),
            trace_width=p.get('HB_TRACE_WIDTH', HB_TRACE_WIDTH),
            total_length=p.get('HB_TOTAL_LENGTH', HB_TOTAL_LENGTH),
            meander_spacing=p.get('HB_MEANDER_SPACING', HB_MEANDER_SPACING),
            num_meanders=p.get('HB_NUM_MEANDERS', HB_NUM_MEANDERS),
            direction='right',
        )

    def generate(self, verbose=True):
        """Generate geometry for both antennas."""
        lb_segs, lb_feed, lb_short = self.lowband.generate_geometry()
        hb_segs, hb_feed, hb_short = self.highband.generate_geometry()

        # Validate
        lb_violations = self.lowband.validate_geometry()
        hb_violations = self.highband.validate_geometry()

        if lb_violations and verbose:
            print(f"WARNING: Low band has {len(lb_violations)} boundary violations")
            for v in lb_violations[:5]:
                print(f"  ({v[0]:.1f}, {v[1]:.1f})")

        if hb_violations and verbose:
            print(f"WARNING: High band has {len(hb_violations)} boundary violations")
            for v in hb_violations[:5]:
                print(f"  ({v[0]:.1f}, {v[1]:.1f})")

        # Check minimum gap between antennas
        min_gap = self._check_gap()
        if min_gap < 5.0 and verbose:
            print(f"WARNING: Minimum gap between antennas is {min_gap:.1f}mm (target: 5mm)")

        if verbose:
            print(f"Low band:  {self.lowband.actual_length:.1f}mm actual "
                  f"(target: {self.lowband.total_length:.1f}mm)")
            print(f"High band: {self.highband.actual_length:.1f}mm actual "
                  f"(target: {self.highband.total_length:.1f}mm)")

        return {
            'lowband': {'segments': lb_segs, 'feed': lb_feed, 'short': lb_short},
            'highband': {'segments': hb_segs, 'feed': hb_feed, 'short': hb_short},
        }

    def _check_gap(self):
        """Check minimum gap between low-band and high-band traces."""
        min_dist = float('inf')
        for lb_seg in self.lowband._segments:
            lx1, ly1, lx2, ly2 = lb_seg
            for hb_seg in self.highband._segments:
                hx1, hy1, hx2, hy2 = hb_seg
                # Approximate: check distance between segment midpoints
                lmx = (lx1 + lx2) / 2
                lmy = (ly1 + ly2) / 2
                hmx = (hx1 + hx2) / 2
                hmy = (hy1 + hy2) / 2
                d = np.sqrt((lmx - hmx)**2 + (lmy - hmy)**2)
                min_dist = min(min_dist, d)
                # Also check endpoints
                for lx, ly in [(lx1, ly1), (lx2, ly2)]:
                    for hx, hy in [(hx1, hy1), (hx2, hy2)]:
                        d = np.sqrt((lx - hx)**2 + (ly - hy)**2)
                        min_dist = min(min_dist, d)
        return min_dist


def build_nec_model(antenna, band='lowband', freq_mhz=900.0,
                    include_ground_plane=True):
    """Build a NEC2 model for one IFA antenna.

    Args:
        antenna: DualBandIFA instance (already generated)
        band: 'lowband' or 'highband'
        freq_mhz: frequency for segment sizing
        include_ground_plane: whether to add ground plane wires

    Returns:
        nec: NEC2 context ready for simulation
        feed_tag: tag number of the feed wire
        feed_seg: segment number for excitation
    """
    import necpp

    nec = necpp.nec_create()

    ifa = antenna.lowband if band == 'lowband' else antenna.highband
    freq_hz = freq_mhz * 1e6
    wire_radius_m = ifa.wire_radius * MM_TO_M

    tag_id = 1
    feed_tag = None
    feed_seg = None

    # --- Add antenna wire segments ---
    for seg in ifa._segments:
        x1_mm, y1_mm, x2_mm, y2_mm = seg
        length_mm = np.sqrt((x2_mm - x1_mm)**2 + (y2_mm - y1_mm)**2)
        if length_mm < 0.1:
            continue

        n_seg = segments_for_length(length_mm, freq_hz, min_seg=3)

        # NEC2 coordinates: x,y are horizontal, z is vertical
        # Map our 2D (x, y) to NEC2 (x, 0, z) where z = y (height above ground)
        necpp.nec_wire(nec, tag_id, n_seg,
                       x1_mm * MM_TO_M, 0, y1_mm * MM_TO_M,
                       x2_mm * MM_TO_M, 0, y2_mm * MM_TO_M,
                       wire_radius_m, 1.0, 1.0)

        # Identify the feed wire (the feed stub from ground to element height)
        if (abs(x1_mm - ifa.feed_x) < 0.01 and abs(y1_mm) < 0.01 and
                abs(x2_mm - ifa.feed_x) < 0.01 and y2_mm > 0):
            feed_tag = tag_id
            feed_seg = 1  # excite at the bottom (ground end) of the feed stub

        tag_id += 1

    # --- Add ground plane wires ---
    if include_ground_plane:
        from config import GND_EXTENSION, BOARD_RADIUS, NEC_GND_SEGMENTS

        gnd_wire_radius = 0.5 * MM_TO_M  # thicker wire for ground plane
        gnd_y = 0  # ground plane is at y=0 (our z=0 in NEC)

        # Approximate ground plane as a grid of wires
        # Horizontal wires along the flat edge and below
        gnd_spacing = 5.0  # mm between ground wires
        n_gnd_wires = int(GND_EXTENSION / gnd_spacing)

        for i in range(n_gnd_wires + 1):
            gy = -i * gnd_spacing  # below the flat edge
            x_extent = BOARD_RADIUS  # full width along the flat edge

            necpp.nec_wire(nec, tag_id, NEC_GND_SEGMENTS,
                           -x_extent * MM_TO_M, 0, gy * MM_TO_M,
                           x_extent * MM_TO_M, 0, gy * MM_TO_M,
                           gnd_wire_radius, 1.0, 1.0)
            tag_id += 1

        # Vertical wires connecting ground plane
        n_vert = 7  # number of vertical wires across width
        for i in range(n_vert):
            gx = -BOARD_RADIUS + i * (2 * BOARD_RADIUS / (n_vert - 1))
            necpp.nec_wire(nec, tag_id, NEC_GND_SEGMENTS,
                           gx * MM_TO_M, 0, 0,
                           gx * MM_TO_M, 0, -GND_EXTENSION * MM_TO_M,
                           gnd_wire_radius, 1.0, 1.0)
            tag_id += 1

    if feed_tag is None:
        print("WARNING: Feed wire not identified! Using tag 2, segment 1")
        feed_tag = 2
        feed_seg = 1

    # Geometry complete - no ground plane symmetry
    necpp.nec_geometry_complete(nec, 0)

    return nec, feed_tag, feed_seg


if __name__ == '__main__':
    from config import ANTENNA_TOPOLOGY

    if ANTENNA_TOPOLOGY == 'branched':
        from utils import plot_antenna_geometry_branched

        ant = BranchedIFA()
        ant.generate_geometry()

        violations = ant.validate_geometry()
        if violations:
            print(f"WARNING: {len(violations)} boundary violations")
        gap = ant.check_gap()
        print(f"LB arm: {ant.actual_lb_length:.1f}mm")
        print(f"HB branch: {ant.actual_hb_length:.1f}mm")
        print(f"Min LB-HB gap: {gap:.1f}mm")

        plot_antenna_geometry_branched(
            ant.lb_trace_2d,
            ant.hb_trace_2d,
            cap_rect=ant.cap_load_rect,
            save_path='results/antenna_layout.png'
        )
        print("\nGeometry generated. See results/antenna_layout.png")
    else:
        from utils import plot_antenna_geometry

        dual = DualBandIFA()
        geom = dual.generate()

        plot_antenna_geometry(
            dual.lowband.trace_2d,
            dual.highband.trace_2d,
            save_path='results/antenna_layout.png'
        )
        print("\nGeometry generated. See results/antenna_layout.png")
