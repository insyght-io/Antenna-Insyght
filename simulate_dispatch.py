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
Dispatch layer for simulation backends.

Routes simulation calls to NEC2 or openEMS based on config.SIM_BACKEND
or an explicit override. This keeps the optimizer and report code
backend-agnostic.

Also provides branched IFA dispatch functions.
"""

from config import SIM_BACKEND


def get_backend(override=None):
    """Return the simulation module for the given backend.

    Args:
        override: 'nec2', 'openems', or None (use config default)

    Returns:
        module with simulate_s11, simulate_radiation, etc.
    """
    backend = override or SIM_BACKEND
    if backend == 'openems':
        import simulate_openems
        return simulate_openems
    elif backend == 'nec2':
        import simulate
        return simulate
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'nec2' or 'openems'.")


def simulate_s11(antenna, band, freqs=None, verbose=True, backend=None):
    """Run S11 sweep using the selected backend."""
    mod = get_backend(backend)
    return mod.simulate_s11(antenna, band, freqs=freqs, verbose=verbose)


def simulate_radiation(antenna, band, freq_hz=None, backend=None):
    """Run radiation pattern simulation using the selected backend."""
    mod = get_backend(backend)
    return mod.simulate_radiation(antenna, band, freq_hz=freq_hz)


def run_full_simulation(params=None, output_dir='results', quick=False, backend=None):
    """Run complete simulation for both bands using the selected backend."""
    mod = get_backend(backend)
    return mod.run_full_simulation(params=params, output_dir=output_dir, quick=quick)


def validate_geometry(antenna, band, backend=None):
    """Validate geometry for the selected backend. Returns (ok, reason)."""
    b = (backend or SIM_BACKEND)
    if b == 'openems':
        import simulate_openems
        return simulate_openems.validate_geometry_for_openems(antenna, band)
    else:
        import simulate
        return simulate.validate_geometry_for_nec(antenna, band)


def simulate_s11_safe(params_dict, band, freqs, timeout=None, backend=None):
    """Subprocess-safe S11 simulation using the selected backend."""
    b = (backend or SIM_BACKEND)
    if b == 'openems':
        import simulate_openems
        t = timeout if timeout is not None else 120
        return simulate_openems.simulate_s11_safe(params_dict, band, freqs, timeout=t)
    else:
        import simulate
        t = timeout if timeout is not None else 30
        return simulate.simulate_s11_safe(params_dict, band, freqs, timeout=t)


# --- Branched IFA dispatch ---

def simulate_s11_branched(antenna, freqs=None, verbose=True, fast=False,
                           explore=False, gnd_clearance=None):
    """Run S11 sweep for branched IFA (openEMS only)."""
    import simulate_openems
    return simulate_openems.simulate_s11_branched(
        antenna, freqs=freqs, verbose=verbose, fast=fast,
        explore=explore, gnd_clearance=gnd_clearance)


def validate_geometry_branched(antenna):
    """Validate branched IFA geometry. Returns (ok, reason)."""
    import simulate_openems
    return simulate_openems.validate_geometry_for_branched(antenna)


def run_full_simulation_branched(params=None, output_dir='results', quick=False):
    """Run complete branched IFA simulation."""
    import simulate_openems
    return simulate_openems.run_full_simulation_branched(
        params=params, output_dir=output_dir, quick=quick)
