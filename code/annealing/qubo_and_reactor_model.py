#!/usr/bin/env python3
"""Ares SA/SQA pattern-generation and OpenMC pipeline.

Generated from Paper_Sample_Constrained(1).ipynb for the SA/SQA benchmark.
The production physics settings remain untouched in build_pwr_core. Runtime
commands may explicitly override them for a reduced dry run.
"""

from __future__ import annotations

import os

for _name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ[_name] = '1'

import argparse
import concurrent.futures
import json
import math
import shutil
import socket
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import dimod
import numpy as np
import openjij as oj
import openmc


"""
=============================================================================
  PWR CORE MODEL — OpenMC  (FULLY 2D, assembly-level peaking, water reflector)
  15×15 core lattice — 193 fuel assemblies + 32 water positions

  Assembly design: Westinghouse 17×17 (dimensions from design table)
    • 264 fuel rods   — UO2 pellet / He gap / Zr-4 clad / coolant
    •  24 guide thimbles — annular Zr-4 tube, water-filled
    •   1 instrument tube — centre (8,8)

# All pin‑level dimensions from BEAVRS 2.0.2 specification (inches → cm):
#   Fuel pellet diameter  : 0.3088 in → r = 0.39218 cm   (Source 8)
#   Clad inner radius     : 0.40005 cm   (derived from 0.360 in OD minus 2×0.0225 in clad thickness)
#   Fuel rod OD           : 0.360  in → clad outer r = 0.45720 cm   (Source 10)
#   Guide thimble OD      : 0.474  in → r_out = 0.60198 cm   (Source 14)
#   Guide thimble ID      : 0.442  in → r_in  = 0.56134 cm   (Source 13)
#   Assembly pitch        : 8.466  in → 21.50364 cm   (Source 30)
#   Pin pitch             : 0.496  in → 1.25984 cm   (Source 31)
=============================================================================
"""

import os
import time
import openmc
import numpy as np
import json
import tempfile
import subprocess
import concurrent.futures
from tqdm import tqdm
from itertools import groupby


# =============================================================================
# CORE GEOMETRY HELPERS
# (mirrored from Section 3 so build_pwr_core is self-contained)
# =============================================================================

def _find_grid_side(N):
    """
    Smallest square grid side >= ceil(sqrt(N)) that minimises 4-fold
    symmetry violations.  For N=193: base=14, 193%4=1 → odd side → 15.
    """
    base      = int(np.ceil(np.sqrt(N)))
    r         = N % 4
    need_even = r in (0, 3)
    need_odd  = r in (1, 2)
    side      = base
    if need_even and side % 2 != 0:
        side += 1
    elif need_odd and side % 2 != 1:
        side += 1
    return side


def _generate_fuel_positions(N=193):
    """
    Select the N innermost positions of a square grid in a circular layout.

    Returns
    -------
    positions   : list of (col, row) tuples, length N, stable ordering
    position_set: set of the same tuples (for O(1) lookup)
    """
    side       = _find_grid_side(N)
    all_points = [(x, y) for x in range(side) for y in range(side)]
    center     = (side - 1) / 2.0
    dists      = {(x, y): round(np.hypot(x - center, y - center), 4)
                  for x, y in all_points}

    def sort_key(p):
        x, y   = p
        rx, ry = abs(x - center), abs(y - center)
        return (dists[p], -min(rx, ry), max(rx, ry), x, y)

    sorted_points = sorted(all_points, key=sort_key)

    from itertools import groupby as _groupby
    rings = [list(g) for _, g in _groupby(sorted_points, key=lambda p: dists[p])]

    selected     = []
    selected_set = set()
    cumulative   = 0

    for ring in rings:
        if cumulative + len(ring) <= N:
            selected.extend(ring)
            selected_set.update(ring)
            cumulative += len(ring)
        else:
            need     = N - cumulative
            ring_set = set(ring)
            used     = set()
            taken    = []
            for p in ring:
                if len(taken) >= need or p in used:
                    continue
                x, y   = p
                rx, ry = x - center, y - center
                copies = set()
                for sx in [rx, -rx]:
                    for sy in [ry, -ry]:
                        cp = (round(center + sx), round(center + sy))
                        if cp in ring_set and cp not in selected_set:
                            copies.add(cp)
                if copies and len(taken) + len(copies) <= need:
                    taken.extend(copies)
                    used.update(copies)
            for p in ring:
                if len(taken) >= need:
                    break
                if p not in used and p not in selected_set:
                    taken.append(p)
                    used.add(p)
            selected.extend(taken[:need])
            selected_set.update(taken[:need])
            break

    positions = selected[:N]
    return positions, set(positions)


# Pre-compute once — used by every call to build_pwr_core
N_FUEL                  = 193
GRID_SIDE               = 15
_FUEL_LIST, _FUEL_SET   = _generate_fuel_positions(N_FUEL)
_IDX_TO_GRID            = {i: p for i, p in enumerate(_FUEL_LIST)}   # idx → (col, row)
_GRID_TO_IDX            = {p: i for i, p in enumerate(_FUEL_LIST)}   # (col, row) → idx


# =============================================================================
# CORE MODEL BUILDER
# =============================================================================

def build_pwr_core(loading_pattern):
    """
    Build a 2D PWR core OpenMC model.

    Core layout
    -----------
    15×15 assembly grid with 193 fuel assemblies arranged in a circular
    footprint (matching the Westinghouse 4-loop PWR core count).  The 32
    grid positions outside the circular selection are filled with pure-water
    assemblies.  A 30 cm water reflector surrounds the core on all four
    sides.  Z boundaries are reflective (true 2D — no axial leakage).

    Assembly design  (Westinghouse 17×17, FSAR design table dimensions)
    -------------------------------------------------------------------
    Each fuel assembly contains a 17×17 array of pin positions:
      • 264 fuel rods        — 4-layer pin: UO2 pellet | He gas gap |
                               Zr-4 cladding | water coolant
      •  24 guide thimbles   — annular Zr-4 tube, water-filled
      •   1 instrument tube  — centre position (8,8), same tube as GT

    Parameters
    ----------
    loading_pattern : array-like
        Either a flat array of length 193 (fuel type per fuel position)
        or a (15,15) integer array (grid[row][col] = fuel type, 0 = water).
        Fuel types:
          1 = fresh UO2       3.1 wt% U-235
          2 = once-burnt UO2  2.4 wt% U-235  (burnup proxy)
          3 = twice-burnt UO2 1.6 wt% U-235  (burnup proxy)

    Returns
    -------
    openmc.Model
    """

    # -------------------------------------------------------------------------
    # Normalise loading_pattern to a (15×15) integer grid
    # -------------------------------------------------------------------------
    lp = np.asarray(loading_pattern)
    if lp.ndim == 1:
        assert len(lp) == N_FUEL, \
            f"Expected flat chromosome of length {N_FUEL}, got {len(lp)}"
        grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)   # 0 = water
        for fuel_idx, ftype in enumerate(lp):
            col, row        = _IDX_TO_GRID[fuel_idx]
            grid[row, col]  = int(ftype)
    else:
        assert lp.shape == (GRID_SIDE, GRID_SIDE), \
            f"Expected shape ({GRID_SIDE},{GRID_SIDE}), got {lp.shape}"
        grid = lp.astype(int)

    nx, ny = GRID_SIDE, GRID_SIDE   # both 15


    # =============================================================================
    # MATERIALS – BEAVRS 2.0.2 (Cycle 1 Hot Zero Power)
    # All number densities (atoms/barn·cm) from benchmark tables.
    # =============================================================================
    # ── Fuel 1.6% enriched (Table 4) ────────────────────────────────────────────
    fuel_1 = openmc.Material(name="UO2 1.6%")
    fuel_1.set_density('g/cm3', 10.31341)
    fuel_1.add_nuclide('O16', 4.5897e-02)
    fuel_1.add_nuclide('O17', 1.09468e-04)   
    fuel_1.add_nuclide('U234', 3.0131e-06)
    fuel_1.add_nuclide('U235', 3.7503e-04)
    fuel_1.add_nuclide('U238', 2.2625e-02)

    # ── Fuel 2.4% enriched (Table 5) ────────────────────────────────────────────
    fuel_2 = openmc.Material(name="UO2 2.4%")
    fuel_2.set_density('g/cm3', 10.29748)
    fuel_2.add_nuclide('O16', 4.5830e-02)
    fuel_2.add_nuclide('O17', 1.09309e-04)  
    fuel_2.add_nuclide('U234', 4.4842e-06)
    fuel_2.add_nuclide('U235', 5.5814e-04)
    fuel_2.add_nuclide('U238', 2.2407e-02)

    # ── Fuel 3.1% enriched (Table 6) ────────────────────────────────────────────
    fuel_3 = openmc.Material(name="UO2 3.1%")
    fuel_3.set_density('g/cm3', 10.30166)
    fuel_3.add_nuclide('O16', 4.5853e-02)
    fuel_3.add_nuclide('O17', 1.09362e-04)   
    fuel_3.add_nuclide('U234', 5.7987e-06)
    fuel_3.add_nuclide('U235', 7.2175e-04)
    fuel_3.add_nuclide('U238', 2.2253e-02)

    # ── Zircaloy‑4 (Table 16) ───────────────────────────────────────────────────
    zircaloy = openmc.Material(name="Zircaloy-4")
    zircaloy.set_density('g/cm3', 6.55)
    zircaloy.add_nuclide('Zr90',  2.1828e-02)
    zircaloy.add_nuclide('Zr91',  4.7601e-03)
    zircaloy.add_nuclide('Zr92',  7.2759e-03)
    zircaloy.add_nuclide('Zr94',  7.3734e-03)
    zircaloy.add_nuclide('Zr96',  1.1879e-03)
    zircaloy.add_nuclide('Sn112', 4.6735e-06)
    zircaloy.add_nuclide('Sn114', 3.1799e-06)
    zircaloy.add_nuclide('Sn115', 1.6381e-06)
    zircaloy.add_nuclide('Sn116', 7.0055e-05)
    zircaloy.add_nuclide('Sn117', 3.7003e-05)
    zircaloy.add_nuclide('Sn118', 1.1669e-04)
    zircaloy.add_nuclide('Sn119', 4.1387e-05)
    zircaloy.add_nuclide('Sn120', 1.5697e-04)
    zircaloy.add_nuclide('Sn122', 2.2308e-05)
    zircaloy.add_nuclide('Sn124', 2.7897e-05)
    zircaloy.add_nuclide('Fe54',  8.6698e-06)
    zircaloy.add_nuclide('Fe56',  1.3610e-04)
    zircaloy.add_nuclide('Fe57',  3.1431e-06)
    zircaloy.add_nuclide('Fe58',  4.1829e-07)
    zircaloy.add_nuclide('Cr50',  3.2962e-06)
    zircaloy.add_nuclide('Cr52',  6.3564e-05)
    zircaloy.add_nuclide('Cr53',  7.2076e-06)
    zircaloy.add_nuclide('Cr54',  1.7941e-06)
    zircaloy.add_nuclide('O16',   3.0744e-04)
    zircaloy.add_nuclide('O17',   7.3328e-07)   

    # ── Helium gas gap (Table 13) ───────────────────────────────────────────────
    helium = openmc.Material(name="Helium gap")
    helium.set_density('g/cm3', 0.0015981)
    helium.add_nuclide('He3', 4.8089e-10)
    helium.add_nuclide('He4', 2.4044e-04)

    # ── Borated water (core coolant, Table 17) ──────────────────────────────────
    water = openmc.Material(name="Coolant")
    water.set_density('g/cm3', 0.740582068)
    water.add_nuclide('H1',  4.9456e-02)
    water.add_nuclide('H2',  7.7035e-06)
    water.add_nuclide('O16', 2.4673e-02)
    water.add_nuclide('O17', 5.88474e-05)   
    water.add_nuclide('B10', 7.9714e-06)
    water.add_nuclide('B11', 3.2247e-05)
    water.add_s_alpha_beta('c_H_in_H2O')

    # ── Reflector water (same composition) ──────────────────────────────────────
    refl_water = openmc.Material(name="Reflector")
    refl_water.set_density('g/cm3', 0.740582068)
    refl_water.add_nuclide('H1',  4.9456e-02)
    refl_water.add_nuclide('H2',  7.7035e-06)
    refl_water.add_nuclide('O16', 2.4673e-02)
    refl_water.add_nuclide('O17', 5.88474e-05)  
    refl_water.add_nuclide('B10', 7.9714e-06)
    refl_water.add_nuclide('B11', 3.2247e-05)
    refl_water.add_s_alpha_beta('c_H_in_H2O')

    # ── Air (BEAVRS Table 9) ───────────────────────────────────────────────────
    air = openmc.Material(name="Air BEAVRS")
    air.set_density('g/cm3', 0.00616)
    air.add_nuclide('N14',  1.9680e-04)
    air.add_nuclide('N15',  7.2354e-07)
    air.add_nuclide('O16',  5.2866e-05)
    air.add_nuclide('O17',  1.26094e-07)   
    air.add_nuclide('Ar36', 7.8730e-09)
    air.add_nuclide('Ar38', 1.4844e-09)
    air.add_nuclide('Ar40', 2.3506e-06)
    air.add_nuclide('C0',  6.8295e-08)

    # ── Temperatures ─────────────────────────────────────────────────────────
    fuel_1.temperature       = 900   # K — fuel pellet (HFP centreline average)
    fuel_2.temperature       = 900
    fuel_3.temperature       = 900
    zircaloy.temperature     = 600   # K — cladding / structural
    helium.temperature       = 600   # K — gap midpoint
    water.temperature        = 600   # K — coolant (HFP bulk)
    refl_water.temperature   = 600
    air.temperature          = 600   # K — instrument tube bore


    mats = openmc.Materials([fuel_1, fuel_2, fuel_3, zircaloy, helium, water, refl_water, air])

    # =========================================================================
    # 2.  GEOMETRY PARAMETERS
    #     All pin-level dimensions from the design table (inches → cm).
    # =========================================================================
    # BEAVRS 2.0.2 dimensions (all in cm)
    IN = 2.54  # conversion factor (inches to cm)

    # Fuel pin radii
    fuel_r      = (0.3088 / 2) * IN      # 0.39218 cm
    gap_r       = (0.3600 / 2 - 0.0225) * IN   # 0.40005 cm  (clad inner radius)
    clad_r_out  = (0.3600 / 2) * IN      # 0.45720 cm

    # Guide tube radii (above dashpot)
    gt_r_in     = (0.442 / 2) * IN       # 0.56134 cm
    gt_r_out    = (0.474 / 2) * IN       # 0.60198 cm

    # ── Instrument tube inner thimble radii — BEAVRS Figure 6 ───────────────
    it_thimble_ir = 0.43688   # cm — IT inner bore outer surface (Air/Zr-4 interface)
    it_thimble_or = 0.48387   # cm — IT inner thimble outer radius (Zr-4/Water interface)

    # Assembly and pin pitch
    assy_pitch  = 8.466 * IN             # 21.50364 cm
    pin_pitch   = 0.496 * IN             # 1.25984 cm

    # ── Assembly and pin pitch ────────────────────────────────────────────────
    n_pins      = 17                     # 17×17 pin array per assembly
    

    # ── Reflector and axial extent ────────────────────────────────────────────
    refl_thick  = 30.0    # cm — water reflector radial thickness
    z_half      = 100.0   # cm — slab half-height; reflective BC → true 2D

    # ── Core half-widths (15 assemblies × assembly pitch / 2) ────────────────
    half_core_x = assy_pitch * nx / 2.0   # 161.277 cm
    half_core_y = assy_pitch * ny / 2.0   # 161.277 cm
    half_refl_x = half_core_x + refl_thick
    half_refl_y = half_core_y + refl_thick

    # ── Non-fuel positions in the 17×17 assembly (0-indexed row, col) ─────────
    #
    # Standard Westinghouse 17×17 layout — verified against 264 fuel pin count:
    #
    #   col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    # row 2:  .  .  G  .  .  G  .  .  G  .  .  G  .  .  G  .  .
    # row 5:  .  .  G  .  .  G  .  .  G  .  .  G  .  .  G  .  .
    # row 8:  .  .  G  .  .  G  .  .  I  .  .  G  .  .  G  .  .   ← I = instrument
    # row11:  .  .  G  .  .  G  .  .  G  .  .  G  .  .  G  .  .
    # row14:  .  .  G  .  .  G  .  .  G  .  .  G  .  .  G  .  .
    #                                                               G = guide thimble
    GT_POSITIONS = {
        (2,  2), (2,  5), (2,  8), (2, 11), (2, 14),
        (5,  2), (5,  5), (5,  8), (5, 11), (5, 14),
        (8,  2), (8,  5),          (8, 11), (8, 14),   # (8,8) = IT
        (11, 2), (11, 5), (11, 8), (11,11), (11,14),
        (14, 2), (14, 5), (14, 8), (14,11), (14,14),
    }                          # 24 guide thimbles
    IT_POSITION = (8, 8)       # 1 instrument tube at centre

    # =========================================================================
    # 3.  UNIVERSE DEFINITIONS
    # =========================================================================

    # Infinite water universe — fills lattice exterior and inside water tubes
    water_univ = openmc.Universe(cells=[openmc.Cell(fill=water)])

    # ── Fuel assembly universe builder ────────────────────────────────────────
    def make_assembly_univ(fuel_mat):
        """
        Build a 17×17 pin-lattice universe for a given fuel material.

        Fuel pin (4 radial zones, inside → outside):
          1. UO2 pellet      r < fuel_r
          2. Helium gas gap  fuel_r < r < gap_r
          3. Zr-4 cladding   gap_r  < r < clad_r_out
          4. Water coolant   r > clad_r_out

        Guide thimble / instrument tube (3 radial zones):
          1. Water (flow channel)   r < gt_r_in
          2. Zr-4 tube wall         gt_r_in < r < gt_r_out
          3. Water coolant          r > gt_r_out
        """

        # ── Fuel pin ─────────────────────────────────────────────────────────
        pellet_surf = openmc.ZCylinder(r=fuel_r)       # UO2 outer surface
        gap_surf    = openmc.ZCylinder(r=gap_r)        # He gap outer = clad inner
        clad_surf   = openmc.ZCylinder(r=clad_r_out)   # cladding outer surface

        fuel_pin_univ = openmc.Universe()
        fuel_pin_univ.add_cells([
            openmc.Cell(fill=fuel_mat, region=-pellet_surf),                   # UO2 pellet
            openmc.Cell(fill=helium,   region=+pellet_surf & -gap_surf),       # He gas gap
            openmc.Cell(fill=zircaloy, region=+gap_surf    & -clad_surf),      # Zr-4 clad
            openmc.Cell(fill=water,    region=+clad_surf),                     # coolant
        ])

        # ── Guide thimble ─────────────────────────────────────────────────────
        gt_inner_surf = openmc.ZCylinder(r=gt_r_in)
        gt_outer_surf = openmc.ZCylinder(r=gt_r_out)

        gt_univ = openmc.Universe()
        gt_univ.add_cells([
            openmc.Cell(fill=water,     region=-gt_inner_surf),                    # water inside
            openmc.Cell(fill=zircaloy,  region=+gt_inner_surf & -gt_outer_surf),   # Zr-4 wall
            openmc.Cell(fill=water,     region=+gt_outer_surf),                    # coolant outside
        ])

        # ── Instrument tube ───────────────────────────────────────────────────
        it_bore_surf    = openmc.ZCylinder(r=it_thimble_ir)  # Air / Zr-4 interface
        it_thimble_surf = openmc.ZCylinder(r=it_thimble_or)  # Zr-4 / Water interface
        # Outer shell reuses gt_inner_surf / gt_outer_surf already defined above
        it_univ = openmc.Universe()
        it_univ.add_cells([
                openmc.Cell(fill=air,   region=-it_bore_surf),                           # Air bore
                openmc.Cell(fill=zircaloy,  region=+it_bore_surf    & -it_thimble_surf), # Zr-4 thimble
                openmc.Cell(fill=water, region=+it_thimble_surf & -gt_inner_surf),       # Water annulus
                openmc.Cell(fill=zircaloy,  region=+gt_inner_surf   & -gt_outer_surf),   # Outer Zr-4 shell
                openmc.Cell(fill=water, region=+gt_outer_surf),                          # Coolant
            ])

        # ── Pin lattice ───────────────────────────────────────────────────────
        pin_lat            = openmc.RectLattice()
        pin_lat.lower_left = (
            -pin_pitch * n_pins / 2.0,
            -pin_pitch * n_pins / 2.0,
        )
        pin_lat.pitch = (pin_pitch, pin_pitch)

        universes = np.empty((n_pins, n_pins), dtype=object)
        for i in range(n_pins):
            for j in range(n_pins):
                if (i, j) == IT_POSITION:
                    universes[i, j] = it_univ
                elif (i, j) in GT_POSITIONS:
                    universes[i, j] = gt_univ
                else:
                    universes[i, j] = fuel_pin_univ
        pin_lat.universes = universes
        pin_lat.outer     = water_univ

        return openmc.Universe(cells=[openmc.Cell(fill=pin_lat)])
    
    type_to_univ = {
        1: make_assembly_univ(fuel_3),   # fresh
        2: make_assembly_univ(fuel_2),   # once-burnt
        3: make_assembly_univ(fuel_1),   # twice-burnt
    }

    # =========================================================================
    # 4.  CORE LATTICE
    #     15×15 grid; 193 fuel positions (circular layout) +
    #     32 water positions (square corners).
    # =========================================================================

    core_lat            = openmc.RectLattice()
    core_lat.lower_left = (-half_core_x, -half_core_y)
    core_lat.pitch      = (assy_pitch, assy_pitch)

    core_universes = np.empty((nx, ny), dtype=object)
    for row in range(nx):
        for col in range(ny):
            ftype = grid[row, col]   # 0 = water corner, 1/2/3 = fuel type
            core_universes[row, col] = (
                type_to_univ[ftype] if ftype in (1, 2, 3) else water_univ
            )
    core_lat.universes = core_universes
    core_lat.outer     = water_univ

    # =========================================================================
    # 5.  ROOT GEOMETRY — 2D slab with water reflector
    #     Z : reflective boundaries  → infinite-height (true 2D)
    #     XY: vacuum at reflector outer edge
    # =========================================================================

    z0 = openmc.ZPlane(z0=-z_half, boundary_type='reflective')
    z1 = openmc.ZPlane(z0= z_half, boundary_type='reflective')

    # Core inner boundary (no boundary_type — internal surfaces)
    cx0 = openmc.XPlane(x0=-half_core_x)
    cx1 = openmc.XPlane(x0= half_core_x)
    cy0 = openmc.YPlane(y0=-half_core_y)
    cy1 = openmc.YPlane(y0= half_core_y)

    # Reflector outer boundary (vacuum)
    rx0 = openmc.XPlane(x0=-half_refl_x, boundary_type='vacuum')
    rx1 = openmc.XPlane(x0= half_refl_x, boundary_type='vacuum')
    ry0 = openmc.YPlane(y0=-half_refl_y, boundary_type='vacuum')
    ry1 = openmc.YPlane(y0= half_refl_y, boundary_type='vacuum')

    core_region = +cx0 & -cx1 & +cy0 & -cy1 & +z0 & -z1
    refl_region = +rx0 & -rx1 & +ry0 & -ry1 & +z0 & -z1 & ~core_region

    root_univ = openmc.Universe(cells=[
        openmc.Cell(fill=core_lat,   region=core_region),
        openmc.Cell(fill=refl_water, region=refl_region),
    ])
    geom = openmc.Geometry(root_univ)

    # =========================================================================
    # 6.  SETTINGS
    # =========================================================================

    sets            = openmc.Settings()
    sets.batches    = 500
    sets.inactive   = 50
    sets.particles  = 500000
    sets.run_mode   = 'eigenvalue'

    # Fission source seeded uniformly over the core volume (not reflector)
    source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-half_core_x, -half_core_y, -z_half),
            ( half_core_x,  half_core_y,  z_half),
        )
    )
    source.constraints = {'fissionable': True}
    sets.source = [source]

    # =========================================================================
    # 7.  TALLIES — Pin-level mesh (255×255) over the core
    #     255 = 17 pins/assembly × 15 assemblies per axis
    #     Scores: fission-q-recoverable → pin power distribution
    # =========================================================================

    mesh             = openmc.RegularMesh()
    mesh.dimension   = [255, 255, 1]                          # pin-level resolution
    mesh.lower_left  = [-half_core_x, -half_core_y, -z_half]
    mesh.upper_right = [ half_core_x,  half_core_y,  z_half]

    power_tally         = openmc.Tally(name="power")
    power_tally.filters = [openmc.MeshFilter(mesh)]
    power_tally.scores  = ['fission-q-recoverable']

    tallies = openmc.Tallies([power_tally])

    return openmc.Model(geom, mats, sets, tallies)

# =============================================================================
# WESTINGHOUSE_193 CONFIGURATION — matched to the OpenMC notebook geometry
# =============================================================================

from collections import Counter, defaultdict

CORE_NAME = 'Westinghouse_193'
N_FUEL    = 193
N_FRESH   = 65
N_ONCE    = 64
N_TWICE   = 64

assert N_FRESH + N_ONCE + N_TWICE == N_FUEL
assert len(_FUEL_LIST) == N_FUEL


def compute_layers(adjacency, n_cells):
    """Return topological ring indices, with 0 representing the outer ring."""
    degree = {i: len(adjacency[i]) for i in range(n_cells)}
    layer = [-1] * n_cells
    remaining = set(range(n_cells))
    current_layer = 0

    while remaining:
        max_degree = max(degree[i] for i in remaining)
        boundary = [i for i in remaining if degree[i] < max_degree]

        if not boundary:
            for i in remaining:
                layer[i] = current_layer
            break

        for i in boundary:
            layer[i] = current_layer
            remaining.remove(i)
            for j in adjacency[i]:
                if j in remaining:
                    degree[j] -= 1

        current_layer += 1

    return layer


def get_westinghouse_config():
    """Build the solver configuration using the OpenMC model's exact ordering."""
    positions = {i: tuple(p) for i, p in enumerate(_FUEL_LIST)}
    coordinate_to_index = {p: i for i, p in positions.items()}

    adjacency = {i: [] for i in range(N_FUEL)}
    for i, (x, y) in positions.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + dx, y + dy)
            if neighbour in coordinate_to_index:
                adjacency[i].append(coordinate_to_index[neighbour])

    peripheral = [
        i for i, (x, y) in positions.items()
        if any(
            (x + dx, y + dy) not in coordinate_to_index
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
    ]

    layers = compute_layers(adjacency, N_FUEL)
    max_layer = max(layers)
    inner_layers = max_layer // 2
    inner_threshold = max_layer - inner_layers
    inner = [i for i in range(N_FUEL) if layers[i] > inner_threshold]

    peripheral_set = set(peripheral)
    adjacent_to_peripheral = [
        any(j in peripheral_set for j in adjacency[i])
        for i in range(N_FUEL)
    ]

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    centre_x = (max(xs) + min(xs)) / 2.0
    centre_y = (max(ys) + min(ys)) / 2.0
    symmetry_quadrant = [
        i for i, (x, y) in positions.items()
        if x >= centre_x and y >= centre_y
    ]

    return {
        'name': CORE_NAME,
        'N': N_FUEL,
        'N_fresh': N_FRESH,
        'N_once': N_ONCE,
        'N_twice': N_TWICE,
        'peripheral': peripheral,
        'inner': inner,
        'adj': adjacency,
        'pos': positions,
        'adjacent_to_peripheral': adjacent_to_peripheral,
        'symmetry_quadrant': symmetry_quadrant,
    }


CORE_CONFIG = get_westinghouse_config()

print(
    f"{CORE_CONFIG['name']}: N={CORE_CONFIG['N']}, "
    f"fresh={CORE_CONFIG['N_fresh']}, once={CORE_CONFIG['N_once']}, "
    f"twice={CORE_CONFIG['N_twice']}, "
    f"peripheral={len(CORE_CONFIG['peripheral'])}, "
    f"inner={len(CORE_CONFIG['inner'])}"
)

# =============================================================================
# QUBO FORMULATION — same constraints as the supplied solver notebook
# =============================================================================


def _variable_name(i, fuel_index):
    return f'x_{i}_{fuel_index}'


def _add_square_penalty(qubo, variables, target, weight):
    """Add weight * (sum(variables) - target)^2 to a QUBO dictionary."""
    variables = list(variables)

    for variable in variables:
        key = (variable, variable)
        qubo[key] += weight * (1.0 - 2.0 * target)

    for left_index, left_variable in enumerate(variables):
        for right_variable in variables[left_index + 1:]:
            key = tuple(sorted((left_variable, right_variable)))
            qubo[key] += 2.0 * weight


class NuclearCoreQUBO:
    def __init__(
        self,
        N,
        N_fresh,
        N_once,
        N_twice,
        peripheral,
        inner,
        adjacency,
        pos,
        adjacent_to_peripheral=None,
        symmetry_quadrant=None,
    ):
        self.N = N
        self.N_fresh = N_fresh
        self.N_once = N_once
        self.N_twice = N_twice
        self.B = peripheral
        self.I = inner
        self.adj = adjacency
        self.pos = pos
        self.adjacent_to_peripheral = adjacent_to_peripheral or [False] * N
        self.Q1 = symmetry_quadrant

        self.coordinate_to_index = {coords: i for i, coords in self.pos.items()}
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        self.centre_x = (max(xs) + min(xs)) / 2.0
        self.centre_y = (max(ys) + min(ys)) / 2.0

    def _get_symmetric_indices(self, i):
        x, y = self.pos[i]
        relative_x = x - self.centre_x
        relative_y = y - self.centre_y

        targets = [
            (-relative_y + self.centre_x,  relative_x + self.centre_y),
            (-relative_x + self.centre_x, -relative_y + self.centre_y),
            ( relative_y + self.centre_x, -relative_x + self.centre_y),
            (-relative_x + self.centre_x, y),
            (x, -relative_y + self.centre_y),
        ]

        indices = []
        for target_x, target_y in targets:
            key = (round(target_x), round(target_y))
            if key in self.coordinate_to_index:
                symmetric_index = self.coordinate_to_index[key]
                if symmetric_index != i:
                    indices.append(symmetric_index)

        return sorted(set(indices))

    def symmetry_pairs(self):
        if self.Q1 is None:
            return []

        q1_set = set(self.Q1)
        pairs = set()
        for i in self.Q1:
            for j in self._get_symmetric_indices(i):
                if j not in q1_set:
                    pairs.add(tuple(sorted((i, j))))
        return sorted(pairs)

    def build_qubo(
        self,
        w_one_per_cell=1.0,
        w_counts=1.0,
        w_forbidden=1.0,
        w_adj=1.0,
        w_sym=1.0,
    ):
        qubo = defaultdict(float)

        # H1: exactly one fuel type per active assembly
        for i in range(self.N):
            _add_square_penalty(
                qubo,
                [_variable_name(i, fuel_index) for fuel_index in range(3)],
                target=1,
                weight=w_one_per_cell,
            )

        # H2: exact global fuel counts
        for fuel_index, target in enumerate(
            (self.N_fresh, self.N_once, self.N_twice)
        ):
            _add_square_penalty(
                qubo,
                [_variable_name(i, fuel_index) for i in range(self.N)],
                target=target,
                weight=w_counts,
            )

        # H3: no twice-burnt fuel on the periphery and no fresh fuel in the inner zone
        _add_square_penalty(
            qubo,
            [_variable_name(i, 2) for i in self.B],
            target=0,
            weight=w_forbidden,
        )
        _add_square_penalty(
            qubo,
            [_variable_name(i, 0) for i in self.I],
            target=0,
            weight=w_forbidden,
        )

        # H4: adjacency restrictions
        peripheral_set = set(self.B)
        for i in range(self.N):
            for j in self.adj.get(i, []):
                if j <= i:
                    continue

                if i not in peripheral_set and j not in peripheral_set:
                    key = tuple(sorted((_variable_name(i, 0), _variable_name(j, 0))))
                    qubo[key] += w_adj

                key = tuple(sorted((_variable_name(i, 2), _variable_name(j, 2))))
                qubo[key] += w_adj

        # H5: optional symmetry penalty, matching the solver notebook semantics
        if self.Q1 is not None:
            for i, j in self.symmetry_pairs():
                for fuel_index in range(3):
                    left = _variable_name(i, fuel_index)
                    right = _variable_name(j, fuel_index)
                    qubo[(left, left)] += w_sym
                    qubo[(right, right)] += w_sym
                    qubo[tuple(sorted((left, right)))] -= 2.0 * w_sym

        return dict(qubo)


# Method-specific weights adapted from the tuned solver sections.
# Larger absolute values improve separation between hard-constraint penalties
# and the small diversity bias used to obtain distinct feasible solutions.
METHOD_WEIGHTS = {
    'sa': {
        'w_one_per_cell': 1.0,
        'w_counts': 1.0,
        'w_forbidden': 1.0,
        'w_adj': 1.0,
        'w_sym': 1.0,
    },
    'sqa': {
        'w_one_per_cell': 1.0,
        'w_counts': 1.0,
        'w_forbidden': 1.0,
        'w_adj': 1.0,
        'w_sym': 1.0,
    },
}


def build_westinghouse_bqm(method, use_symmetry, diversity_seed=None, diversity_strength=1.0):
    if method not in METHOD_WEIGHTS:
        raise ValueError(f'Unsupported method: {method}')

    cfg = CORE_CONFIG
    symmetry_quadrant = cfg['symmetry_quadrant'] if use_symmetry else None

    model = NuclearCoreQUBO(
        N=cfg['N'],
        N_fresh=cfg['N_fresh'],
        N_once=cfg['N_once'],
        N_twice=cfg['N_twice'],
        peripheral=cfg['peripheral'],
        inner=cfg['inner'],
        adjacency=cfg['adj'],
        pos=cfg['pos'],
        adjacent_to_peripheral=cfg['adjacent_to_peripheral'],
        symmetry_quadrant=symmetry_quadrant,
    )

    qubo = model.build_qubo(**METHOD_WEIGHTS[method])

    # Exact-QUBO mode: parallelization must not modify the optimization
    # problem. Diversity is obtained only from independent sampler seeds.
    if diversity_seed is not None or float(diversity_strength) != 0.0:
        raise ValueError(
            'Exact-QUBO mode does not permit linear diversity perturbations. '
            'Use distinct solver seeds while keeping diversity_strength=0.0.'
        )

    return dimod.BinaryQuadraticModel.from_qubo(qubo), model


def solve_qubo(bqm, method='sa', num_reads=100, seed=None, **kwargs):
    """Solve one BQM batch with SA or SQA."""
    n_assemblies = bqm.num_variables // 3
    default_sweeps = min(2500, 30 * n_assemblies)

    if method == 'sa':
        qubo_values = np.abs(
            [value for value in list(bqm.linear.values()) + list(bqm.quadratic.values()) if value != 0]
        )
        energy_scale = float(np.percentile(qubo_values, 75)) if len(qubo_values) else 1.0
        sampler = oj.SASampler()
        return sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=kwargs.get('num_sweeps', default_sweeps),
            beta_min=kwargs.get('beta_min', 1.0 / energy_scale),
            beta_max=kwargs.get('beta_max', 15.0 / energy_scale),
            seed=seed,
        )

    if method == 'sqa':
        sampler = oj.SQASampler()
        return sampler.sample(
            bqm,
            num_reads=num_reads,
            beta=kwargs.get('beta', 8.0),
            gamma=kwargs.get('gamma', 4.0),
            trotter=kwargs.get('trotter', 4),
            num_sweeps=kwargs.get('num_sweeps', default_sweeps),
            seed=seed,
        )


    raise ValueError(f'Unknown solver method: {method}')

# =============================================================================
# FEASIBILITY FILTER, DECODING, AND STAGED 6-GROUP PATTERN-BANK GENERATION
# =============================================================================

import json
import math
from pathlib import Path


def check_feasibility(
    sample,
    N,
    B,
    I,
    adj,
    adjacent_to_peripheral,
    N_fresh,
    N_once,
    N_twice,
):
    """Exact hard-constraint checker copied from the solver workflow."""
    del adjacent_to_peripheral  # retained in the signature for solver compatibility

    peripheral_set = set(B)
    inner_set = set(I)

    fresh = np.zeros(N, dtype=int)
    once = np.zeros(N, dtype=int)
    twice = np.zeros(N, dtype=int)

    for variable, value in sample.items():
        if int(value) != 1:
            continue
        parts = str(variable).split('_')
        if len(parts) != 3 or parts[0] != 'x':
            continue
        i = int(parts[1])
        fuel_index = int(parts[2])
        if fuel_index == 0:
            fresh[i] = 1
        elif fuel_index == 1:
            once[i] = 1
        elif fuel_index == 2:
            twice[i] = 1

    # 1. Exactly one fuel type per active assembly
    if np.any(fresh + once + twice != 1):
        return False

    # 2. Exact counts
    if int(fresh.sum()) != N_fresh:
        return False
    if int(once.sum()) != N_once:
        return False
    if int(twice.sum()) != N_twice:
        return False

    # 3. No twice-burnt fuel on the periphery
    if any(twice[i] for i in peripheral_set):
        return False

    # 4. No fresh fuel in the inner zone
    if any(fresh[i] for i in inner_set):
        return False

    # 5. No interior fresh-fresh adjacency
    for i in range(N):
        for j in adj.get(i, []):
            if j > i and fresh[i] and fresh[j]:
                if not (i in peripheral_set or j in peripheral_set):
                    return False

    # 6. No twice-burnt--twice-burnt adjacency
    for i in range(N):
        for j in adj.get(i, []):
            if j > i and twice[i] and twice[j]:
                return False

    return True


def decode_sample_to_pattern(sample, n_cells=N_FUEL):
    """Convert a feasible one-hot QUBO sample into OpenMC fuel labels 1/2/3."""
    pattern = []
    for i in range(n_cells):
        active = [
            int(sample.get(_variable_name(i, fuel_index), 0))
            for fuel_index in range(3)
        ]
        if sum(active) != 1:
            return None
        pattern.append(active.index(1) + 1)
    return pattern


def symmetry_mismatch_count(pattern, model):
    """Count unequal fuel labels across the symmetry pairs used in H5."""
    if not model.symmetry_pairs():
        return 0
    return sum(pattern[i] != pattern[j] for i, j in model.symmetry_pairs())


def validate_decoded_pattern(pattern, cfg=CORE_CONFIG):
    """Independent pattern-level verification before OpenMC simulation."""
    if pattern is None or len(pattern) != cfg['N']:
        return False

    counts = Counter(int(value) for value in pattern)
    if counts != Counter({1: cfg['N_fresh'], 2: cfg['N_once'], 3: cfg['N_twice']}):
        return False

    peripheral_set = set(cfg['peripheral'])
    inner_set = set(cfg['inner'])

    if any(pattern[i] == 3 for i in peripheral_set):
        return False
    if any(pattern[i] == 1 for i in inner_set):
        return False

    for i in range(cfg['N']):
        for j in cfg['adj'][i]:
            if j <= i:
                continue
            if pattern[i] == 3 and pattern[j] == 3:
                return False
            if (
                pattern[i] == 1
                and pattern[j] == 1
                and i not in peripheral_set
                and j not in peripheral_set
            ):
                return False

    return True


SOLVER_BATCH_READS = {
    'sa': 1000,
    'sqa': 1000,
}

SOLVER_METHODS = ('sa', 'sqa')
SYMMETRY_SETTINGS = (True, False)


def _run_solver_batch(
    method,
    use_symmetry,
    batch_index,
    base_seed,
    diversity_strength,
):
    """
    Run, decode, and validate one independent solver batch.

    This function is executed inside a joblib ``loky`` worker. Each worker
    solves the exact same unmodified BQM used by the sequential solver. Only the
    random sampler seed differs between independent batches.
    """
    cfg = CORE_CONFIG

    solver_seed = int((base_seed + 100003 * batch_index) % (2**31 - 1))
    # Keep the optimization problem identical to the sequential solver.
    # Different solver seeds provide independent trajectories without changing Q.
    diversity_seed = None
    if float(diversity_strength) != 0.0:
        raise ValueError('Exact-QUBO batches require diversity_strength=0.0')

    base_bqm, base_model = build_westinghouse_bqm(
        method=method,
        use_symmetry=use_symmetry,
        diversity_seed=None,
        diversity_strength=0.0,
    )

    sample_set = solve_qubo(
        base_bqm,
        method=method,
        num_reads=SOLVER_BATCH_READS[method],
        seed=solver_seed,
    )

    batch_records = []
    batch_seen = set()
    rows_examined = 0
    feasible_hits = 0

    for row_index, row in enumerate(sample_set.data(['sample', 'energy'])):
        rows_examined += 1
        sample = row.sample

        if not check_feasibility(
            sample,
            cfg['N'],
            cfg['peripheral'],
            cfg['inner'],
            cfg['adj'],
            cfg['adjacent_to_peripheral'],
            cfg['N_fresh'],
            cfg['N_once'],
            cfg['N_twice'],
        ):
            continue

        feasible_hits += 1
        pattern = decode_sample_to_pattern(sample, cfg['N'])
        if not validate_decoded_pattern(pattern, cfg):
            continue

        pattern_key = tuple(pattern)
        if pattern_key in batch_seen:
            continue
        batch_seen.add(pattern_key)

        batch_records.append({
            'solver': method,
            'symmetry_enabled': bool(use_symmetry),
            'pattern': pattern,
            'qubo_energy': float(base_bqm.energy(sample)),
            'symmetry_mismatches': int(
                symmetry_mismatch_count(pattern, base_model)
            ),
            'solver_seed': solver_seed,
            'diversity_seed': diversity_seed,
            '_batch_index': int(batch_index),
            '_row_index': int(row_index),
        })

    return {
        'solver': method,
        'symmetry_enabled': bool(use_symmetry),
        'batch_index': int(batch_index),
        'rows_examined': int(rows_examined),
        'feasible_hits': int(feasible_hits),
        'unique_feasible_hits': int(len(batch_records)),
        'records': batch_records,
    }


def _group_sort_key(record, use_symmetry):
    """Deterministic ranking used after parallel candidate collection."""
    stable_tail = (
        record['solver_seed'],
        record['_batch_index'],
        record['_row_index'],
    )
    if use_symmetry:
        return (
            record['symmetry_mismatches'],
            record['qubo_energy'],
            *stable_tail,
        )
    return (
        record['qubo_energy'],
        *stable_tail,
    )



def _atomic_write_json(payload, output_path):
    """Write JSON through a temporary file and atomically replace the target."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + '.tmp')

    with temporary_path.open('w', encoding='utf-8') as file_pointer:
        json.dump(payload, file_pointer, indent=2)
        file_pointer.flush()
        os.fsync(file_pointer.fileno())

    os.replace(temporary_path, output_path)


def _stage_group_key(use_symmetry):
    return 'symmetry_true' if use_symmetry else 'symmetry_false'


def _stage_checkpoint_payload(
    method,
    group_state,
    samples_per_group,
    diversity_strength,
    base_seed,
    complete=False,
):
    """Create the restartable checkpoint document for one solver family."""
    return {
        'schema_version': 3,
        'core_name': CORE_NAME,
        'method': method,
        'solver_batch_reads': int(SOLVER_BATCH_READS[method]),
        'samples_per_group': int(samples_per_group),
        'diversity_strength': float(diversity_strength),
        'base_seed': int(base_seed),
        'complete': bool(complete),
        'groups': {
            _stage_group_key(use_symmetry): {
                'symmetry_enabled': bool(use_symmetry),
                'next_batch': int(group_state[use_symmetry]['next_batch']),
                'records': group_state[use_symmetry]['accepted'],
            }
            for use_symmetry in SYMMETRY_SETTINGS
        },
    }


def _restore_stage_checkpoint(
    checkpoint_file,
    method,
    samples_per_group,
    diversity_strength,
    base_seed,
):
    """Restore valid records and batch counters from a compatible checkpoint."""
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.exists():
        return None

    try:
        with checkpoint_file.open('r', encoding='utf-8') as file_pointer:
            payload = json.load(file_pointer)
    except (OSError, json.JSONDecodeError) as exception:
        print(
            f'Ignoring unreadable {method.upper()} checkpoint '
            f'{checkpoint_file}: {exception}'
        )
        return None

    compatible = (
        isinstance(payload, dict)
        and payload.get('schema_version') == 3
        and payload.get('core_name') == CORE_NAME
        and payload.get('method') == method
        and int(payload.get('solver_batch_reads', -1)) == int(SOLVER_BATCH_READS[method])
        and int(payload.get('samples_per_group', -1)) == int(samples_per_group)
        and float(payload.get('diversity_strength', -1.0)) == float(diversity_strength)
        and int(payload.get('base_seed', -1)) == int(base_seed)
    )
    if not compatible:
        print(
            f'Ignoring incompatible {method.upper()} checkpoint: '
            f'{checkpoint_file}'
        )
        return None

    restored = {}
    for use_symmetry in SYMMETRY_SETTINGS:
        group_payload = payload.get('groups', {}).get(
            _stage_group_key(use_symmetry),
            {},
        )
        seen = set()
        accepted = []

        for record in group_payload.get('records', []):
            if record.get('solver') != method:
                continue
            if bool(record.get('symmetry_enabled')) != bool(use_symmetry):
                continue
            pattern = record.get('pattern')
            if not validate_decoded_pattern(pattern, CORE_CONFIG):
                continue
            pattern_key = tuple(pattern)
            if pattern_key in seen:
                continue
            seen.add(pattern_key)
            accepted.append(record)

        restored[use_symmetry] = {
            'accepted': accepted,
            'seen': seen,
            'next_batch': max(0, int(group_payload.get('next_batch', 0))),
        }

    print(
        f'Restored {method.upper()} checkpoint: '
        + ' | '.join(
            f'sym={use_symmetry}: '
            f"{len(restored[use_symmetry]['accepted'])}/{samples_per_group}"
            for use_symmetry in SYMMETRY_SETTINGS
        )
    )
    return restored


def generate_solver_stage(
    method,
    samples_per_group=100,
    checkpoint_file=None,
    n_jobs=4,
    batches_per_group_per_wave=None,
    max_batches_per_group=5000,
    diversity_strength=0.0,
    base_seed=20260718,
    resume=True,
):
    """
    Generate symmetry=True and symmetry=False groups for one solver family.

    Only the two symmetry groups belonging to ``method`` share a worker pool.
    The function returns only after both groups have been deterministically
    merged, independently audited, and checkpointed as complete.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError as exception:
        raise ImportError(
            'Parallel solver generation requires joblib. '
            'Install it with: pip install joblib'
        ) from exception

    if method not in SOLVER_METHODS:
        raise ValueError(f'Unsupported method: {method}')
    if float(diversity_strength) != 0.0:
        raise ValueError(
            'This notebook preserves the original QUBO exactly; '
            'diversity_strength must be 0.0.'
        )
    if samples_per_group <= 0:
        raise ValueError('samples_per_group must be positive')
    if n_jobs <= 0:
        raise ValueError('n_jobs must be a positive integer')
    if max_batches_per_group <= 0:
        raise ValueError('max_batches_per_group must be positive')

    if batches_per_group_per_wave is None:
        # Two symmetry groups share the pool. This fills approximately all
        # workers without creating a large queue of unnecessary future tasks.
        batches_per_group_per_wave = max(1, math.ceil(n_jobs / 2))
    if batches_per_group_per_wave <= 0:
        raise ValueError('batches_per_group_per_wave must be positive')

    checkpoint_file = Path(
        checkpoint_file
        or f'solver_checkpoints/{method}_westinghouse_193.json'
    )

    method_seed_offsets = {'sa': 0, 'sqa': 2_000_000}
    method_seed = int(base_seed + method_seed_offsets[method])

    group_state = {
        use_symmetry: {
            'accepted': [],
            'seen': set(),
            'next_batch': 0,
            'base_seed': int(
                method_seed + 10_000 * symmetry_index
            ),
        }
        for symmetry_index, use_symmetry in enumerate(SYMMETRY_SETTINGS)
    }

    if resume:
        restored = _restore_stage_checkpoint(
            checkpoint_file=checkpoint_file,
            method=method,
            samples_per_group=samples_per_group,
            diversity_strength=diversity_strength,
            base_seed=base_seed,
        )
        if restored is not None:
            for use_symmetry in SYMMETRY_SETTINGS:
                group_state[use_symmetry]['accepted'] = restored[use_symmetry]['accepted']
                group_state[use_symmetry]['seen'] = restored[use_symmetry]['seen']
                group_state[use_symmetry]['next_batch'] = restored[use_symmetry]['next_batch']

    def group_is_complete(use_symmetry):
        return (
            len(group_state[use_symmetry]['accepted'])
            >= samples_per_group
        )

    wave_index = 0
    print(
        f'\n=== {method.upper()} stage: {n_jobs} workers, '
        f'dynamic full-node scheduling ==='
    )

    with Parallel(
        n_jobs=n_jobs,
        backend='loky',
        batch_size=1,
        verbose=0,
    ) as parallel:
        while not all(group_is_complete(value) for value in SYMMETRY_SETTINGS):
            wave_index += 1
            task_specs = []

            # Dynamically redistribute the full worker pool. While both
            # symmetry groups are incomplete they split the node evenly.
            # As soon as one group finishes, all available workers are moved
            # to the remaining group instead of leaving half the node idle.
            active_groups = [
                use_symmetry
                for use_symmetry in SYMMETRY_SETTINGS
                if (
                    not group_is_complete(use_symmetry)
                    and group_state[use_symmetry]['next_batch']
                    < max_batches_per_group
                )
            ]

            if active_groups:
                base_allocation = n_jobs // len(active_groups)
                extra_workers = n_jobs % len(active_groups)

                for group_index, use_symmetry in enumerate(active_groups):
                    state = group_state[use_symmetry]
                    requested_batches = (
                        base_allocation
                        + (1 if group_index < extra_workers else 0)
                    )
                    remaining_budget = (
                        max_batches_per_group - state['next_batch']
                    )
                    group_batches = min(
                        requested_batches,
                        remaining_budget,
                    )

                    for _slot in range(group_batches):
                        batch_index = state['next_batch']
                        state['next_batch'] += 1
                        task_specs.append({
                            'method': method,
                            'use_symmetry': use_symmetry,
                            'batch_index': batch_index,
                            'base_seed': state['base_seed'],
                            'diversity_strength': diversity_strength,
                        })

            if not task_specs:
                incomplete = {
                    str(use_symmetry): len(
                        group_state[use_symmetry]['accepted']
                    )
                    for use_symmetry in SYMMETRY_SETTINGS
                    if not group_is_complete(use_symmetry)
                }
                raise RuntimeError(
                    f'{method.upper()} exhausted its batch budget before '
                    f'completion: {incomplete}'
                )

            try:
                batch_results = parallel(
                    delayed(_run_solver_batch)(**task_spec)
                    for task_spec in task_specs
                )
            except Exception as exception:
                # The checkpoint from the previous completed wave remains valid.
                raise RuntimeError(
                    f'{method.upper()} stage failed during wave {wave_index}. '
                    f'Rerun with resume=True to continue from the last '
                    f'checkpoint.'
                ) from exception

            # Joblib preserves submission order. Parent-only merging makes
            # duplicate handling deterministic regardless of worker finish time.
            for result in batch_results:
                use_symmetry = bool(result['symmetry_enabled'])
                state = group_state[use_symmetry]

                for record in result['records']:
                    pattern_key = tuple(record['pattern'])
                    if pattern_key in state['seen']:
                        continue
                    state['seen'].add(pattern_key)
                    state['accepted'].append(record)

            wave_rows = sum(result['rows_examined'] for result in batch_results)
            wave_feasible = sum(result['feasible_hits'] for result in batch_results)
            wave_unique = sum(result['unique_feasible_hits'] for result in batch_results)

            print(
                f'{method.upper()} wave {wave_index:>4} '
                f'({len(task_specs)} batches, reads={wave_rows}, '
                f'feasible={wave_feasible}, batch-unique={wave_unique}): '
                + ' | '.join(
                    f'sym={str(use_symmetry):<5}: '
                    f"{len(group_state[use_symmetry]['accepted'])}"
                    f'/{samples_per_group}'
                    for use_symmetry in SYMMETRY_SETTINGS
                )
            )

            _atomic_write_json(
                _stage_checkpoint_payload(
                    method=method,
                    group_state=group_state,
                    samples_per_group=samples_per_group,
                    diversity_strength=diversity_strength,
                    base_seed=base_seed,
                    complete=False,
                ),
                checkpoint_file,
            )

        # Deterministically rank and trim each group only after both groups have
        # enough unique feasible candidates.
        selected_records = []
        for use_symmetry in SYMMETRY_SETTINGS:
            records = group_state[use_symmetry]['accepted']
            records.sort(
                key=lambda record, symmetry=use_symmetry: _group_sort_key(
                    record,
                    symmetry,
                )
            )
            selected_records.extend(records[:samples_per_group])

        audit_results = parallel(
            delayed(validate_decoded_pattern)(record['pattern'], CORE_CONFIG)
            for record in selected_records
        )

    if not all(audit_results):
        failed_indices = [
            index for index, passed in enumerate(audit_results) if not passed
        ]
        raise AssertionError(
            f'{method.upper()} final feasibility audit failed for records: '
            f'{failed_indices}'
        )

    group_counts = Counter(
        bool(record['symmetry_enabled']) for record in selected_records
    )
    for use_symmetry in SYMMETRY_SETTINGS:
        if group_counts[use_symmetry] != samples_per_group:
            raise AssertionError(
                f'{method}, symmetry={use_symmetry}: expected '
                f'{samples_per_group}, got {group_counts[use_symmetry]}'
            )

    # Replace the in-memory state with the selected records before writing the
    # complete checkpoint, keeping the restart document compact and canonical.
    for use_symmetry in SYMMETRY_SETTINGS:
        group_state[use_symmetry]['accepted'] = [
            record
            for record in selected_records
            if bool(record['symmetry_enabled']) == bool(use_symmetry)
        ]
        group_state[use_symmetry]['seen'] = {
            tuple(record['pattern'])
            for record in group_state[use_symmetry]['accepted']
        }

    _atomic_write_json(
        _stage_checkpoint_payload(
            method=method,
            group_state=group_state,
            samples_per_group=samples_per_group,
            diversity_strength=diversity_strength,
            base_seed=base_seed,
            complete=True,
        ),
        checkpoint_file,
    )

    print(
        f'Completed and checkpointed {method.upper()}: '
        f'{len(selected_records)} validated patterns -> {checkpoint_file}'
    )
    return selected_records


def generate_solver_pattern_bank_staged(
    samples_per_group=100,
    output_file='qubo_patterns_westinghouse_193.json',
    checkpoint_dir='solver_checkpoints',
    worker_counts=None,
    batches_per_group_per_wave=None,
    max_batches_per_group=5000,
    diversity_strength=0.0,
    base_seed=20260718,
    resume=True,
):
    """
    Run SA and SQA as sequential solver-family stages.

    Symmetry True/False groups remain parallel inside each active stage. The
    combined output is atomically updated after each completed solver family.
    """
    available_cpus = max(1, os.cpu_count() or 1)
    if worker_counts is None:
        worker_counts = {
            'sa': min(4, max(1, available_cpus - 1)),
            'sqa': min(3, max(1, available_cpus // 2)),
        }

    if batches_per_group_per_wave is None:
        batches_per_group_per_wave = {
            method: max(1, math.ceil(worker_counts[method] / 2))
            for method in SOLVER_METHODS
        }

    missing_workers = set(SOLVER_METHODS) - set(worker_counts)
    if missing_workers:
        raise ValueError(
            f'Missing worker counts for: {sorted(missing_workers)}'
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_patterns = []
    print('\nStaged solver order: SA -> SQA')

    for stage_number, method in enumerate(SOLVER_METHODS, start=1):
        stage_records = generate_solver_stage(
            method=method,
            samples_per_group=samples_per_group,
            checkpoint_file=(
                checkpoint_dir
                / f'{stage_number:02d}_{method}_westinghouse_193.json'
            ),
            n_jobs=int(worker_counts[method]),
            batches_per_group_per_wave=int(
                batches_per_group_per_wave[method]
            ),
            max_batches_per_group=max_batches_per_group,
            diversity_strength=diversity_strength,
            base_seed=base_seed,
            resume=resume,
        )

        all_patterns.extend(stage_records)

        # Save a clean partial master dataset after every completed solver stage.
        partial_records = []
        for sample_id, record in enumerate(all_patterns):
            cleaned_record = {
                key: value
                for key, value in record.items()
                if not key.startswith('_')
            }
            cleaned_record['id'] = sample_id
            partial_records.append(cleaned_record)
        _atomic_write_json(partial_records, output_file)

        print(
            f'Master checkpoint after {method.upper()}: '
            f'{len(partial_records)} records -> {output_file}'
        )

    expected_total = (
        len(SOLVER_METHODS)
        * len(SYMMETRY_SETTINGS)
        * samples_per_group
    )
    if len(all_patterns) != expected_total:
        raise AssertionError(
            f'Expected {expected_total} solver patterns, '
            f'got {len(all_patterns)}.'
        )

    final_patterns = []
    for sample_id, record in enumerate(all_patterns):
        cleaned_record = {
            key: value
            for key, value in record.items()
            if not key.startswith('_')
        }
        cleaned_record['id'] = sample_id
        final_patterns.append(cleaned_record)

    group_counts = Counter(
        (record['solver'], record['symmetry_enabled'])
        for record in final_patterns
    )
    for method in SOLVER_METHODS:
        for use_symmetry in SYMMETRY_SETTINGS:
            actual = group_counts[(method, use_symmetry)]
            if actual != samples_per_group:
                raise AssertionError(
                    f'{method}, symmetry={use_symmetry}: expected '
                    f'{samples_per_group}, got {actual}'
                )

    # Report cross-group overlap without deleting provenance. A pattern found by
    # two solvers remains a valid independent solver outcome; within-group
    # duplicates have already been removed deterministically.
    pattern_groups = defaultdict(set)
    for record in final_patterns:
        pattern_groups[tuple(record['pattern'])].add(
            (record['solver'], record['symmetry_enabled'])
        )
    cross_group_overlaps = sum(
        len(groups) > 1 for groups in pattern_groups.values()
    )

    _atomic_write_json(final_patterns, output_file)
    print(f'Final group counts: {dict(group_counts)}')
    print(
        f'Cross-group repeated loading patterns (provenance retained): '
        f'{cross_group_overlaps}'
    )
    print(
        f'Saved {len(final_patterns)} feasible staged QUBO patterns to '
        f'{output_file}'
    )
    return final_patterns


# Backward-compatible name: calls now use the reliable staged implementation.
generate_solver_pattern_bank = generate_solver_pattern_bank_staged




# =============================================================================
# ARES DRY-RUN DRIVER
# =============================================================================

def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_environment():
    cross_sections = os.environ.get('OPENMC_CROSS_SECTIONS')
    if not cross_sections:
        raise RuntimeError('OPENMC_CROSS_SECTIONS is not set')
    if not Path(cross_sections).is_file():
        raise RuntimeError(f'Cross-section XML does not exist: {cross_sections}')
    executable = shutil.which('openmc')
    if not executable:
        raise RuntimeError('openmc executable is not on PATH')
    return executable, str(Path(cross_sections).resolve())


def environment_record():
    executable, cross_sections = require_environment()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    return {
        'hostname': socket.gethostname(),
        'python_executable': os.sys.executable,
        'python_version': os.sys.version,
        'openmc_python_version': getattr(openmc, '__version__', 'unknown'),
        'openmc_executable': executable,
        'cross_sections': cross_sections,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'allowed_cpu_count': len(affinity),
        'allowed_cpu_ids': affinity,
    }


def run_solver_phase(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'solver_checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)

    print('Environment:')
    print(json.dumps(environment_record(), indent=2))
    print('\nSolver benchmark target:')
    print(f'  samples requested per group: {args.samples_per_group}')
    print(f'  SA workers:  {args.sa_workers}')
    print(f'  SQA workers: {args.sqa_workers}')
    print(f'  Reads per independent batch: SA={SOLVER_BATCH_READS["sa"]}, SQA={SOLVER_BATCH_READS["sqa"]}')

    records = []
    stages = {}
    overall_start = time.perf_counter()

    for stage_number, method in enumerate(SOLVER_METHODS, start=1):
        workers = args.sa_workers if method == 'sa' else args.sqa_workers
        batches_per_group = max(1, math.ceil(workers / 2))
        checkpoint = checkpoints / f'{stage_number:02d}_{method}_dryrun.json'

        started = time.perf_counter()
        stage_records = generate_solver_stage(
            method=method,
            samples_per_group=args.samples_per_group,
            checkpoint_file=checkpoint,
            n_jobs=workers,
            batches_per_group_per_wave=batches_per_group,
            max_batches_per_group=args.max_batches_per_group,
            diversity_strength=0.0,
            base_seed=args.base_seed,
            resume=args.resume,
        )
        elapsed = time.perf_counter() - started

        checkpoint_payload = json.loads(checkpoint.read_text())
        group_details = {}
        for group_name, group in checkpoint_payload['groups'].items():
            group_details[group_name] = {
                'symmetry_enabled': bool(group['symmetry_enabled']),
                'accepted_samples': len(group['records']),
                'independent_batches_consumed': int(group['next_batch']),
            }

        stages[method] = {
            'elapsed_seconds_for_both_symmetry_groups_running_concurrently': elapsed,
            'workers': workers,
            'batches_per_group_per_wave': batches_per_group,
            'reads_per_batch': SOLVER_BATCH_READS[method],
            'groups': group_details,
        }
        records.extend(stage_records)
        print(f'\n{method.upper()} stage wall time: {elapsed:.3f} s')

    # Round-robin the four groups before assigning IDs. Consequently every
    # 50-sample production chunk contains an approximately equal mix of SA/SQA
    # and symmetry=True/False records instead of long solver-specific blocks.
    group_order = [
        (method, bool(symmetry))
        for method in SOLVER_METHODS
        for symmetry in SYMMETRY_SETTINGS
    ]
    grouped = {
        key: sorted(
            [
                record for record in records
                if (record['solver'], bool(record['symmetry_enabled'])) == key
            ],
            key=lambda record: (
                int(record.get('_batch_index', 0)),
                int(record.get('_row_index', 0)),
                int(record.get('solver_seed', 0)),
            ),
        )
        for key in group_order
    }

    cleaned = []
    for group_index in range(args.samples_per_group):
        for key in group_order:
            record = grouped[key][group_index]
            item = {
                field: value
                for field, value in record.items()
                if not field.startswith('_')
            }
            item['id'] = len(cleaned)
            cleaned.append(item)

    expected = len(SOLVER_METHODS) * len(SYMMETRY_SETTINGS) * args.samples_per_group
    if len(cleaned) != expected:
        raise AssertionError(f'Expected {expected} patterns, got {len(cleaned)}')

    counts = Counter((item['solver'], item['symmetry_enabled']) for item in cleaned)
    pattern_file = (
        Path(args.pattern_file).resolve()
        if args.pattern_file
        else out / 'patterns_sa_sqa.json'
    )
    atomic_json(cleaned, pattern_file)

    metrics = {
        'phase': 'solver',
        'environment': environment_record(),
        'samples_per_solver_symmetry_group': args.samples_per_group,
        'total_patterns': len(cleaned),
        'group_counts': {f'{method}|symmetry={symmetry}': counts[(method, symmetry)]
                         for method in SOLVER_METHODS for symmetry in SYMMETRY_SETTINGS},
        'stages': stages,
        'total_elapsed_seconds': time.perf_counter() - overall_start,
        'pattern_file': str(pattern_file),
    }
    atomic_json(metrics, out / 'solver_metrics.json')
    print('\nSolver phase complete:')
    print(json.dumps(metrics, indent=2))


def cpu_slots(worker_count, threads_per_run):
    if not hasattr(os, 'sched_getaffinity'):
        allowed = list(range(os.cpu_count() or 1))
    else:
        allowed = sorted(os.sched_getaffinity(0))

    slurm_target = int(os.environ.get('SLURM_CPUS_PER_TASK', len(allowed)))
    usable = allowed[:min(len(allowed), slurm_target)]
    minimum = worker_count * threads_per_run
    if len(usable) < minimum:
        raise RuntimeError(
            f'Need at least {minimum} CPUs for {worker_count}x{threads_per_run}, '
            f'but Slurm affinity exposes only {len(usable)}: {usable}'
        )

    # Distribute all remaining allocated cores across the active OpenMC slots.
    # For the Ares production default 5 workers on 48 cores this gives 3x10 + 2x9 = 48.
    slot_sizes = [threads_per_run] * worker_count
    for index in range(len(usable) - minimum):
        slot_sizes[index % worker_count] += 1

    slots = []
    start = 0
    for size in slot_sizes:
        slots.append(usable[start:start + size])
        start += size
    return slots, usable


def simulate_pattern_dry(record, batches, inactive, particles, threads, cpu_ids, tmp_root):
    sample_id = int(record['id'])
    pattern = np.asarray(record['pattern'], dtype=int)
    started_total = time.perf_counter()

    if pattern.ndim != 1 or len(pattern) != 193:
        return {'error': f'Sample {sample_id}: invalid pattern shape {pattern.shape}'}
    if not validate_decoded_pattern(pattern.tolist()):
        return {'error': f'Sample {sample_id}: hard-constraint validation failed'}

    executable, _ = require_environment()
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'sample_{sample_id:04d}_', dir=tmp_root) as tmpdir:
        try:
            build_started = time.perf_counter()
            model = build_pwr_core(pattern)
            model.settings.batches = int(batches)
            model.settings.inactive = int(inactive)
            model.settings.particles = int(particles)
            model.settings.seed = int(20260718 + sample_id)
            model.export_to_xml(tmpdir)
            build_seconds = time.perf_counter() - build_started

            cpu_list = ','.join(str(value) for value in cpu_ids)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(threads)
            env['OMP_PROC_BIND'] = 'close'
            env['OMP_PLACES'] = 'cores'
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            env['NUMEXPR_NUM_THREADS'] = '1'

            command = ['taskset', '--cpu-list', cpu_list, executable, '-s', str(threads)]
            openmc_started = time.perf_counter()
            result = subprocess.run(
                command,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            openmc_seconds = time.perf_counter() - openmc_started
            if result.returncode != 0:
                return {
                    'error': f'Sample {sample_id}: OpenMC failed with code {result.returncode}',
                    'stderr_tail': result.stderr[-2000:],
                    'stdout_tail': result.stdout[-2000:],
                    'command': command,
                }

            post_started = time.perf_counter()
            statepoint_path = Path(tmpdir) / f'statepoint.{batches}.h5'
            if not statepoint_path.is_file():
                return {'error': f'Sample {sample_id}: missing {statepoint_path.name}'}

            with openmc.StatePoint(statepoint_path) as statepoint:
                k_eff = float(statepoint.keff.n)
                tally = statepoint.get_tally(name='power')
                raw = tally.get_values(scores=['fission-q-recoverable'])
                pin_power = raw.reshape((GRID_SIDE * 17, GRID_SIDE * 17))
                if float(np.max(pin_power)) < 1.0e-10:
                    return {'error': f'Sample {sample_id}: near-zero power tally'}

                assembly_power = pin_power.reshape(GRID_SIDE, 17, GRID_SIDE, 17).sum(axis=(1, 3))
                assembly_rates = assembly_power[assembly_power > 0]
                Fa = float(np.max(assembly_rates) / np.mean(assembly_rates))

                pin_rates = pin_power[pin_power > 1.0e-4 * np.max(pin_power)]
                Fq = float(np.max(pin_rates) / np.mean(pin_rates))

            post_seconds = time.perf_counter() - post_started
            return {
                'id': sample_id,
                'solver': record['solver'],
                'symmetry_enabled': bool(record['symmetry_enabled']),
                'pattern': pattern.tolist(),
                'keff': k_eff,
                'peaking_assembly': Fa,
                'peaking_pin': Fq,
                'dry_batches': int(batches),
                'dry_inactive': int(inactive),
                'dry_particles': int(particles),
                'histories': int(batches) * int(particles),
                'openmc_threads': int(threads),
                'cpu_ids': list(cpu_ids),
                'build_export_seconds': build_seconds,
                'openmc_seconds': openmc_seconds,
                'postprocess_seconds': post_seconds,
                'total_seconds': time.perf_counter() - started_total,
            }
        except Exception as exception:
            return {'error': f'Sample {sample_id}: {type(exception).__name__}: {exception}'}


def _interleave_records_by_group(records):
    """Round-robin the four solver/symmetry groups for balanced OpenMC waves."""
    group_order = [
        (method, bool(symmetry))
        for method in SOLVER_METHODS
        for symmetry in SYMMETRY_SETTINGS
    ]
    grouped = {}
    for key in group_order:
        grouped[key] = sorted(
            [
                item for item in records
                if (item['solver'], bool(item['symmetry_enabled'])) == key
            ],
            key=lambda item: int(item['id']),
        )

    lengths = {key: len(items) for key, items in grouped.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f'Unequal dry-run group sizes: {lengths}')

    ordered = []
    group_size = next(iter(lengths.values()), 0)
    for sample_index in range(group_size):
        for key in group_order:
            ordered.append(grouped[key][sample_index])
    return ordered, group_order


def run_openmc_phase(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pattern_file = (
        Path(args.pattern_file).resolve()
        if args.pattern_file
        else out / 'dryrun_patterns_sa_sqa.json'
    )
    if not pattern_file.is_file():
        raise FileNotFoundError(f'Pattern file not found: {pattern_file}')

    records = json.loads(pattern_file.read_text())
    expected = (
        len(SOLVER_METHODS)
        * len(SYMMETRY_SETTINGS)
        * args.samples_per_group
    )
    if len(records) != expected:
        raise AssertionError(
            f'Expected {expected} dry-run patterns, found {len(records)}'
        )

    records, group_order = _interleave_records_by_group(records)

    # Ares-oriented default: 5 concurrent OpenMC subprocesses share 48 physical cores.
    # cpu_slots() distributes spare cores, yielding 10,10,10,9,9 threads.
    slots, allowed = cpu_slots(
        args.openmc_workers,
        args.threads_per_run,
    )
    used = sum(len(slot) for slot in slots)
    slot_thread_counts = [len(slot) for slot in slots]

    print('Environment:')
    print(json.dumps(environment_record(), indent=2))
    print('\nOpenMC dry-run plan:')
    print(f'  patterns: {len(records)}')
    print(
        f'  workers: {args.openmc_workers}; '
        f'slot thread counts: {slot_thread_counts}; total={used} CPUs'
    )
    print(
        f'  CPUs exposed by Slurm: {len(allowed)}; '
        f'reserved/unallocated within task: {len(allowed) - used}'
    )
    print(
        f'  dry physics: batches={args.batches}, '
        f'inactive={args.inactive}, particles={args.particles}'
    )
    print(f'  dry histories/sample: {args.batches * args.particles:,}')
    print(f'  full histories/sample: {500 * 500000:,}')
    print(
        '  group order is round-robin so every full 16-sample wave '
        'contains an equal mix of SA/SQA and symmetry settings'
    )

    tmp_root = out / (
        f'tmp_openmc_{os.environ.get("SLURM_JOB_ID", "manual")}'
    )
    tmp_root.mkdir(parents=True, exist_ok=True)

    successful = []
    failures = []
    waves = []
    overall_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.openmc_workers
    ) as executor:
        for wave_index, wave_start_index in enumerate(
            range(0, len(records), args.openmc_workers),
            start=1,
        ):
            wave_records = records[
                wave_start_index:
                wave_start_index + args.openmc_workers
            ]

            wave_start = time.perf_counter()
            future_pairs = []
            for slot_index, record in enumerate(wave_records):
                future = executor.submit(
                    simulate_pattern_dry,
                    record,
                    args.batches,
                    args.inactive,
                    args.particles,
                    len(slots[slot_index]),
                    slots[slot_index],
                    str(tmp_root),
                )
                future_pairs.append((future, record))

            wave_results = []
            for future, record in future_pairs:
                result = future.result()
                result.setdefault('id', int(record['id']))
                result.setdefault('solver', record['solver'])
                result.setdefault(
                    'symmetry_enabled',
                    bool(record['symmetry_enabled']),
                )
                wave_results.append(result)

            wave_elapsed = time.perf_counter() - wave_start
            successful_wave = [
                item for item in wave_results if 'error' not in item
            ]
            failed_wave = [
                item for item in wave_results if 'error' in item
            ]
            successful.extend(successful_wave)
            failures.extend(failed_wave)

            group_counts = Counter(
                (
                    item['solver'],
                    bool(item['symmetry_enabled']),
                )
                for item in wave_records
            )
            active_cores = sum(
                len(slots[index])
                for index in range(len(wave_records))
            )
            wave_record = {
                'wave': wave_index,
                'samples': len(wave_records),
                'wall_seconds': wave_elapsed,
                'active_core_slots': active_cores,
                'allocated_core_slots': used,
                'slot_utilization_fraction': (
                    active_cores / used if used else 0.0
                ),
                'group_counts': {
                    f'{method}|symmetry={symmetry}': int(count)
                    for (method, symmetry), count in group_counts.items()
                },
                'successful': len(successful_wave),
                'failed': len(failed_wave),
                'mean_openmc_seconds_per_sample': (
                    float(np.mean([
                        item['openmc_seconds']
                        for item in successful_wave
                    ]))
                    if successful_wave else None
                ),
                'max_openmc_seconds_per_sample': (
                    float(np.max([
                        item['openmc_seconds']
                        for item in successful_wave
                    ]))
                    if successful_wave else None
                ),
            }
            waves.append(wave_record)
            print(
                f'Wave {wave_index}: samples={len(wave_records)}, '
                f'active_cores={active_cores}/{used}, '
                f'wall={wave_elapsed:.3f}s, '
                f'success={len(successful_wave)}, '
                f'failed={len(failed_wave)}'
            )

            atomic_json(
                sorted(successful, key=lambda item: item['id']),
                out / 'dryrun_openmc_results.partial.json',
            )
            if failures:
                atomic_json(
                    failures,
                    out / 'dryrun_openmc_errors.partial.json',
                )

    total_elapsed = time.perf_counter() - overall_start
    shutil.rmtree(tmp_root, ignore_errors=True)

    successful.sort(key=lambda item: item['id'])
    atomic_json(
        successful,
        out / 'dryrun_openmc_results.json',
    )
    if failures:
        atomic_json(
            failures,
            out / 'dryrun_openmc_errors.json',
        )

    group_metrics = {}
    for method, symmetry in group_order:
        key_name = f'{method}|symmetry={symmetry}'
        group_success = [
            item for item in successful
            if (
                item['solver'],
                bool(item['symmetry_enabled']),
            ) == (method, symmetry)
        ]
        group_failure = [
            item for item in failures
            if (
                item.get('solver'),
                bool(item.get('symmetry_enabled')),
            ) == (method, symmetry)
        ]
        total_times = [
            item['total_seconds'] for item in group_success
        ]
        openmc_times = [
            item['openmc_seconds'] for item in group_success
        ]
        group_metrics[key_name] = {
            'solver': method,
            'symmetry_enabled': symmetry,
            'requested_samples': args.samples_per_group,
            'successful_samples': len(group_success),
            'failed_samples': len(group_failure),
            'mean_total_seconds_per_sample': (
                float(np.mean(total_times))
                if total_times else None
            ),
            'max_total_seconds_per_sample': (
                float(np.max(total_times))
                if total_times else None
            ),
            'mean_openmc_seconds_per_sample': (
                float(np.mean(openmc_times))
                if openmc_times else None
            ),
            'max_openmc_seconds_per_sample': (
                float(np.max(openmc_times))
                if openmc_times else None
            ),
            # All 10 samples in a group could run concurrently under the
            # proven 16x8 layout. The slowest individual result is therefore
            # a useful approximate dedicated 10-sample wall time.
            'estimated_dedicated_10_sample_wall_seconds': (
                float(np.max(total_times))
                if total_times else None
            ),
            'aggregate_openmc_sample_seconds': (
                float(np.sum(openmc_times))
                if openmc_times else None
            ),
        }

    dry_histories = args.batches * args.particles
    full_histories = 500 * 500000
    history_ratio = full_histories / dry_histories

    # Diagnostic-only scaling. The user's measured 14.83 billed-h/sample
    # remains the production planning baseline.
    rough_full_50_seconds = (
        total_elapsed
        * history_ratio
        * (50.0 / len(records))
    )

    metrics = {
        'phase': 'openmc',
        'environment': environment_record(),
        'pattern_file': str(pattern_file),
        'requested_samples': len(records),
        'successful_samples': len(successful),
        'failed_samples': len(failures),
        'openmc_workers': args.openmc_workers,
        'threads_per_run': args.threads_per_run,
        'slot_thread_counts': slot_thread_counts,
        'cpu_threads_used': used,
        'cpu_threads_exposed': len(allowed),
        'dry_settings': {
            'batches': args.batches,
            'inactive': args.inactive,
            'particles': args.particles,
            'histories_per_sample': dry_histories,
        },
        'full_settings': {
            'batches': 500,
            'inactive': 50,
            'particles': 500000,
            'histories_per_sample': full_histories,
        },
        'history_ratio_full_over_dry': history_ratio,
        'waves': waves,
        'groups': group_metrics,
        'total_elapsed_seconds': total_elapsed,
        'rough_history_scaled_full_50_sample_seconds': (
            rough_full_50_seconds
        ),
        'rough_history_scaled_full_50_sample_hours': (
            rough_full_50_seconds / 3600.0
        ),
        'warning': (
            'The history-scaled estimate is diagnostic only. '
            'Use the existing measured 14.83 billed-h/sample '
            'for production budgeting until a full-fidelity '
            'benchmark with the new patterns is completed.'
        ),
    }
    atomic_json(metrics, out / 'openmc_metrics.json')

    print('\nOpenMC phase complete:')
    print(json.dumps(metrics, indent=2))
    if failures:
        raise RuntimeError(
            f'{len(failures)} of {len(records)} '
            'OpenMC dry-run samples failed'
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='phase', required=True)

    solver = subparsers.add_parser('solver', help='Generate SA/SQA patterns per symmetry setting')
    solver.add_argument('--output-dir', required=True)
    solver.add_argument('--pattern-file', help='Final SA/SQA pattern-bank JSON path')
    solver.add_argument('--samples-per-group', type=int, default=10)
    solver.add_argument('--sa-workers', type=int, default=48)
    solver.add_argument('--sqa-workers', type=int, default=48)
    solver.add_argument('--max-batches-per-group', type=int, default=5000)
    solver.add_argument('--base-seed', type=int, default=20260718)
    solver.add_argument('--resume', action='store_true')

    openmc_parser = subparsers.add_parser('openmc', help='Run reduced OpenMC on the generated patterns')
    openmc_parser.add_argument('--output-dir', required=True)
    openmc_parser.add_argument('--pattern-file')
    openmc_parser.add_argument('--samples-per-group', type=int, default=10)
    openmc_parser.add_argument('--openmc-workers', type=int, default=5)
    openmc_parser.add_argument('--threads-per-run', type=int, default=9)
    openmc_parser.add_argument('--batches', type=int, default=50)
    openmc_parser.add_argument('--inactive', type=int, default=10)
    openmc_parser.add_argument('--particles', type=int, default=20000)
    return parser


def main():
    args = build_parser().parse_args()
    if args.phase == 'solver':
        run_solver_phase(args)
    elif args.phase == 'openmc':
        if args.inactive >= args.batches:
            raise ValueError('inactive must be smaller than batches')
        run_openmc_phase(args)
    else:
        raise AssertionError(args.phase)


if __name__ == '__main__':
    main()
