# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solves the **Pekeris waveguide problem** - a two-layer (water over sediment) acoustic waveguide model for underwater acoustics. The solution computes complex acoustic pressure fields and gradients in cylindrical coordinates (r, z) using the Helmholtz equation.

Three independent solver implementations exist:
- **Analytical** (`pekeris.py`): Closed-form solution with discrete modes + continuous spectrum
- **FEM** (`pekeris_fem.py` + `pekeris_gmsh.py`): DOLFINx/FEniCSx finite element solver with PML boundaries
- **JAX** (`pekeris_jax.py`): Discrete-modes-only solver with automatic differentiation support

## Running

```bash
# Analytical solver (core, no optional deps)
python pekeris.py [--discrete-only] [--no-plot] [--animate-field] [-j JOBS]

# FEM solver (requires FEniCSx + complex PETSc)
python pekeris_fem.py [--compare] [--no-viz] [--lc-fine 2.0] [--lc-coarse 15.0]

# Verification scripts (serve as regression tests)
python verify_lloyds_mirror.py    # FEM vs method-of-images (homogeneous half-space)
python verify_pekeris_gradients.py  # FEM gradients vs analytical Pekeris gradients
```

No automated test framework (pytest, etc.) - verification scripts are the tests.

## Dependencies

**Core:** numpy, scipy, matplotlib

**FEM (optional):** FEniCSx/DOLFINx with complex number support, gmsh, mpi4py, petsc4py. Requires `export PETSC_DIR=/usr/lib/petscdir/petsc-complex`. If using a venv, create it with `--system-site-packages` to access system FEniCSx packages.

**JAX (optional):** jax, jaxlib. Requires `jax.config.update("jax_enable_x64", True)` for sufficient precision.

## Architecture

### Analytical Solver (`pekeris.py`)

`PekerisWaveguide` class computes pressure via modal decomposition:
- **Discrete modes**: Eigenvalues found by `scipy.optimize.brentq` on the transcendental equation. Pressure uses Hankel functions H0^(2) = J0 - iY0.
- **Continuous spectrum**: Two QUADPACK adaptive integrals (definite region using J0/Y0, indefinite region using K0). Skipped when `discrete_modes_only=True`.
- **Parallelization**: `ProcessPoolExecutor` with module-level worker functions (`_compute_pressure_point`, `_compute_gradient_point`) for pickling compatibility.
- **Time convention**: exp(+iwt); results are conjugated.

### FEM Solver (`pekeris_fem.py` + `pekeris_gmsh.py`)

Axisymmetric weak form with cylindrical measure `r` in integration. Key aspects:
- **Mesh** (`pekeris_gmsh.py`): Gmsh OpenCASCADE kernel with boolean operations. Source singularity handled by excluding a semicircular disk at (0, z_s). Two characteristic lengths: `lc_fine` near source, `lc_coarse` far field.
- **Physical domains**: WATER_DOMAIN (tag 1), SEDIMENT_DOMAIN (tag 2), four PML regions (tags 3-6). Material properties are piecewise constant (DG0 function space).
- **PML**: Complex coordinate stretching tensors absorb outgoing waves at domain boundaries.
- **Boundaries**: Pressure-release (Dirichlet p=0) at top, normal velocity source BC on semicircle, symmetry on axis (r=0).
- **Solver**: PETSc with complex scalar type.

### JAX Solver (`pekeris_jax.py`)

Discrete-modes-only reimplementation using JAX for automatic differentiation. Custom VJP rules wrap scipy Bessel functions (Y0, Y1, K0, K1) via `jax.pure_callback`.

### Domain Tags (shared between `pekeris_gmsh.py` and `pekeris_fem.py`)

Physical surfaces: WATER=1, SEDIMENT=2, PML_WATER_RIGHT=3, PML_SEDIMENT_RIGHT=4, PML_BOTTOM=5, PML_CORNER=6. Boundaries: SOURCE=10, TOP=11, BOTTOM=12, RIGHT=13, INTERFACE=14, AXIS=15.

## Physics Conventions

- z is depth (positive downward)
- r is radial distance in cylindrical coordinates
- Layer 1 = water (top, sound speed c1, density rho1)
- Layer 2 = sediment (bottom, sound speed c2 > c1, density rho2)
- H = water layer depth; source at depth z_s

## Legacy Code

Original Fortran/C implementation in `legacy-src/` with its own `CLAUDE.md`. The Python solvers were generated from this code. Not actively developed but kept as reference.
