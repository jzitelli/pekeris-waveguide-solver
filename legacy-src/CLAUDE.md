# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fortran/C library implementing the exact solution for the **Pekeris waveguide** - a classical two-layer acoustic waveguide model used in underwater acoustics. The solver computes:
- Complex acoustic pressure field at arbitrary positions
- Pressure gradients (radial and vertical derivatives)
- Discrete normal modes (trapped modes) and continuous spectrum contributions

The Pekeris model consists of an upper water layer with constant sound speed over a semi-infinite sediment layer.

## Building

No automated build system exists. This library is designed to be linked into external solvers (originally part of an hp2d finite element code).

**Dependencies:**
- Fortran compiler (gfortran or ifort)
- C compiler
- GNU Scientific Library (GSL) - for Brent root solver used in eigenvalue computation

**Compilation order** (modules have dependencies):
1. `pml.f90` - PML absorbing boundary conditions
2. `underwater_layers.f90` - layer property definitions
3. `user_routines.f90` - material property functions (uses underwater_layers, pml)
4. `underwater.f90` - problem setup module (uses underwater_layers, user_routines, pml)
5. `pekeris_lookup.f90` - result caching
6. `pekeris.f90` - main solver (uses underwater, pekeris_lookup)
7. `pekeris.c` - C eigenvalue solver (requires GSL)
8. `spfun.f`, `d1mach.f`, `r1mach.f`, `i1mach.f` - SLATEC/NETLIB special functions

**Example compile commands:**
```bash
gfortran -c pml.f90 underwater_layers.f90 user_routines.f90 underwater.f90 pekeris_lookup.f90 pekeris.f90 spfun.f d1mach.f r1mach.f i1mach.f
gcc -c pekeris.c -lgsl -lgslcblas
```

## Architecture

### Module Dependency Chain
```
pml
 └─► underwater_layers
      └─► user_routines
           └─► underwater
                └─► pekeris (main solver)
                     └─► pekeris_lookup (caching)
                          └─► pekeris.c (GSL eigenvalue solver)
```

### Key Public Interfaces

**`exact_pekeris(x, H, gradH)`** - Main entry point (`pekeris.f90:246`)
- `x(2)`: position (r, z) in physical coordinates
- `H`: complex pressure output
- `gradH(2)`: complex pressure gradient output
- Automatically initializes on first call

**`pekeris_moi(x, n_rays, p, grad_p)`** - Method of images for near-field (`pekeris.f90:421`)
- Used when r < 1.0 (near the source)
- `n_rays`: number of image sources to include

**`setPekerisParams_()` / `getEigenvalues_()`** - C interface (`pekeris.c`)
- Fortran-callable functions for eigenvalue computation via GSL Brent solver

### Physical Conventions
- Coordinate system: r = radial distance, z = depth (positive downward internally)
- Time dependence: exp(+iωt) convention (results are conjugated for this)
- Layer 1 = water (top), Layer 2 = sediment (bottom)
- Heights/depths use negative y-coordinate convention from FEM solver

### Numerical Methods
- QUADPACK adaptive quadrature (`dqagse`) for continuous spectrum integrals
- GSL Brent solver for discrete mode eigenvalues
- SLATEC Bessel functions (DBESJ0, DBESJ1, DBESY0, DBESY1, DBESK0, DBESK1)
- Lookup table caching (~4M entries) for computed values

### Key Parameters
- `quadrature_error`: integration tolerance (1e-7)
- `USE_DISCRETE_MODES_ONLY`: flag to skip continuous spectrum (faster but incomplete)
- `PML_*` variables: perfectly matched layer boundary parameters
