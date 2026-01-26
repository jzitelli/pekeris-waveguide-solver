"""
FEniCSx/DOLFINx finite element solver for the Pekeris waveguide.

This module implements an axisymmetric Helmholtz solver with:
- Variable density and sound speed (water and sediment layers)
- PML (Perfectly Matched Layer) absorbing boundaries
- Pressure-release boundary condition at the surface (z=0)
- Normal velocity source condition at the source boundary

The axisymmetric Helmholtz equation in (r, z) coordinates is:
    (1/r) d/dr(r/rho * dp/dr) + d/dz(1/rho * dp/dz) + omega^2/(rho*c^2) * p = 0

The weak form includes a factor of r in the measure due to cylindrical symmetry.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio

from pekeris_gmsh import (
    gmsh_pekeris_waveguide,
    WATER_DOMAIN, SEDIMENT_DOMAIN,
    PML_WATER_RIGHT, PML_SEDIMENT_RIGHT, PML_BOTTOM, PML_CORNER,
    SOURCE_BOUNDARY, TOP_BOUNDARY, AXIS_BOUNDARY,
)


def pml_stretch_r(r, alpha, k0, r_max, r_pml):
    """
    PML coordinate stretching in radial direction.

    For r > r_max, applies complex coordinate transformation:
        r' = r + j*alpha/k0 * ((r - r_max) / (r_pml - r_max))^2 * (r - r_max)

    Parameters
    ----------
    r : ufl expression
        Radial coordinate
    alpha : float
        PML absorption strength
    k0 : float
        Reference wavenumber
    r_max : float
        Start of PML region
    r_pml : float
        Total extent including PML
    """
    # Smooth ramp from r_max to r_pml
    delta = r_pml - r_max
    # Quadratic profile for PML absorption
    sigma = alpha / k0 * ((r - r_max) / delta) ** 2
    return r * (1 + 1j * sigma)


def pml_stretch_z(z, alpha, k0, z_max, z_pml):
    """
    PML coordinate stretching in z direction.

    For z > z_max, applies complex coordinate transformation.
    """
    delta = z_pml - z_max
    sigma = alpha / k0 * ((z - z_max) / delta) ** 2
    return z * (1 + 1j * sigma)


def create_pml_tensors(x, alpha, k0, r_max, z_max, pml_r, pml_z, region):
    """
    Create PML transformation tensors for the axisymmetric formulation.

    The PML is implemented via coordinate stretching, which results in
    modified material tensors. For the Helmholtz equation:
        nabla . (A * nabla p) + k^2 * B * p = 0

    where A and B are the PML tensors.

    Parameters
    ----------
    x : ufl.SpatialCoordinate
        Spatial coordinates (r, z)
    alpha : float
        PML absorption parameter
    k0 : float
        Reference wavenumber
    r_max, z_max : float
        Start of PML regions
    pml_r, pml_z : float
        PML thicknesses
    region : str
        One of 'r', 'z', or 'corner'

    Returns
    -------
    A_tensor : ufl matrix
        Tensor for gradient terms
    B_scalar : ufl scalar
        Scalar for mass term
    """
    r, z = x[0], x[1]
    r_pml = r_max + pml_r
    z_pml = z_max + pml_z

    # Stretching factors (complex)
    if region == 'r':
        # Only radial stretching
        delta_r = pml_r
        sigma_r = alpha / k0 * ((r - r_max) / delta_r) ** 2
        s_r = 1 + 1j * sigma_r
        s_z = 1.0
    elif region == 'z':
        # Only vertical stretching
        delta_z = pml_z
        sigma_z = alpha / k0 * ((z - z_max) / delta_z) ** 2
        s_r = 1.0
        s_z = 1 + 1j * sigma_z
    elif region == 'corner':
        # Both directions
        delta_r = pml_r
        delta_z = pml_z
        sigma_r = alpha / k0 * ((r - r_max) / delta_r) ** 2
        sigma_z = alpha / k0 * ((z - z_max) / delta_z) ** 2
        s_r = 1 + 1j * sigma_r
        s_z = 1 + 1j * sigma_z
    else:
        raise ValueError(f"Unknown PML region: {region}")

    # For axisymmetric Helmholtz, the PML tensors are:
    # A = diag(s_z/s_r, s_r/s_z) for the gradient
    # B = s_r * s_z for the mass term
    A_rr = s_z / s_r
    A_zz = s_r / s_z
    A_tensor = ufl.as_matrix([[A_rr, 0], [0, A_zz]])
    B_scalar = s_r * s_z

    return A_tensor, B_scalar


def solve_pekeris_fem(
    omega: float = 200.0,
    c1: float = 1500.0,
    c2: float = 1800.0,
    rho1: float = 1000.0,
    rho2: float = 1800.0,
    H: float = 150.0,
    z_s: float = 30.0,
    r_max: float = 1000.0,
    z_max: float = 225.0,
    r_source: float = 1.0,
    pml_r: float = 100.0,
    pml_z: float = 100.0,
    lc_fine: float = 2.0,
    lc_coarse: float = 20.0,
    degree: int = 2,
    pml_alpha: float = 2.0,
    source_velocity: float = 1.0,
):
    """
    Solve the Pekeris waveguide problem using FEM.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    c1 : float
        Sound speed in water (m/s)
    c2 : float
        Sound speed in sediment (m/s)
    rho1 : float
        Density of water (kg/m^3)
    rho2 : float
        Density of sediment (kg/m^3)
    H : float
        Water layer depth (m)
    z_s : float
        Source depth (m)
    r_max : float
        Maximum radial extent before PML (m)
    z_max : float
        Maximum depth before PML (m)
    r_source : float
        Radius of source exclusion (m)
    pml_r : float
        Radial PML thickness (m)
    pml_z : float
        Vertical PML thickness (m)
    lc_fine : float
        Fine mesh size near source (m)
    lc_coarse : float
        Coarse mesh size far from source (m)
    degree : int
        Polynomial degree for finite elements
    pml_alpha : float
        PML absorption strength parameter
    source_velocity : float
        Normal velocity at source boundary (m/s)

    Returns
    -------
    mesh_data : MeshData
        DOLFINx mesh with tags
    uh : Function
        Complex pressure solution
    params : dict
        Problem parameters
    """

    if not np.issubdtype(default_scalar_type, np.complexfloating):
        raise RuntimeError("This solver requires DOLFINx compiled with complex PETSc")

    # Reference wavenumber (water)
    k0 = omega / c1

    print(f"Pekeris FEM Solver")
    print(f"  omega = {omega} rad/s (f = {omega/(2*np.pi):.2f} Hz)")
    print(f"  c1 = {c1} m/s, c2 = {c2} m/s")
    print(f"  rho1 = {rho1} kg/m^3, rho2 = {rho2} kg/m^3")
    print(f"  H = {H} m, z_s = {z_s} m")
    print(f"  k0 = {k0:.6f} rad/m, wavelength = {2*np.pi/k0:.2f} m")
    print(f"  Mesh: degree={degree}, lc_fine={lc_fine}, lc_coarse={lc_coarse}")

    # Generate mesh using Gmsh
    print("\nGenerating mesh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    model = gmsh_pekeris_waveguide(
        model, "Pekeris",
        H=H, z_s=z_s, r_max=r_max, z_max=z_max,
        r_source=r_source, pml_r=pml_r, pml_z=pml_z,
        lc_fine=lc_fine, lc_coarse=lc_coarse,
    )

    # Convert to DOLFINx mesh
    mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()

    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    n_cells = msh.topology.index_map(2).size_global
    n_vertices = msh.topology.index_map(0).size_global
    print(f"  Mesh: {n_cells} cells, {n_vertices} vertices")

    # Create function space
    V = fem.functionspace(msh, ("Lagrange", degree))
    n_dofs = V.dofmap.index_map.size_global
    print(f"  DOFs: {n_dofs}")

    # Define material properties as piecewise functions
    # Use DG0 space for discontinuous properties
    Q = fem.functionspace(msh, ("DG", 0))

    # Sound speed
    c = fem.Function(Q)
    c.x.array[:] = c1  # Default to water
    sediment_cells = cell_tags.find(SEDIMENT_DOMAIN)
    c.x.array[sediment_cells] = c2
    # PML regions inherit from adjacent physical domain
    pml_water_cells = cell_tags.find(PML_WATER_RIGHT)
    c.x.array[pml_water_cells] = c1
    pml_sediment_cells = cell_tags.find(PML_SEDIMENT_RIGHT)
    c.x.array[pml_sediment_cells] = c2
    pml_bottom_cells = cell_tags.find(PML_BOTTOM)
    c.x.array[pml_bottom_cells] = c2  # Bottom PML is in sediment
    pml_corner_cells = cell_tags.find(PML_CORNER)
    c.x.array[pml_corner_cells] = c2  # Corner is sediment

    # Density
    rho = fem.Function(Q)
    rho.x.array[:] = rho1  # Default to water
    rho.x.array[sediment_cells] = rho2
    rho.x.array[pml_water_cells] = rho1
    rho.x.array[pml_sediment_cells] = rho2
    rho.x.array[pml_bottom_cells] = rho2
    rho.x.array[pml_corner_cells] = rho2

    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    r = x[0]  # Radial coordinate
    z = x[1]  # Depth coordinate (note: in mesh, z increases downward)

    # Wavenumber squared
    k_squared = (omega / c) ** 2

    # Define measures with subdomain data
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Physical domains (no PML)
    dx_water = dx(WATER_DOMAIN)
    dx_sediment = dx(SEDIMENT_DOMAIN)
    dx_physical = dx_water + dx_sediment

    # PML domains
    dx_pml_water_r = dx(PML_WATER_RIGHT)
    dx_pml_sediment_r = dx(PML_SEDIMENT_RIGHT)
    dx_pml_bottom = dx(PML_BOTTOM)
    dx_pml_corner = dx(PML_CORNER)

    # =========================================================================
    # Variational formulation
    # =========================================================================
    # Axisymmetric Helmholtz weak form (note factor of r in measure):
    #   integral[(1/rho) * grad(p) . grad(v) - k^2/rho * p * v] * r dr dz
    #   = integral[source terms] on boundaries
    #
    # In PML regions, we modify the equation using coordinate stretching.
    # =========================================================================

    # Physical domain contribution (water + sediment)
    # Note: ufl.inner(a, b) computes a . conj(b) in complex mode
    # For the bilinear form, we want: integral[(1/rho) * grad(p) . conj(grad(v)) - k^2/rho * p * conj(v)] r dr dz
    a_phys = (
        (1/rho) * ufl.inner(ufl.grad(p), ufl.grad(v)) * r * dx_physical
        - k_squared / rho * ufl.inner(p, v) * r * dx_physical
    )

    # PML contributions with coordinate stretching
    # Create PML tensors for each region
    A_pml_r, B_pml_r = create_pml_tensors(x, pml_alpha, k0, r_max, z_max, pml_r, pml_z, 'r')
    A_pml_z, B_pml_z = create_pml_tensors(x, pml_alpha, k0, r_max, z_max, pml_r, pml_z, 'z')
    A_pml_corner, B_pml_corner = create_pml_tensors(x, pml_alpha, k0, r_max, z_max, pml_r, pml_z, 'corner')

    # Water right PML
    a_pml_water_r = (
        (1/rho) * ufl.inner(A_pml_r * ufl.grad(p), ufl.grad(v)) * r * dx_pml_water_r
        - k_squared / rho * B_pml_r * ufl.inner(p, v) * r * dx_pml_water_r
    )

    # Sediment right PML
    a_pml_sediment_r = (
        (1/rho) * ufl.inner(A_pml_r * ufl.grad(p), ufl.grad(v)) * r * dx_pml_sediment_r
        - k_squared / rho * B_pml_r * ufl.inner(p, v) * r * dx_pml_sediment_r
    )

    # Bottom PML
    a_pml_bottom = (
        (1/rho) * ufl.inner(A_pml_z * ufl.grad(p), ufl.grad(v)) * r * dx_pml_bottom
        - k_squared / rho * B_pml_z * ufl.inner(p, v) * r * dx_pml_bottom
    )

    # Corner PML
    a_pml_corner = (
        (1/rho) * ufl.inner(A_pml_corner * ufl.grad(p), ufl.grad(v)) * r * dx_pml_corner
        - k_squared / rho * B_pml_corner * ufl.inner(p, v) * r * dx_pml_corner
    )

    # Total bilinear form
    a = a_phys + a_pml_water_r + a_pml_sediment_r + a_pml_bottom + a_pml_corner

    # =========================================================================
    # Boundary conditions
    # =========================================================================

    # Pressure-release at top surface (z=0): p = 0 (Dirichlet)
    top_facets = facet_tags.find(TOP_BOUNDARY)
    top_dofs = fem.locate_dofs_topological(V, 1, top_facets)
    bc_top = fem.dirichletbc(
        fem.Constant(msh, default_scalar_type(0.0)),
        top_dofs,
        V
    )

    # Normal velocity source at source boundary (Neumann)
    # The weak form naturally incorporates: integral[(1/rho) * dp/dn * v] ds
    # For prescribed normal velocity v_n, we have: (1/rho) * dp/dn = -j*omega*v_n
    # So the RHS contribution is: -j*omega*v_n * v * r ds

    # Source term (normal velocity boundary condition)
    # Note: The source is a semicircle at (r=0, z=z_s)
    # The normal points inward (toward the source), so v_n > 0 means outward flow
    # Use ufl.inner for proper complex conjugation of test function
    source_coeff = default_scalar_type(-1j * omega * rho1 * source_velocity)
    L = ufl.inner(fem.Constant(msh, source_coeff), v) * r * ds(SOURCE_BOUNDARY)

    # Axis boundary (r=0): natural boundary condition (dp/dr = 0 for axisymmetric)
    # This is automatically satisfied by the weak form

    bcs = [bc_top]

    # =========================================================================
    # Solve the linear system
    # =========================================================================
    print("\nAssembling and solving linear system...")

    # Check for available solvers
    sys = PETSc.Sys()
    use_superlu = PETSc.IntType == np.int64
    if sys.hasExternalPackage("mumps") and not use_superlu:
        solver_type = "mumps"
    elif sys.hasExternalPackage("superlu_dist"):
        solver_type = "superlu_dist"
    else:
        solver_type = "petsc"

    print(f"  Using solver: {solver_type}")

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="pekeris_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": solver_type,
            "ksp_error_if_not_converged": True,
        },
    )

    uh = problem.solve()
    uh.name = "pressure"

    print("  Solve complete.")

    # Compute some statistics
    p_max = np.max(np.abs(uh.x.array))
    print(f"  Max |p| = {p_max:.6e}")

    params = {
        'omega': omega,
        'c1': c1, 'c2': c2,
        'rho1': rho1, 'rho2': rho2,
        'H': H, 'z_s': z_s,
        'r_max': r_max, 'z_max': z_max,
        'r_source': r_source,
        'pml_r': pml_r, 'pml_z': pml_z,
        'k0': k0,
        'degree': degree,
    }

    return mesh_data, uh, params


def visualize_solution(mesh_data, uh, params, save_only=False, filename="pekeris_fem_field.png",
                       n_subdivisions=3):
    """
    Visualize the FEM solution using PyVista.

    Parameters
    ----------
    mesh_data : MeshData
        DOLFINx mesh data
    uh : Function
        Complex pressure solution
    params : dict
        Problem parameters
    save_only : bool
        If True, save to file without displaying
    filename : str
        Output filename for the plot
    n_subdivisions : int
        Number of subdivisions per element edge for higher-order visualization
    """
    try:
        import pyvista
        from dolfinx import plot
    except ImportError:
        print("PyVista not available for visualization")
        return

    if save_only:
        pyvista.OFF_SCREEN = True

    msh = mesh_data.mesh
    degree = params['degree']

    # For higher-order elements, we need to sample within each cell
    # Create a finer visualization by interpolating to a higher-order Lagrange space
    # and using VTK's capability to handle it

    # Create a finer Lagrange space for visualization
    # The number of subdivisions determines the visual fidelity
    viz_degree = max(degree, n_subdivisions)

    # Create VTK mesh with subdivision for proper higher-order visualization
    # This creates additional points within each cell
    V_viz = fem.functionspace(msh, ("Lagrange", viz_degree))
    p_viz = fem.Function(V_viz)
    p_viz.interpolate(uh)

    # Get the VTK mesh with higher-order points
    topology, cell_types, geometry = plot.vtk_mesh(V_viz)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Get nodal values from the visualization function
    p_values = p_viz.x.array

    # Add point data (values at nodes)
    grid.point_data["Re(p)"] = np.real(p_values)
    grid.point_data["Im(p)"] = np.imag(p_values)
    grid.point_data["|p|"] = np.abs(p_values)

    # Compute dB scale (relative to max)
    p_abs = np.abs(p_values)
    p_max = np.max(p_abs)
    p_db = 20 * np.log10(p_abs / p_max + 1e-20)
    # Clamp dB values before tessellation to avoid interpolation artifacts
    # at the pressure-release boundary where p→0 gives p_db→-∞
    p_db = np.clip(p_db, -60, 0)
    grid.point_data["p (dB)"] = p_db

    # Subdivide the grid for smooth visualization of higher-order data
    # This tessellates curved elements into linear pieces for rendering
    grid = grid.tessellate()

    # Create figure with multiple views
    H = params['H']
    z_s = params['z_s']
    r_max = params['r_max']
    z_max = params['z_max']
    pml_r = params['pml_r']
    pml_z = params['pml_z']

    plotter = pyvista.Plotter(shape=(2, 2), window_size=[1600, 1200])

    # Plot 1: Magnitude in dB
    plotter.subplot(0, 0)
    plotter.add_text("|p| (dB re max)", font_size=12)
    plotter.add_mesh(grid.copy(), scalars="p (dB)", cmap='viridis', clim=[-60, 0], show_edges=False)
    # Add interface line
    plotter.add_mesh(
        pyvista.Line([0, H, 0], [r_max + pml_r, H, 0]),
        color='white', line_width=2
    )
    # Add source depth marker
    plotter.add_mesh(
        pyvista.Line([0, z_s, 0], [50, z_s, 0]),
        color='red', line_width=2
    )
    plotter.view_xy()
    plotter.camera.SetViewUp(0, -1, 0)  # Flip to have depth increasing downward

    # Plot 2: Real part
    plotter.subplot(0, 1)
    plotter.add_text("Re(p)", font_size=12)
    vlim = np.max(np.abs(np.real(p_values))) * 0.1
    plotter.add_mesh(grid.copy(), scalars="Re(p)", cmap='RdBu_r', clim=[-vlim, vlim], show_edges=False)
    plotter.add_mesh(
        pyvista.Line([0, H, 0], [r_max + pml_r, H, 0]),
        color='black', line_width=2
    )
    plotter.view_xy()
    plotter.camera.SetViewUp(0, -1, 0)

    # Plot 3: Imaginary part
    plotter.subplot(1, 0)
    plotter.add_text("Im(p)", font_size=12)
    vlim_im = np.max(np.abs(np.imag(p_values))) * 0.1
    plotter.add_mesh(grid.copy(), scalars="Im(p)", cmap='RdBu_r', clim=[-vlim_im, vlim_im], show_edges=False)
    plotter.add_mesh(
        pyvista.Line([0, H, 0], [r_max + pml_r, H, 0]),
        color='black', line_width=2
    )
    plotter.view_xy()
    plotter.camera.SetViewUp(0, -1, 0)

    # Plot 4: Absolute value
    plotter.subplot(1, 1)
    plotter.add_text("|p|", font_size=12)
    plotter.add_mesh(grid.copy(), scalars="|p|", cmap='hot', show_edges=False)
    plotter.add_mesh(
        pyvista.Line([0, H, 0], [r_max + pml_r, H, 0]),
        color='cyan', line_width=2
    )
    plotter.view_xy()
    plotter.camera.SetViewUp(0, -1, 0)

    # Add title
    freq = params['omega'] / (2 * np.pi)
    plotter.add_text(
        f"Pekeris FEM: f={freq:.1f} Hz, H={H}m, c1={params['c1']}m/s, c2={params['c2']}m/s",
        position='upper_edge', font_size=14
    )

    if save_only:
        plotter.screenshot(filename)
        print(f"Saved visualization to {filename}")
    else:
        plotter.show()

    plotter.close()


def extract_field_on_grid(mesh_data, uh, r_array, z_array):
    """
    Extract the FEM solution on a regular grid for comparison.

    Parameters
    ----------
    mesh_data : MeshData
        DOLFINx mesh data
    uh : Function
        Complex pressure solution
    r_array : np.ndarray
        1D array of radial coordinates
    z_array : np.ndarray
        1D array of depth coordinates

    Returns
    -------
    P : np.ndarray
        2D complex array of pressure, shape (len(z_array), len(r_array))
    """
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    msh = mesh_data.mesh
    gdim = msh.geometry.dim

    # Create grid points (use 2D or 3D depending on mesh)
    nr, nz = len(r_array), len(z_array)
    if gdim == 2:
        points = np.zeros((nr * nz, 3))  # bb_tree requires 3D points
        for i, z in enumerate(z_array):
            for j, r in enumerate(r_array):
                points[i * nr + j, :2] = [r, z]
    else:
        points = np.zeros((nr * nz, 3))
        for i, z in enumerate(z_array):
            for j, r in enumerate(r_array):
                points[i * nr + j, :] = [r, z, 0]

    # Build bounding box tree for efficient point location
    tree = bb_tree(msh, msh.topology.dim)

    # Find cells containing points
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(msh, cell_candidates, points)

    # Evaluate solution at points
    P = np.zeros((nz, nr), dtype=complex)

    for i, z in enumerate(z_array):
        for j, r in enumerate(r_array):
            idx = i * nr + j
            cells = colliding_cells.links(idx)
            if len(cells) > 0:
                # Evaluate at this point - use appropriate dimensionality
                if gdim == 2:
                    point = np.array([[r, z, 0.0]])  # eval expects 3D
                else:
                    point = np.array([[r, z, 0.0]])
                cell = cells[0]
                try:
                    P[i, j] = uh.eval(point, [cell])[0]
                except Exception:
                    P[i, j] = np.nan
            else:
                P[i, j] = np.nan

    return P


def compare_with_analytical(mesh_data, uh, params, nr=200, nz=100, r_min=10.0):
    """
    Compare FEM solution with analytical Pekeris solution.

    Parameters
    ----------
    mesh_data : MeshData
        DOLFINx mesh data
    uh : Function
        Complex pressure solution
    params : dict
        Problem parameters
    nr, nz : int
        Number of grid points for comparison
    r_min : float
        Minimum radial distance (avoid source singularity)

    Returns
    -------
    comparison : dict
        Dictionary with comparison results
    """
    from pekeris import PekerisWaveguide

    # Extract parameters
    omega = params['omega']
    c1, c2 = params['c1'], params['c2']
    rho1, rho2 = params['rho1'], params['rho2']
    H = params['H']
    z_s = params['z_s']
    r_max = params['r_max']

    # Create analytical solution
    print("\nComputing analytical solution for comparison...")
    wg = PekerisWaveguide(omega, c1, c2, rho1, rho2, H, z_s, discrete_modes_only=False)
    print(f"  Number of discrete modes: {wg.n_modes}")

    # Grid for comparison (stay within physical domain, not PML)
    r_array = np.linspace(r_min, r_max * 0.9, nr)
    z_array = np.linspace(1.0, H - 1.0, nz)  # Stay in water layer for now

    print(f"  Extracting FEM solution on {nr}x{nz} grid...")
    P_fem = extract_field_on_grid(mesh_data, uh, r_array, z_array)

    print(f"  Computing analytical solution...")
    P_analytical = wg.pressure_field(r_array, z_array, show_progress=True)

    # The FEM solution needs scaling to match the analytical solution
    # The analytical solution is for a point source with specific normalization
    # We need to find the appropriate scaling factor

    # Use a least-squares fit to find the best scaling
    valid_mask = ~np.isnan(P_fem) & ~np.isnan(P_analytical) & (np.abs(P_analytical) > 1e-20)

    if np.sum(valid_mask) > 0:
        # Complex scaling factor: P_fem = scale * P_analytical
        # Minimize |P_fem - scale * P_analytical|^2
        P_fem_valid = P_fem[valid_mask]
        P_anal_valid = P_analytical[valid_mask]

        # Least squares: scale = (P_anal^H * P_fem) / (P_anal^H * P_anal)
        scale = np.vdot(P_anal_valid, P_fem_valid) / np.vdot(P_anal_valid, P_anal_valid)

        # Compute scaled error
        P_fem_scaled = P_fem / scale
        error = np.abs(P_fem_scaled - P_analytical)
        relative_error = error / (np.abs(P_analytical) + 1e-20)

        mean_rel_error = np.nanmean(relative_error[valid_mask])
        max_rel_error = np.nanmax(relative_error[valid_mask])

        print(f"\nComparison results:")
        print(f"  Scaling factor: {scale:.6e}")
        print(f"  Mean relative error: {mean_rel_error:.2%}")
        print(f"  Max relative error: {max_rel_error:.2%}")
    else:
        print("Warning: No valid points for comparison")
        scale = 1.0
        mean_rel_error = np.nan
        max_rel_error = np.nan
        P_fem_scaled = P_fem

    return {
        'r_array': r_array,
        'z_array': z_array,
        'P_fem': P_fem,
        'P_fem_scaled': P_fem_scaled,
        'P_analytical': P_analytical,
        'scale': scale,
        'mean_relative_error': mean_rel_error,
        'max_relative_error': max_rel_error,
        'waveguide': wg,
    }


def plot_comparison(comparison, params, save_only=False, filename="pekeris_comparison.png"):
    """
    Plot comparison between FEM and analytical solutions.
    """
    import matplotlib
    if save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r_array = comparison['r_array']
    z_array = comparison['z_array']
    P_fem = comparison['P_fem_scaled']
    P_analytical = comparison['P_analytical']
    H = params['H']
    z_s = params['z_s']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Magnitude comparison
    P_fem_db = 20 * np.log10(np.abs(P_fem) / np.nanmax(np.abs(P_fem)) + 1e-20)
    P_anal_db = 20 * np.log10(np.abs(P_analytical) / np.nanmax(np.abs(P_analytical)) + 1e-20)

    ax = axes[0, 0]
    im = ax.pcolormesh(r_array, z_array, P_fem_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(H, color='white', linestyle='--', linewidth=1)
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1)
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM |p| (dB re max)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.pcolormesh(r_array, z_array, P_anal_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(H, color='white', linestyle='--', linewidth=1)
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1)
    ax.set_title('Analytical |p| (dB re max)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    diff_db = np.abs(P_fem_db - P_anal_db)
    im = ax.pcolormesh(r_array, z_array, diff_db, shading='auto', cmap='hot', vmin=0, vmax=10)
    ax.axhline(H, color='cyan', linestyle='--', linewidth=1)
    ax.set_title('|Difference| (dB)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    # Row 2: Real part comparison
    vlim = max(np.nanmax(np.abs(np.real(P_fem))), np.nanmax(np.abs(np.real(P_analytical)))) * 0.05

    ax = axes[1, 0]
    im = ax.pcolormesh(r_array, z_array, np.real(P_fem), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.axhline(H, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM Re(p)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.pcolormesh(r_array, z_array, np.real(P_analytical), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.axhline(H, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Range (m)')
    ax.set_title('Analytical Re(p)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    rel_error = np.abs(P_fem - P_analytical) / (np.abs(P_analytical) + 1e-20)
    im = ax.pcolormesh(r_array, z_array, rel_error * 100, shading='auto', cmap='hot', vmin=0, vmax=50)
    ax.axhline(H, color='cyan', linestyle='--', linewidth=1)
    ax.set_xlabel('Range (m)')
    ax.set_title('Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    freq = params['omega'] / (2 * np.pi)
    plt.suptitle(
        f"Pekeris Waveguide: FEM vs Analytical (f={freq:.1f} Hz)\n"
        f"Mean error: {comparison['mean_relative_error']:.1%}, Max error: {comparison['max_relative_error']:.1%}",
        fontsize=12
    )
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    print(f"Saved comparison plot to {filename}")

    if not save_only:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pekeris waveguide FEM solver")
    parser.add_argument("--omega", type=float, default=200.0, help="Angular frequency (rad/s)")
    parser.add_argument("--H", type=float, default=150.0, help="Water depth (m)")
    parser.add_argument("--z-s", type=float, default=30.0, help="Source depth (m)")
    parser.add_argument("--r-max", type=float, default=500.0, help="Max radial extent (m)")
    parser.add_argument("--z-max", type=float, default=225.0, help="Max depth (m)")
    parser.add_argument("--pml-r", type=float, default=100.0, help="Radial PML thickness (m)")
    parser.add_argument("--pml-z", type=float, default=100.0, help="Vertical PML thickness (m)")
    parser.add_argument("--lc-fine", type=float, default=2.0, help="Fine mesh size (m)")
    parser.add_argument("--lc-coarse", type=float, default=15.0, help="Coarse mesh size (m)")
    parser.add_argument("--degree", type=int, default=3, help="FEM polynomial degree")
    parser.add_argument("--pml-alpha", type=float, default=2.0, help="PML strength")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--compare", action="store_true", help="Compare with analytical solution")
    parser.add_argument("--save-only", action="store_true", help="Save plots without displaying")
    args = parser.parse_args()

    # Solve the problem
    mesh_data, uh, params = solve_pekeris_fem(
        omega=args.omega,
        H=args.H,
        z_s=args.z_s,
        r_max=args.r_max,
        z_max=args.z_max,
        pml_r=args.pml_r,
        pml_z=args.pml_z,
        lc_fine=args.lc_fine,
        lc_coarse=args.lc_coarse,
        degree=args.degree,
        pml_alpha=args.pml_alpha,
    )

    # Visualize
    if not args.no_viz:
        visualize_solution(mesh_data, uh, params, save_only=args.save_only)

    # Compare with analytical solution
    if args.compare:
        comparison = compare_with_analytical(mesh_data, uh, params)
        plot_comparison(comparison, params, save_only=args.save_only)
