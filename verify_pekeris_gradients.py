"""
Verification of FEM solver gradients against analytical Pekeris solution.

This script computes the L2 and H1 errors between the FEM solution and the
analytical Pekeris waveguide solution in the water layer. The H1 error
includes gradient comparisons.
"""

import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dolfinx import fem, default_scalar_type
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

from pekeris import PekerisWaveguide
from pekeris_gmsh import WATER_DOMAIN


def extract_fem_solution_and_gradient(mesh_data, uh, r_array, z_array):
    """
    Extract FEM solution and its gradient on a regular grid.

    Returns
    -------
    P : ndarray
        Pressure values, shape (nz, nr)
    grad_r, grad_z : ndarray
        Gradient components, shape (nz, nr)
    """
    msh = mesh_data.mesh

    # Create grid points
    nr, nz = len(r_array), len(z_array)
    points = np.zeros((nr * nz, 3))
    for i, z in enumerate(z_array):
        for j, r in enumerate(r_array):
            points[i * nr + j, :2] = [r, z]

    # Build bounding box tree
    tree = bb_tree(msh, msh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(msh, cell_candidates, points)

    # Compute gradient as a function
    element = uh.function_space.ufl_element()
    degree = element.degree
    V_grad = fem.functionspace(msh, ("DG", max(degree - 1, 0), (2,)))
    grad_uh = fem.Function(V_grad)
    grad_expr = fem.Expression(ufl.grad(uh), V_grad.element.interpolation_points)
    grad_uh.interpolate(grad_expr)

    # Extract values
    P = np.zeros((nz, nr), dtype=complex)
    grad_r = np.zeros((nz, nr), dtype=complex)
    grad_z = np.zeros((nz, nr), dtype=complex)

    for i, z in enumerate(z_array):
        for j, r in enumerate(r_array):
            idx = i * nr + j
            cells = colliding_cells.links(idx)
            if len(cells) > 0:
                point = np.array([[r, z, 0.0]])
                cell = cells[0]
                try:
                    P[i, j] = uh.eval(point, [cell])[0]
                    grad_val = grad_uh.eval(point, [cell])
                    grad_r[i, j] = grad_val[0]
                    grad_z[i, j] = grad_val[1]
                except Exception:
                    P[i, j] = np.nan
                    grad_r[i, j] = np.nan
                    grad_z[i, j] = np.nan
            else:
                P[i, j] = np.nan
                grad_r[i, j] = np.nan
                grad_z[i, j] = np.nan

    return P, grad_r, grad_z


def compute_analytical_field_and_gradient(wg, r_array, z_array, show_progress=True, n_jobs=None):
    """
    Compute analytical Pekeris solution and gradients on a grid.

    Uses parallelized field computation methods from PekerisWaveguide.

    Returns
    -------
    P : ndarray
        Pressure values, shape (nz, nr)
    grad_r, grad_z : ndarray
        Gradient components, shape (nz, nr)
    """
    # Use parallelized methods
    P = wg.pressure_field(r_array, z_array, show_progress=show_progress, n_jobs=n_jobs)
    grad_r, grad_z = wg.pressure_gradient_field(r_array, z_array, show_progress=show_progress, n_jobs=n_jobs)

    return P, grad_r, grad_z


def find_scaling_factor(P_fem, P_analytical, mask):
    """
    Find complex scaling factor: P_fem ≈ scale * P_analytical
    """
    P_fem_valid = P_fem[mask]
    P_anal_valid = P_analytical[mask]

    # Least squares: scale = (P_anal^H * P_fem) / (P_anal^H * P_anal)
    scale = np.vdot(P_anal_valid, P_fem_valid) / np.vdot(P_anal_valid, P_anal_valid)
    return scale


def compute_errors_on_mesh(mesh_data, uh, wg, scale):
    """
    Compute L2 and H1 errors by integrating over the mesh.

    This integrates over the physical water domain only (not PML or sediment).
    The analytical solution is evaluated using UFL expressions for the discrete modes.
    """
    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags

    # Get function space
    V = uh.function_space

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    r, z = x[0], x[1]

    # Build UFL expression for analytical solution (discrete modes only for simplicity)
    # This is complex because the full Pekeris solution includes integrals
    # Instead, we'll interpolate the analytical solution onto the mesh

    # Create a function to hold the analytical solution
    p_analytical_func = fem.Function(V)
    grad_r_analytical_func = fem.Function(V)
    grad_z_analytical_func = fem.Function(V)

    # Get DOF coordinates
    V_dofs = V.tabulate_dof_coordinates()

    # Evaluate analytical solution at DOF points
    p_vals = np.zeros(len(V_dofs), dtype=complex)
    for i, coord in enumerate(V_dofs):
        ri, zi = coord[0], coord[1]
        if ri > 0.5 and zi <= wg.H:  # Only in water layer, away from axis
            p_vals[i] = scale * wg.pressure(ri, zi)
        else:
            p_vals[i] = 0.0

    p_analytical_func.x.array[:] = p_vals

    # Error in pressure
    error_p = uh - p_analytical_func

    # Define measure restricted to water domain only
    dx_water = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)(WATER_DOMAIN)

    # L2 norm with axisymmetric weighting
    L2_error_sq = fem.form(ufl.inner(error_p, error_p) * r * dx_water)
    L2_norm_sq = fem.form(ufl.inner(p_analytical_func, p_analytical_func) * r * dx_water)

    L2_error = np.sqrt(np.abs(fem.assemble_scalar(L2_error_sq)))
    L2_norm = np.sqrt(np.abs(fem.assemble_scalar(L2_norm_sq)))

    # H1 semi-norm (gradient only)
    grad_error = ufl.grad(error_p)
    H1_semi_error_sq = fem.form(ufl.inner(grad_error, grad_error) * r * dx_water)

    grad_analytical = ufl.grad(p_analytical_func)
    H1_semi_norm_sq = fem.form(ufl.inner(grad_analytical, grad_analytical) * r * dx_water)

    H1_semi_error = np.sqrt(np.abs(fem.assemble_scalar(H1_semi_error_sq)))
    H1_semi_norm = np.sqrt(np.abs(fem.assemble_scalar(H1_semi_norm_sq)))

    # Full H1 norm
    H1_error = np.sqrt(L2_error**2 + H1_semi_error**2)
    H1_norm = np.sqrt(L2_norm**2 + H1_semi_norm**2)

    return {
        'L2_error': L2_error,
        'L2_norm': L2_norm,
        'L2_relative': L2_error / L2_norm if L2_norm > 0 else np.inf,
        'H1_semi_error': H1_semi_error,
        'H1_semi_norm': H1_semi_norm,
        'H1_semi_relative': H1_semi_error / H1_semi_norm if H1_semi_norm > 0 else np.inf,
        'H1_error': H1_error,
        'H1_norm': H1_norm,
        'H1_relative': H1_error / H1_norm if H1_norm > 0 else np.inf,
    }


def main():
    from pekeris_fem import solve_pekeris_fem

    # Problem parameters (matching pekeris.py example)
    omega = 200.0       # rad/s
    c1 = 1500.0         # m/s (water)
    c2 = 1800.0         # m/s (sediment)
    rho1 = 1000.0       # kg/m³ (water)
    rho2 = 1800.0       # kg/m³ (sediment)
    H = 150.0           # m (water depth)
    z_s = 30.0          # m (source depth)
    r_max = 500.0       # domain extent
    z_max = 225.0

    k1 = omega / c1
    wavelength = 2 * np.pi / k1

    print("=" * 60)
    print("Pekeris Waveguide Gradient Verification")
    print("=" * 60)
    print(f"  omega = {omega} rad/s (f = {omega/(2*np.pi):.2f} Hz)")
    print(f"  c1 = {c1} m/s, c2 = {c2} m/s")
    print(f"  rho1 = {rho1} kg/m³, rho2 = {rho2} kg/m³")
    print(f"  H = {H} m, z_s = {z_s} m")
    print(f"  k1 = {k1:.6f} rad/m, wavelength = {wavelength:.2f} m")
    print()

    # Create analytical Pekeris waveguide
    print("Creating analytical Pekeris waveguide...")
    wg = PekerisWaveguide(omega, c1, c2, rho1, rho2, H, z_s, discrete_modes_only=False)
    print(f"  Number of discrete modes: {wg.n_modes}")

    # Solve FEM
    print("\nRunning FEM solver...")
    mesh_data, uh, params = solve_pekeris_fem(
        omega=omega,
        c1=c1, c2=c2,
        rho1=rho1, rho2=rho2,
        H=H,
        z_s=z_s,
        r_max=r_max,
        z_max=z_max,
        pml_r=100.0,
        pml_z=100.0,
        lc_fine=2.0,
        lc_coarse=15.0,
        degree=4,
        pml_alpha=2.0,
    )

    # Create comparison grid (stay in water layer, avoid source region)
    r_min = 10.0  # Stay away from source
    nr, nz = 80, 60
    r_array = np.linspace(r_min, r_max * 0.9, nr)
    z_array = np.linspace(5.0, H - 5.0, nz)  # Stay in water layer

    R_grid, Z_grid = np.meshgrid(r_array, z_array)

    print("\nExtracting FEM solution and gradients on comparison grid...")
    P_fem, grad_r_fem, grad_z_fem = extract_fem_solution_and_gradient(
        mesh_data, uh, r_array, z_array
    )

    print("Computing analytical Pekeris solution and gradients...")
    P_analytical, grad_r_analytical, grad_z_analytical = compute_analytical_field_and_gradient(
        wg, r_array, z_array, show_progress=True
    )

    # Find scaling factor
    valid_mask = ~np.isnan(P_fem) & (np.abs(P_analytical) > 1e-20)
    scale = find_scaling_factor(P_fem, P_analytical, valid_mask)
    print(f"\nScaling factor: {scale:.6e}")
    print(f"  |scale| = {np.abs(scale):.6e}")
    print(f"  phase(scale) = {np.angle(scale) * 180/np.pi:.2f} degrees")

    # Scale analytical solution to match FEM
    P_analytical_scaled = scale * P_analytical
    grad_r_analytical_scaled = scale * grad_r_analytical
    grad_z_analytical_scaled = scale * grad_z_analytical

    # Compute point-wise errors
    error_p = np.abs(P_fem - P_analytical_scaled)
    rel_error_p = error_p / (np.abs(P_analytical_scaled) + 1e-20)

    error_grad_r = np.abs(grad_r_fem - grad_r_analytical_scaled)
    error_grad_z = np.abs(grad_z_fem - grad_z_analytical_scaled)
    error_grad = np.sqrt(error_grad_r**2 + error_grad_z**2)
    grad_mag_analytical = np.sqrt(np.abs(grad_r_analytical_scaled)**2 + np.abs(grad_z_analytical_scaled)**2)
    rel_error_grad = error_grad / (grad_mag_analytical + 1e-20)

    print("\n" + "=" * 60)
    print("Point-wise Errors (on comparison grid in water layer)")
    print("=" * 60)
    print(f"  Pressure:")
    print(f"    Mean relative error: {np.nanmean(rel_error_p[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_error_p[valid_mask]):.2%}")
    print(f"    RMS relative error:  {np.sqrt(np.nanmean(rel_error_p[valid_mask]**2)):.2%}")
    print(f"  Gradient (dp/dr):")
    rel_err_gr = np.abs(grad_r_fem - grad_r_analytical_scaled) / (np.abs(grad_r_analytical_scaled) + 1e-20)
    print(f"    Mean relative error: {np.nanmean(rel_err_gr[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_err_gr[valid_mask]):.2%}")
    print(f"  Gradient (dp/dz):")
    rel_err_gz = np.abs(grad_z_fem - grad_z_analytical_scaled) / (np.abs(grad_z_analytical_scaled) + 1e-20)
    print(f"    Mean relative error: {np.nanmean(rel_err_gz[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_err_gz[valid_mask]):.2%}")
    print(f"  Gradient (total |∇p|):")
    print(f"    Mean relative error: {np.nanmean(rel_error_grad[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_error_grad[valid_mask]):.2%}")

    # Compute integrated errors over the mesh
    print("\n" + "=" * 60)
    print("Integrated Errors (over water domain, excluding PML)")
    print("=" * 60)

    errors = compute_errors_on_mesh(mesh_data, uh, wg, scale)
    print(f"  L2 error:")
    print(f"    Absolute: {errors['L2_error']:.6e}")
    print(f"    Relative: {errors['L2_relative']:.2%}")
    print(f"  H1 semi-norm error (gradient only):")
    print(f"    Absolute: {errors['H1_semi_error']:.6e}")
    print(f"    Relative: {errors['H1_semi_relative']:.2%}")
    print(f"  H1 error (full):")
    print(f"    Absolute: {errors['H1_error']:.6e}")
    print(f"    Relative: {errors['H1_relative']:.2%}")

    # Create visualization
    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    # Row 1: Pressure magnitude comparison
    P_fem_db = 20 * np.log10(np.abs(P_fem) / np.nanmax(np.abs(P_fem)) + 1e-20)
    P_anal_db = 20 * np.log10(np.abs(P_analytical_scaled) / np.nanmax(np.abs(P_analytical_scaled)) + 1e-20)

    ax = axes[0, 0]
    im = ax.pcolormesh(r_array, z_array, P_fem_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(H, color='white', linestyle='--', linewidth=1, label='Seafloor')
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1, label=f'Source z={z_s}m')
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM |p| (dB re max)')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.pcolormesh(r_array, z_array, P_anal_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(H, color='white', linestyle='--', linewidth=1)
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1)
    ax.set_title('Analytical |p| (dB re max)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.pcolormesh(r_array, z_array, rel_error_p * 100, shading='auto', cmap='hot', vmin=0, vmax=20)
    ax.set_title('Pressure Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    # Row 2: Real part comparison
    vlim = np.nanmax(np.abs(np.real(P_fem))) * 0.1

    ax = axes[1, 0]
    im = ax.pcolormesh(r_array, z_array, np.real(P_fem), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.axhline(H, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM Re(p)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.pcolormesh(r_array, z_array, np.real(P_analytical_scaled), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.axhline(H, color='black', linestyle='--', linewidth=1)
    ax.set_title('Analytical Re(p)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    diff_real = np.abs(np.real(P_fem) - np.real(P_analytical_scaled))
    vlim_diff = np.nanmax(diff_real) * 0.5
    im = ax.pcolormesh(r_array, z_array, diff_real, shading='auto', cmap='hot', vmin=0, vmax=vlim_diff)
    ax.set_title('|Re(p) difference|')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    # Row 3: Radial gradient comparison
    vlim_gr = np.nanmax(np.abs(np.real(grad_r_fem))) * 0.3

    ax = axes[2, 0]
    im = ax.pcolormesh(r_array, z_array, np.real(grad_r_fem), shading='auto', cmap='RdBu_r', vmin=-vlim_gr, vmax=vlim_gr)
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM Re(dp/dr)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[2, 1]
    im = ax.pcolormesh(r_array, z_array, np.real(grad_r_analytical_scaled), shading='auto', cmap='RdBu_r', vmin=-vlim_gr, vmax=vlim_gr)
    ax.set_title('Analytical Re(dp/dr)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[2, 2]
    im = ax.pcolormesh(r_array, z_array, rel_err_gr * 100, shading='auto', cmap='hot', vmin=0, vmax=50)
    ax.set_title('dp/dr Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    # Row 4: Vertical gradient comparison
    vlim_gz = np.nanmax(np.abs(np.real(grad_z_fem))) * 0.3

    ax = axes[3, 0]
    im = ax.pcolormesh(r_array, z_array, np.real(grad_z_fem), shading='auto', cmap='RdBu_r', vmin=-vlim_gz, vmax=vlim_gz)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM Re(dp/dz)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[3, 1]
    im = ax.pcolormesh(r_array, z_array, np.real(grad_z_analytical_scaled), shading='auto', cmap='RdBu_r', vmin=-vlim_gz, vmax=vlim_gz)
    ax.set_xlabel('Range (m)')
    ax.set_title('Analytical Re(dp/dz)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[3, 2]
    im = ax.pcolormesh(r_array, z_array, rel_err_gz * 100, shading='auto', cmap='hot', vmin=0, vmax=50)
    ax.set_xlabel('Range (m)')
    ax.set_title('dp/dz Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    plt.suptitle(
        f"Pekeris Waveguide: FEM vs Analytical Gradient Verification\n"
        f"f={omega/(2*np.pi):.1f} Hz, H={H}m, c1={c1}m/s, c2={c2}m/s | "
        f"L2 error: {errors['L2_relative']:.1%}, H1 error: {errors['H1_relative']:.1%}",
        fontsize=12
    )
    plt.tight_layout()

    filename = "pekeris_gradient_verification.png"
    plt.savefig(filename, dpi=150)
    print(f"\nSaved comparison plot to {filename}")

    # 1D comparison plots
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))

    # Choose a depth for 1D comparison
    z_compare = 75.0  # mid-depth
    iz = np.argmin(np.abs(z_array - z_compare))

    ax = axes2[0, 0]
    ax.semilogy(r_array, np.abs(P_fem[iz, :]), 'b-', label='FEM', linewidth=2)
    ax.semilogy(r_array, np.abs(P_analytical_scaled[iz, :]), 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('|p|')
    ax.set_title(f'Pressure magnitude at z = {z_array[iz]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    ax.plot(r_array, np.real(grad_r_fem[iz, :]), 'b-', label='FEM', linewidth=2)
    ax.plot(r_array, np.real(grad_r_analytical_scaled[iz, :]), 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Re(dp/dr)')
    ax.set_title(f'Radial gradient at z = {z_array[iz]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 2]
    ax.plot(r_array, np.real(grad_z_fem[iz, :]), 'b-', label='FEM', linewidth=2)
    ax.plot(r_array, np.real(grad_z_analytical_scaled[iz, :]), 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Re(dp/dz)')
    ax.set_title(f'Vertical gradient at z = {z_array[iz]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vertical profiles at fixed range
    r_compare = 200.0
    ir = np.argmin(np.abs(r_array - r_compare))

    ax = axes2[1, 0]
    ax.plot(np.abs(P_fem[:, ir]), z_array, 'b-', label='FEM', linewidth=2)
    ax.plot(np.abs(P_analytical_scaled[:, ir]), z_array, 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('|p|')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Pressure profile at r = {r_array[ir]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes2[1, 1]
    ax.plot(np.real(grad_r_fem[:, ir]), z_array, 'b-', label='FEM', linewidth=2)
    ax.plot(np.real(grad_r_analytical_scaled[:, ir]), z_array, 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('Re(dp/dr)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Radial gradient profile at r = {r_array[ir]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes2[1, 2]
    ax.plot(np.real(grad_z_fem[:, ir]), z_array, 'b-', label='FEM', linewidth=2)
    ax.plot(np.real(grad_z_analytical_scaled[:, ir]), z_array, 'r--', label='Analytical', linewidth=2)
    ax.set_xlabel('Re(dp/dz)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Vertical gradient profile at r = {r_array[ir]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.suptitle(f"Pekeris Waveguide: 1D Gradient Comparisons", fontsize=12)
    plt.tight_layout()

    filename2 = "pekeris_gradient_1d_comparison.png"
    plt.savefig(filename2, dpi=150)
    print(f"Saved 1D comparison plot to {filename2}")

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"  L2 relative error (pressure):  {errors['L2_relative']:.2%}")
    print(f"  H1 semi-norm error (gradient): {errors['H1_semi_relative']:.2%}")
    print(f"  H1 relative error (full):      {errors['H1_relative']:.2%}")
    print("=" * 60)

    return {
        'errors': errors,
        'scale': scale,
        'wg': wg,
        'params': params,
    }


if __name__ == "__main__":
    results = main()
