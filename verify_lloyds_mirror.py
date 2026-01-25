"""
Verification of FEM solver against Lloyd's mirror analytical solution.

Lloyd's mirror corresponds to a homogeneous half-space with a pressure-release
surface at z=0. The analytical solution uses the method of images:
    p(r,z) = G(r, z-z_s) - G(r, z+z_s)

where G is the 3D free-space Green's function for the Helmholtz equation:
    G(R) = exp(ikR) / (4πR)

This script compares the FEM solution (with sediment properties = water properties)
against this analytical solution.
"""

import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dolfinx import fem, default_scalar_type
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl


def lloyds_mirror_pressure(r, z, z_s, k, source_strength=1.0):
    """
    Compute Lloyd's mirror pressure field.

    Parameters
    ----------
    r : array_like
        Radial coordinates (can be 1D or 2D array)
    z : array_like
        Depth coordinates (can be 1D or 2D array)
    z_s : float
        Source depth
    k : float
        Wavenumber (omega/c)
    source_strength : float
        Source amplitude

    Returns
    -------
    p : array_like
        Complex pressure field
    """
    # Distance to real source at (0, z_s)
    R1 = np.sqrt(r**2 + (z - z_s)**2)
    # Distance to image source at (0, -z_s)
    R2 = np.sqrt(r**2 + (z + z_s)**2)

    # Avoid division by zero
    R1 = np.maximum(R1, 1e-10)
    R2 = np.maximum(R2, 1e-10)

    # Green's function: G(R) = exp(ikR) / (4πR)
    # Lloyd's mirror: p = G(R1) - G(R2)
    p = source_strength / (4 * np.pi) * (np.exp(1j * k * R1) / R1 - np.exp(1j * k * R2) / R2)

    return p


def lloyds_mirror_gradient(r, z, z_s, k, source_strength=1.0):
    """
    Compute gradient of Lloyd's mirror pressure field.

    Returns
    -------
    grad_r, grad_z : array_like
        Radial and vertical components of pressure gradient
    """
    # Distance to real source
    R1 = np.sqrt(r**2 + (z - z_s)**2)
    # Distance to image source
    R2 = np.sqrt(r**2 + (z + z_s)**2)

    R1 = np.maximum(R1, 1e-10)
    R2 = np.maximum(R2, 1e-10)

    # Gradient of exp(ikR)/R is: (ik - 1/R) * exp(ikR) / R * (position/R)
    # d/dr [exp(ikR)/R] = (ik - 1/R) * exp(ikR) / R * (r/R)
    # d/dz [exp(ikR)/R] = (ik - 1/R) * exp(ikR) / R * ((z-z_s)/R)

    coeff = source_strength / (4 * np.pi)

    # Real source contribution
    factor1 = (1j * k - 1/R1) * np.exp(1j * k * R1) / R1
    grad_r_1 = factor1 * (r / R1)
    grad_z_1 = factor1 * ((z - z_s) / R1)

    # Image source contribution (negative)
    factor2 = (1j * k - 1/R2) * np.exp(1j * k * R2) / R2
    grad_r_2 = factor2 * (r / R2)
    grad_z_2 = factor2 * ((z + z_s) / R2)

    grad_r = coeff * (grad_r_1 - grad_r_2)
    grad_z = coeff * (grad_z_1 - grad_z_2)

    return grad_r, grad_z


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
    # Get degree from the element (degree is an attribute, not a method)
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


def compute_errors_on_mesh(mesh_data, uh, z_s, k, scale):
    """
    Compute L2 and H1 errors by integrating over the mesh.

    This is more accurate than point-wise comparison on a grid.
    Only integrates over the physical water domain (not PML or sediment).
    """
    from pekeris_gmsh import WATER_DOMAIN

    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags

    # Get function space
    V = uh.function_space

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    r, z = x[0], x[1]

    # Analytical solution (scaled)
    R1 = ufl.sqrt(r**2 + (z - z_s)**2 + 1e-20)  # small offset to avoid singularity
    R2 = ufl.sqrt(r**2 + (z + z_s)**2 + 1e-20)

    p_analytical = scale / (4 * np.pi) * (ufl.exp(1j * k * R1) / R1 - ufl.exp(1j * k * R2) / R2)

    # Error
    error = uh - p_analytical

    # Define measure restricted to water domain only (no PML, no sediment)
    dx_water = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)(WATER_DOMAIN)

    # L2 norm: ||e||_L2 = sqrt(integral |e|^2 r dr dz)
    # Note: factor of r for axisymmetric integration
    L2_error_sq = fem.form(ufl.inner(error, error) * r * dx_water)
    L2_norm_sq = fem.form(ufl.inner(p_analytical, p_analytical) * r * dx_water)

    L2_error = np.sqrt(np.abs(fem.assemble_scalar(L2_error_sq)))
    L2_norm = np.sqrt(np.abs(fem.assemble_scalar(L2_norm_sq)))

    # H1 semi-norm: |e|_H1 = sqrt(integral |grad(e)|^2 r dr dz)
    grad_error = ufl.grad(error)
    H1_semi_error_sq = fem.form(ufl.inner(grad_error, grad_error) * r * dx_water)

    grad_analytical = ufl.grad(p_analytical)
    H1_semi_norm_sq = fem.form(ufl.inner(grad_analytical, grad_analytical) * r * dx_water)

    H1_semi_error = np.sqrt(np.abs(fem.assemble_scalar(H1_semi_error_sq)))
    H1_semi_norm = np.sqrt(np.abs(fem.assemble_scalar(H1_semi_norm_sq)))

    # Full H1 norm: ||e||_H1 = sqrt(||e||_L2^2 + |e|_H1^2)
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


def find_scaling_factor(P_fem, P_analytical, mask):
    """
    Find complex scaling factor: P_fem ≈ scale * P_analytical
    """
    P_fem_valid = P_fem[mask]
    P_anal_valid = P_analytical[mask]

    # Least squares: scale = (P_anal^H * P_fem) / (P_anal^H * P_anal)
    scale = np.vdot(P_anal_valid, P_fem_valid) / np.vdot(P_anal_valid, P_anal_valid)
    return scale


def main():
    from pekeris_fem import solve_pekeris_fem

    # Problem parameters
    omega = 200.0  # rad/s
    c = 1500.0     # m/s (same for water and sediment = Lloyd's mirror)
    rho = 1000.0   # kg/m³
    H = 150.0      # water depth (but sediment has same properties)
    z_s = 30.0     # source depth
    r_max = 500.0  # domain extent
    z_max = 225.0

    k = omega / c
    wavelength = 2 * np.pi / k

    print("=" * 60)
    print("Lloyd's Mirror Verification")
    print("=" * 60)
    print(f"  omega = {omega} rad/s (f = {omega/(2*np.pi):.2f} Hz)")
    print(f"  c = {c} m/s")
    print(f"  k = {k:.6f} rad/m, wavelength = {wavelength:.2f} m")
    print(f"  Source depth z_s = {z_s} m")
    print()

    # Solve FEM with homogeneous properties (Lloyd's mirror case)
    print("Running FEM solver with homogeneous properties...")
    mesh_data, uh, params = solve_pekeris_fem(
        omega=omega,
        c1=c, c2=c,        # Same sound speed
        rho1=rho, rho2=rho,  # Same density
        H=H,
        z_s=z_s,
        r_max=r_max,
        z_max=z_max,
        pml_r=100.0,
        pml_z=100.0,
        lc_fine=2.0,
        lc_coarse=15.0,
        degree=2,
        pml_alpha=2.0,
    )

    # Create comparison grid (avoid source region and boundaries)
    # Only compare in water layer (z < H) where Lloyd's mirror is valid
    r_min = 10.0  # Stay away from source
    nr, nz = 80, 60
    r_array = np.linspace(r_min, r_max * 0.9, nr)
    z_array = np.linspace(5.0, H - 5.0, nz)  # Stay in water layer only

    R_grid, Z_grid = np.meshgrid(r_array, z_array)

    print("\nExtracting FEM solution on comparison grid...")
    P_fem, grad_r_fem, grad_z_fem = extract_fem_solution_and_gradient(
        mesh_data, uh, r_array, z_array
    )

    print("Computing analytical Lloyd's mirror solution...")
    P_analytical = lloyds_mirror_pressure(R_grid, Z_grid, z_s, k)
    grad_r_analytical, grad_z_analytical = lloyds_mirror_gradient(R_grid, Z_grid, z_s, k)

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

    # Compute point-wise errors on grid
    error_p = np.abs(P_fem - P_analytical_scaled)
    rel_error_p = error_p / (np.abs(P_analytical_scaled) + 1e-20)

    error_grad_r = np.abs(grad_r_fem - grad_r_analytical_scaled)
    error_grad_z = np.abs(grad_z_fem - grad_z_analytical_scaled)
    error_grad = np.sqrt(error_grad_r**2 + error_grad_z**2)
    grad_mag_analytical = np.sqrt(np.abs(grad_r_analytical_scaled)**2 + np.abs(grad_z_analytical_scaled)**2)
    rel_error_grad = error_grad / (grad_mag_analytical + 1e-20)

    print("\n" + "=" * 60)
    print("Point-wise Errors (on comparison grid)")
    print("=" * 60)
    print(f"  Pressure:")
    print(f"    Mean relative error: {np.nanmean(rel_error_p[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_error_p[valid_mask]):.2%}")
    print(f"    RMS relative error:  {np.sqrt(np.nanmean(rel_error_p[valid_mask]**2)):.2%}")
    print(f"  Gradient:")
    print(f"    Mean relative error: {np.nanmean(rel_error_grad[valid_mask]):.2%}")
    print(f"    Max relative error:  {np.nanmax(rel_error_grad[valid_mask]):.2%}")

    # Compute integrated errors over the mesh
    print("\n" + "=" * 60)
    print("Integrated Errors (over mesh, excluding PML)")
    print("=" * 60)

    errors = compute_errors_on_mesh(mesh_data, uh, z_s, k, scale)
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

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: Magnitude comparison
    P_fem_db = 20 * np.log10(np.abs(P_fem) / np.nanmax(np.abs(P_fem)) + 1e-20)
    P_anal_db = 20 * np.log10(np.abs(P_analytical_scaled) / np.nanmax(np.abs(P_analytical_scaled)) + 1e-20)

    ax = axes[0, 0]
    im = ax.pcolormesh(r_array, z_array, P_fem_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1, label=f'Source z={z_s}m')
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM |p| (dB re max)')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.pcolormesh(r_array, z_array, P_anal_db, shading='auto', cmap='viridis', vmin=-60, vmax=0)
    ax.axhline(z_s, color='red', linestyle=':', linewidth=1)
    ax.set_title("Lloyd's Mirror |p| (dB re max)")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    diff_db = np.abs(P_fem_db - P_anal_db)
    im = ax.pcolormesh(r_array, z_array, diff_db, shading='auto', cmap='hot', vmin=0, vmax=5)
    ax.set_title('|Difference| (dB)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    # Row 2: Real part comparison
    vlim = np.nanmax(np.abs(np.real(P_fem))) * 0.1

    ax = axes[1, 0]
    im = ax.pcolormesh(r_array, z_array, np.real(P_fem), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM Re(p)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.pcolormesh(r_array, z_array, np.real(P_analytical_scaled), shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_title("Lloyd's Mirror Re(p)")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.pcolormesh(r_array, z_array, rel_error_p * 100, shading='auto', cmap='hot', vmin=0, vmax=20)
    ax.set_title('Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    # Row 3: Gradient comparison
    grad_mag_fem = np.sqrt(np.abs(grad_r_fem)**2 + np.abs(grad_z_fem)**2)
    grad_mag_anal = np.sqrt(np.abs(grad_r_analytical_scaled)**2 + np.abs(grad_z_analytical_scaled)**2)

    vlim_grad = np.nanmax(grad_mag_fem) * 0.3

    ax = axes[2, 0]
    im = ax.pcolormesh(r_array, z_array, grad_mag_fem, shading='auto', cmap='viridis', vmin=0, vmax=vlim_grad)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('FEM |∇p|')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[2, 1]
    im = ax.pcolormesh(r_array, z_array, grad_mag_anal, shading='auto', cmap='viridis', vmin=0, vmax=vlim_grad)
    ax.set_xlabel('Range (m)')
    ax.set_title("Lloyd's Mirror |∇p|")
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

    ax = axes[2, 2]
    im = ax.pcolormesh(r_array, z_array, rel_error_grad * 100, shading='auto', cmap='hot', vmin=0, vmax=20)
    ax.set_xlabel('Range (m)')
    ax.set_title('Gradient Relative Error (%)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='%')

    plt.suptitle(
        f"Lloyd's Mirror Verification: FEM vs Analytical\n"
        f"f={omega/(2*np.pi):.1f} Hz, c={c} m/s, z_s={z_s} m | "
        f"L2 error: {errors['L2_relative']:.1%}, H1 error: {errors['H1_relative']:.1%}",
        fontsize=12
    )
    plt.tight_layout()

    filename = "lloyds_mirror_verification.png"
    plt.savefig(filename, dpi=150)
    print(f"\nSaved comparison plot to {filename}")

    # Also create a 1D comparison along a horizontal line
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

    # Choose a depth for 1D comparison
    z_compare = 75.0  # mid-depth
    iz = np.argmin(np.abs(z_array - z_compare))

    ax = axes2[0, 0]
    ax.semilogy(r_array, np.abs(P_fem[iz, :]), 'b-', label='FEM', linewidth=2)
    ax.semilogy(r_array, np.abs(P_analytical_scaled[iz, :]), 'r--', label="Lloyd's Mirror", linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('|p|')
    ax.set_title(f'Pressure magnitude at z = {z_array[iz]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[0, 1]
    ax.plot(r_array, np.real(P_fem[iz, :]), 'b-', label='FEM Re(p)', linewidth=2)
    ax.plot(r_array, np.real(P_analytical_scaled[iz, :]), 'r--', label="Lloyd's Re(p)", linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Re(p)')
    ax.set_title(f'Real part at z = {z_array[iz]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1, 0]
    ax.semilogy(r_array, rel_error_p[iz, :] * 100, 'k-', linewidth=2)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'Pressure relative error at z = {z_array[iz]:.1f} m')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.1, 100])

    # Vertical profile at a fixed range
    r_compare = 200.0
    ir = np.argmin(np.abs(r_array - r_compare))

    ax = axes2[1, 1]
    ax.plot(np.abs(P_fem[:, ir]), z_array, 'b-', label='FEM', linewidth=2)
    ax.plot(np.abs(P_analytical_scaled[:, ir]), z_array, 'r--', label="Lloyd's Mirror", linewidth=2)
    ax.set_xlabel('|p|')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Vertical profile at r = {r_array[ir]:.1f} m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.suptitle(f"Lloyd's Mirror: 1D Comparisons", fontsize=12)
    plt.tight_layout()

    filename2 = "lloyds_mirror_1d_comparison.png"
    plt.savefig(filename2, dpi=150)
    print(f"Saved 1D comparison plot to {filename2}")

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"  L2 relative error:  {errors['L2_relative']:.2%}")
    print(f"  H1 relative error:  {errors['H1_relative']:.2%}")
    print("=" * 60)

    return errors


if __name__ == "__main__":
    errors = main()
