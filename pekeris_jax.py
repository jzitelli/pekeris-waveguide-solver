"""
Pekeris waveguide solver using JAX for automatic differentiation.

This module computes the acoustic pressure field in a two-layer Pekeris waveguide
using only discrete (trapped) modes. Gradients are computed automatically via JAX.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import bessel_jn
from scipy.optimize import brentq
from scipy import special as sp
import numpy as np

# Enable 64-bit precision (important for accuracy)
jax.config.update("jax_enable_x64", True)


def _bessel_j0(x):
    """Bessel function of the first kind, order 0.

    Uses higher n_iter for accuracy at large arguments.
    """
    # n_iter=100 needed for arguments up to ~100, n_iter=200 for larger
    return bessel_jn(x, v=0, n_iter=150)[0]


def _bessel_j1(x):
    """Bessel function of the first kind, order 1."""
    return bessel_jn(x, v=1, n_iter=150)[1]


# Define Y0 and Y1 with custom autodiff rules
# We need to handle the mutual recursion in the derivatives carefully

def _scipy_y0(x):
    """Call scipy's Y0 via pure_callback."""
    x = jnp.asarray(x)
    return jax.pure_callback(
        lambda x: np.asarray(sp.yv(0, np.asarray(x))),
        jax.ShapeDtypeStruct(x.shape, jnp.float64),
        x,
        vmap_method='broadcast_all'
    )


def _scipy_y1(x):
    """Call scipy's Y1 via pure_callback."""
    x = jnp.asarray(x)
    return jax.pure_callback(
        lambda x: np.asarray(sp.yv(1, np.asarray(x))),
        jax.ShapeDtypeStruct(x.shape, jnp.float64),
        x,
        vmap_method='broadcast_all'
    )


@jax.custom_vjp
def _bessel_y0(x):
    """Bessel function of the second kind, order 0."""
    return _scipy_y0(x)


def _bessel_y0_fwd(x):
    y0 = _scipy_y0(x)
    return y0, x


def _bessel_y0_bwd(x, g):
    # d/dx Y0(x) = -Y1(x)
    y1 = _scipy_y1(x)
    return (g * (-y1),)


_bessel_y0.defvjp(_bessel_y0_fwd, _bessel_y0_bwd)


@jax.custom_vjp
def _bessel_y1(x):
    """Bessel function of the second kind, order 1."""
    return _scipy_y1(x)


def _bessel_y1_fwd(x):
    y1 = _scipy_y1(x)
    return y1, x


def _bessel_y1_bwd(x, g):
    # d/dx Y1(x) = Y0(x) - Y1(x)/x
    y0 = _scipy_y0(x)
    y1 = _scipy_y1(x)
    return (g * (y0 - y1 / x),)


_bessel_y1.defvjp(_bessel_y1_fwd, _bessel_y1_bwd)


def compute_eigenvalues(omega: float, c1: float, c2: float,
                        rho1: float, rho2: float, H: float):
    """
    Compute discrete mode eigenvalues using scipy (not JIT-compiled).

    Returns eigenvalues (kr²), horizontal wavenumbers, vertical wavenumbers,
    and normalization constants.
    """
    k1 = omega / c1
    k2 = omega / c2
    k1_sqrd = k1**2
    k2_sqrd = k2**2
    M = rho2 / rho1

    # Maximum number of modes
    n_modes = int(k1 * H * np.sqrt(1 - (c1/c2)**2) / np.pi + 0.5)

    if n_modes == 0:
        return {
            'n_modes': 0,
            'eigenvalues': np.array([]),
            'k_r': np.array([]),
            'k1_z': np.array([]),
            'k2_z': np.array([]),
            'A_sqrd': np.array([]),
        }

    def eigenvalue_equation(kr_sqrd, n):
        k1z = np.sqrt(k1_sqrd - kr_sqrd)
        k2z = np.sqrt(kr_sqrd - k2_sqrd)
        return k1z * H - np.arctan(k2z / (M * k1z)) - (n - 0.5) * np.pi

    # Search bounds
    kr_sqrd_lo = k2_sqrd + 1e-10
    kr_sqrd_hi = k1_sqrd - 1e-10

    eigenvalues = []
    for n in range(1, n_modes + 1):
        kr_sqrd = brentq(eigenvalue_equation, kr_sqrd_lo, kr_sqrd_hi, args=(n,))
        eigenvalues.append(kr_sqrd)

    eigenvalues = np.array(eigenvalues)
    k_r = np.sqrt(eigenvalues)
    k1_z = np.sqrt(k1_sqrd - eigenvalues)
    k2_z = np.sqrt(eigenvalues - k2_sqrd)

    # Normalization constants
    term1 = (1/rho1) * (H - np.sin(2*k1_z*H) / (2*k1_z))
    term2 = (1/rho2) * np.sin(k1_z*H)**2 / k2_z
    A_sqrd = 2.0 / (term1 + term2)

    return {
        'n_modes': n_modes,
        'eigenvalues': eigenvalues,
        'k_r': k_r,
        'k1_z': k1_z,
        'k2_z': k2_z,
        'A_sqrd': A_sqrd,
    }


def _discrete_pressure_water(r: float, z: float, z_s: float, H: float, rho1: float,
                              k_r: jnp.ndarray, k1_z: jnp.ndarray, k2_z: jnp.ndarray,
                              A_sqrd: jnp.ndarray) -> complex:
    """
    Compute discrete modes pressure for a point in the water layer (z <= H).

    This function is designed to be differentiated by JAX.
    """
    # Hankel function of the second kind: H0^(2)(x) = J0(x) - i*Y0(x)
    def hankel2_0(x):
        return _bessel_j0(x) - 1j * _bessel_y0(x)

    # Sum over all modes
    modal_amps = A_sqrd * jnp.sin(k1_z * z_s) * jnp.sin(k1_z * z)
    hankel_vals = hankel2_0(k_r * r)
    p = jnp.sum(modal_amps * hankel_vals)

    # Scale and phase adjustment, then conjugate for exp(+iωt)
    p_scaled = (-p.imag + 1j * p.real) * jnp.pi / rho1
    return jnp.conj(p_scaled)


def _discrete_pressure_sediment(r: float, z: float, z_s: float, H: float, rho1: float,
                                 k_r: jnp.ndarray, k1_z: jnp.ndarray, k2_z: jnp.ndarray,
                                 A_sqrd: jnp.ndarray) -> complex:
    """
    Compute discrete modes pressure for a point in the sediment layer (z > H).
    """
    def hankel2_0(x):
        return _bessel_j0(x) - 1j * _bessel_y0(x)

    modal_amps = (A_sqrd * jnp.sin(k1_z * z_s) *
                  jnp.sin(k1_z * H) * jnp.exp(k2_z * (H - z)))
    hankel_vals = hankel2_0(k_r * r)
    p = jnp.sum(modal_amps * hankel_vals)

    p_scaled = (-p.imag + 1j * p.real) * jnp.pi / rho1
    return jnp.conj(p_scaled)


class PekerisWaveguideJAX:
    """
    Pekeris waveguide solver using JAX for automatic differentiation.

    This implementation only includes discrete modes (no continuous spectrum).
    Gradients are computed automatically using JAX's autodiff.

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s)
    c1 : float
        Sound speed in water layer (m/s)
    c2 : float
        Sound speed in sediment layer (m/s), must be > c1
    rho1 : float
        Density of water layer (kg/m³)
    rho2 : float
        Density of sediment layer (kg/m³)
    H : float
        Depth of water layer (m)
    z_s : float
        Source depth (m), measured from surface (positive downward)
    """

    def __init__(self, omega: float, c1: float, c2: float,
                 rho1: float, rho2: float, H: float, z_s: float):

        if c2 <= c1:
            raise ValueError("Pekeris waveguide requires c2 > c1 for trapped modes")
        if z_s > H:
            raise ValueError("Source depth must be within water layer (z_s <= H)")

        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.rho1 = rho1
        self.rho2 = rho2
        self.H = H
        self.z_s = z_s

        # Compute eigenvalues (using numpy/scipy, not JIT)
        mode_data = compute_eigenvalues(omega, c1, c2, rho1, rho2, H)
        self.n_modes = mode_data['n_modes']
        self.eigenvalues = mode_data['eigenvalues']

        # Convert to JAX arrays for use in JIT-compiled functions
        self.k_r = jnp.array(mode_data['k_r'])
        self.k1_z = jnp.array(mode_data['k1_z'])
        self.k2_z = jnp.array(mode_data['k2_z'])
        self.A_sqrd = jnp.array(mode_data['A_sqrd'])

        # Create JIT-compiled pressure functions
        self._setup_jit_functions()

    def _setup_jit_functions(self):
        """Set up JIT-compiled functions with captured parameters."""

        # Capture parameters for closure
        z_s, H, rho1 = self.z_s, self.H, self.rho1
        k_r, k1_z, k2_z, A_sqrd = self.k_r, self.k1_z, self.k2_z, self.A_sqrd

        @jax.jit
        def pressure_water(r, z):
            return _discrete_pressure_water(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd)

        @jax.jit
        def pressure_sediment(r, z):
            return _discrete_pressure_sediment(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd)

        self._pressure_water = pressure_water
        self._pressure_sediment = pressure_sediment

        # Gradient functions using autodiff
        # For complex output, we compute gradients of real and imaginary parts separately
        def pressure_real_water(r, z):
            return _discrete_pressure_water(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd).real

        def pressure_imag_water(r, z):
            return _discrete_pressure_water(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd).imag

        def pressure_real_sediment(r, z):
            return _discrete_pressure_sediment(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd).real

        def pressure_imag_sediment(r, z):
            return _discrete_pressure_sediment(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd).imag

        # Create gradient functions
        self._grad_real_water = jax.jit(jax.grad(pressure_real_water, argnums=(0, 1)))
        self._grad_imag_water = jax.jit(jax.grad(pressure_imag_water, argnums=(0, 1)))
        self._grad_real_sediment = jax.jit(jax.grad(pressure_real_sediment, argnums=(0, 1)))
        self._grad_imag_sediment = jax.jit(jax.grad(pressure_imag_sediment, argnums=(0, 1)))

    def pressure(self, r: float, z: float) -> complex:
        """
        Compute acoustic pressure at position (r, z).

        Parameters
        ----------
        r : float
            Radial distance from source (m)
        z : float
            Depth (m), positive downward from surface

        Returns
        -------
        complex
            Complex acoustic pressure
        """
        if self.n_modes == 0:
            return 0j

        r, z = float(r), float(z)

        if z <= self.H:
            return complex(self._pressure_water(r, z))
        else:
            return complex(self._pressure_sediment(r, z))

    def pressure_gradient(self, r: float, z: float) -> tuple[complex, complex]:
        """
        Compute pressure gradient at position (r, z) using autodiff.

        Parameters
        ----------
        r : float
            Radial distance from source (m)
        z : float
            Depth (m), positive downward from surface

        Returns
        -------
        tuple[complex, complex]
            (dp/dr, dp/dz) - complex pressure gradients
        """
        if self.n_modes == 0:
            return 0j, 0j

        r, z = float(r), float(z)

        if z <= self.H:
            grad_real = self._grad_real_water(r, z)
            grad_imag = self._grad_imag_water(r, z)
        else:
            grad_real = self._grad_real_sediment(r, z)
            grad_imag = self._grad_imag_sediment(r, z)

        grad_r = complex(grad_real[0] + 1j * grad_imag[0])
        grad_z = complex(grad_real[1] + 1j * grad_imag[1])

        # Match sign convention from original code
        return grad_r, -grad_z

    def pressure_field(self, r_array: np.ndarray, z_array: np.ndarray,
                       show_progress: bool = True) -> np.ndarray:
        """
        Compute pressure field on a grid using vectorized JAX operations.

        Parameters
        ----------
        r_array : np.ndarray
            1D array of radial distances (m)
        z_array : np.ndarray
            1D array of depths (m)
        show_progress : bool
            Print progress indicator

        Returns
        -------
        np.ndarray
            2D complex array of pressure values, shape (len(z_array), len(r_array))
        """
        nz, nr = len(z_array), len(r_array)

        if self.n_modes == 0:
            return np.zeros((nz, nr), dtype=complex)

        if show_progress:
            print(f"  Computing {nr}x{nz} grid with JAX...")

        # Create meshgrid
        R, Z = np.meshgrid(r_array, z_array)

        # Vectorized computation using vmap
        z_s, H, rho1 = self.z_s, self.H, self.rho1
        k_r, k1_z, k2_z, A_sqrd = self.k_r, self.k1_z, self.k2_z, self.A_sqrd

        @jax.jit
        def pressure_point(r, z):
            # Use lax.cond for differentiable branching
            return jax.lax.cond(
                z <= H,
                lambda: _discrete_pressure_water(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd),
                lambda: _discrete_pressure_sediment(r, z, z_s, H, rho1, k_r, k1_z, k2_z, A_sqrd)
            )

        # Vectorize over the grid
        pressure_vec = jax.vmap(jax.vmap(pressure_point, in_axes=(0, None)), in_axes=(None, 0))

        # Compute (note: vmap structure gives us (nz, nr) directly)
        P = pressure_vec(jnp.array(r_array), jnp.array(z_array))

        if show_progress:
            print(f"  Completed {nr*nz} points")

        return np.array(P)

    def __repr__(self):
        return (f"PekerisWaveguideJAX(omega={self.omega}, c1={self.c1}, c2={self.c2}, "
                f"rho1={self.rho1}, rho2={self.rho2}, H={self.H}, z_s={self.z_s}, "
                f"n_modes={self.n_modes})")


def compare_with_original():
    """Compare JAX implementation with original numpy implementation."""
    from pekeris import PekerisWaveguide

    # Test parameters
    omega = 200.0
    c1 = 1500.0
    c2 = 1800.0
    rho1 = 1000.0
    rho2 = 1800.0
    H = 150.0
    z_s = 30.0

    # Create both waveguides
    wg_orig = PekerisWaveguide(omega, c1, c2, rho1, rho2, H, z_s, discrete_modes_only=True)
    wg_jax = PekerisWaveguideJAX(omega, c1, c2, rho1, rho2, H, z_s)

    print("Pekeris Waveguide: JAX vs Original (discrete modes only)")
    print("=" * 60)
    print(f"Number of modes: {wg_jax.n_modes}")
    print()

    # Test points
    test_points = [
        (100.0, 50.0),   # water layer
        (500.0, 75.0),   # water layer
        (200.0, 200.0),  # sediment layer
    ]

    print("Pressure comparison:")
    print("-" * 60)
    for r, z in test_points:
        p_orig = wg_orig.pressure(r, z)
        p_jax = wg_jax.pressure(r, z)
        rel_err = abs(p_orig - p_jax) / abs(p_orig) if abs(p_orig) > 0 else 0
        print(f"  (r={r}, z={z}):")
        print(f"    Original: {p_orig}")
        print(f"    JAX:      {p_jax}")
        print(f"    Rel err:  {rel_err:.2e}")
        print()

    print("Gradient comparison (autodiff vs manual):")
    print("-" * 60)
    for r, z in test_points:
        gr_orig, gz_orig = wg_orig.pressure_gradient(r, z)
        gr_jax, gz_jax = wg_jax.pressure_gradient(r, z)

        rel_err_r = abs(gr_orig - gr_jax) / abs(gr_orig) if abs(gr_orig) > 0 else 0
        rel_err_z = abs(gz_orig - gz_jax) / abs(gz_orig) if abs(gz_orig) > 0 else 0

        print(f"  (r={r}, z={z}):")
        print(f"    dp/dr Original: {gr_orig}")
        print(f"    dp/dr JAX:      {gr_jax}")
        print(f"    dp/dr Rel err:  {rel_err_r:.2e}")
        print(f"    dp/dz Original: {gz_orig}")
        print(f"    dp/dz JAX:      {gz_jax}")
        print(f"    dp/dz Rel err:  {rel_err_z:.2e}")
        print()


def example(show_plot=True):
    """Run an example calculation matching the Fortran test case."""

    # Example parameters (from commented main() in pekeris.c)
    omega = 200.0       # rad/s
    c1 = 1500.0         # m/s (water)
    c2 = 1800.0         # m/s (sediment)
    rho1 = 1000.0       # kg/m³ (water)
    rho2 = 1800.0       # kg/m³ (sediment)
    H = 150.0           # m (water depth)
    z_s = 30.0          # m (source depth)

    # Create waveguide
    wg = PekerisWaveguideJAX(omega, c1, c2, rho1, rho2, H, z_s)

    print(f"Pekeris Waveguide Parameters:")
    print(f"  omega = {omega} rad/s (f = {omega/(2*np.pi):.2f} Hz)")
    print(f"  c1 = {c1} m/s, c2 = {c2} m/s")
    print(f"  rho1 = {rho1} kg/m³, rho2 = {rho2} kg/m³")
    print(f"  H = {H} m, z_s = {z_s} m")
    print(f"\nNumber of discrete modes: {wg.n_modes}")
    print(f"\nEigenvalues (kr²):")
    for i, ev in enumerate(wg.eigenvalues):
        print(f"  Mode {i+1}: kr² = {ev:.10f}, kr = {np.sqrt(ev):.10f}")

    # Compute pressure at a test point
    r_test = 100.0  # m
    z_test = 50.0   # m
    p = wg.pressure(r_test, z_test)
    grad_r, grad_z = wg.pressure_gradient(r_test, z_test)

    print(f"\nPressure at (r={r_test}, z={z_test}):")
    print(f"  p = {p}")
    print(f"  |p| = {np.abs(p)}")
    print(f"  dp/dr = {grad_r}")
    print(f"  dp/dz = {grad_z}")

    if show_plot:
        plot_field(wg)

    return wg


def plot_field(wg: PekerisWaveguideJAX, r_max: float = 1000.0, z_max: float = None,
               nr: int = 200, nz: int = 100, r_min: float = 1.0, save_only: bool = False):
    """
    Plot the pressure field magnitude and phase.

    Parameters
    ----------
    wg : PekerisWaveguide
        Waveguide instance
    r_max : float
        Maximum range (m)
    z_max : float
        Maximum depth (m), defaults to 1.5 * H to show some sediment
    nr : int
        Number of range points
    nz : int
        Number of depth points
    r_min : float
        Minimum range (m), avoid r=0 singularity
    save_only : bool
        If True, only save to file without displaying (for headless environments)
    """
    import matplotlib
    if save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if z_max is None:
        z_max = 1.5 * wg.H

    # Create grid
    r_array = np.linspace(r_min, r_max, nr)
    z_array = np.linspace(0, z_max, nz)

    mode_str = "discrete only"
    print(f"\nComputing pressure field on {nr}x{nz} grid ({mode_str})...")
    P = wg.pressure_field(r_array, z_array)

    # Compute transmission loss: TL = -20*log10(|p| * r) relative to 1m
    # For display, use magnitude in dB relative to max
    P_abs = np.abs(P)
    P_db = 20 * np.log10(P_abs + 1e-20)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Magnitude (dB)
    ax1 = axes[0]
    im1 = ax1.pcolormesh(r_array, z_array, P_db, shading='auto', cmap='viridis', vmin=-100, vmax=0)
    ax1.axhline(wg.H, color='white', linestyle='--', linewidth=1, label='Seafloor')
    ax1.axhline(wg.z_s, color='red', linestyle=':', linewidth=1, label=f'Source (z={wg.z_s}m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title(f'Pressure Magnitude (dB re max) - {wg.n_modes} modes ({mode_str})')
    ax1.invert_yaxis()
    ax1.legend(loc='lower right')
    cbar1 = fig.colorbar(im1, ax=ax1, label='dB')

    # Plot 2: Real part (shows interference pattern)
    ax2 = axes[1]
    P_real = np.real(P)
    vlim = 0.01 #np.max(np.abs(P_real)) * 0.00002
    im2 = ax2.pcolormesh(r_array, z_array, P_real, shading='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax2.axhline(wg.H, color='black', linestyle='--', linewidth=1)
    ax2.axhline(wg.z_s, color='red', linestyle=':', linewidth=1)
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Pressure (Real part) - showing interference pattern')
    ax2.invert_yaxis()
    cbar2 = fig.colorbar(im2, ax=ax2, label='Re(p)')

    plt.suptitle(f'Pekeris Waveguide: f={wg.omega/(2*np.pi):.1f} Hz, H={wg.H}m, '
                 f'c₁={wg.c1} m/s, c₂={wg.c2} m/s', fontsize=12)
    plt.tight_layout()
    filename = 'pekeris_field.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved figure to {filename}")
    if not save_only:
        plt.show()


def animate_field(wg: PekerisWaveguideJAX, r_max: float = 1000.0, z_max: float = None,
                  nr: int = 200, nz: int = 100, r_min: float = 1.0,
                  filename: str = 'pekeris_animation.gif'):
    """
    Generate an animated GIF of the time-harmonic pressure field.

    Parameters
    ----------
    wg : PekerisWaveguide
        Waveguide instance
    r_max : float
        Maximum range (m)
    z_max : float
        Maximum depth (m), defaults to 1.5 * H
    nr : int
        Number of range points
    nz : int
        Number of depth points
    r_min : float
        Minimum range (m)
    filename : str
        Output GIF filename
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    n_frames = 30

    if z_max is None:
        z_max = 1.5 * wg.H

    # Create grid
    r_array = np.linspace(r_min, r_max, nr)
    z_array = np.linspace(0, z_max, nz)

    mode_str = "discrete only"
    print(f"\nComputing pressure field on {nr}x{nz} grid ({mode_str})...")
    P = wg.pressure_field(r_array, z_array)

    # Time phases for animation (one full period)
    phases = np.linspace(0, 2*np.pi, n_frames, endpoint=False)

    # Determine color scale from the full range of real values
    P_real_max = np.max(np.abs(P))
    vlim = 0.01 # P_real_max * 0.00015

    frames = []
    print(f"Generating {n_frames} animation frames...")

    for i, phase in enumerate(phases):
        # Instantaneous pressure: Re(P * exp(i*phase))
        P_instant = np.real(P * np.exp(-1j * phase))

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.pcolormesh(r_array, z_array, P_instant, shading='auto',
                          cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        ax.axhline(wg.H, color='black', linestyle='--', linewidth=1, label='Seafloor')
        ax.axhline(wg.z_s, color='red', linestyle=':', linewidth=1, label='Source')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Pekeris Waveguide: f={wg.omega/(2*np.pi):.1f} Hz')
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        fig.colorbar(im, ax=ax, label='Re(p)')
        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # Save as animated GIF
    print(f"Saving animation to {filename}...")
    frames[0].save(filename, save_all=True, append_images=frames[1:],
                   duration=20, loop=0)
    print(f"Animation saved to {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pekeris waveguide solver')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--animate-field', action='store_true',
                       help='Generate animated GIF of time-harmonic field')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                       help='Number of parallel processes (default: all CPUs)')
    args = parser.parse_args()

    compare_with_original()

    wg = example(show_plot=not args.no_plot)

    if args.animate_field:
        animate_field(wg, nr=400, nz=200)
