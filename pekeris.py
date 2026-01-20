"""
Pekeris waveguide exact solution.

This module computes the acoustic pressure field in a two-layer Pekeris waveguide:
- Layer 1 (water): constant sound speed c1, density rho1, depth H
- Layer 2 (sediment): constant sound speed c2 > c1, density rho2, semi-infinite

The solution includes both discrete (trapped) modes and the continuous spectrum.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, yv, kv  # Bessel functions J_n, Y_n, K_n
from scipy.integrate import quad


class PekerisWaveguide:
    """
    Pekeris waveguide solver.

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
    discrete_modes_only : bool
        If True, only compute discrete modes (faster but incomplete).
        If False (default), include continuous spectrum contribution.
    quadrature_rtol : float
        Relative tolerance for continuous spectrum integration (default 1e-7)
    """

    def __init__(self, omega: float, c1: float, c2: float,
                 rho1: float, rho2: float, H: float, z_s: float,
                 discrete_modes_only: bool = False,
                 quadrature_rtol: float = 1e-7):

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
        self.discrete_modes_only = discrete_modes_only
        self.quadrature_rtol = quadrature_rtol

        # Wavenumbers
        self.k1 = omega / c1  # wavenumber in water
        self.k2 = omega / c2  # wavenumber in sediment
        self.k1_sqrd = self.k1**2
        self.k2_sqrd = self.k2**2

        # Density ratio
        self.M = rho2 / rho1

        # Branch point for continuous spectrum (where k_r = k2)
        self.branch_start = np.sqrt(self.k1_sqrd - self.k2_sqrd)

        # Compute eigenvalues and mode properties
        self._compute_modes()

    def _eigenvalue_equation(self, kr_sqrd: float, n: int) -> float:
        """
        Eigenvalue equation for the n-th mode.

        The discrete mode eigenvalues kr² satisfy:
            k1z * H - atan(k2z / (M * k1z)) - (n - 0.5) * pi = 0

        where:
            k1z = sqrt(k1² - kr²)  (vertical wavenumber in water)
            k2z = sqrt(kr² - k2²)  (vertical wavenumber in sediment)

        Parameters
        ----------
        kr_sqrd : float
            Horizontal wavenumber squared (eigenvalue candidate)
        n : int
            Mode number (1, 2, 3, ...)

        Returns
        -------
        float
            Residual of the eigenvalue equation
        """
        k1z = np.sqrt(self.k1_sqrd - kr_sqrd)
        k2z = np.sqrt(kr_sqrd - self.k2_sqrd)

        return k1z * self.H - np.arctan(k2z / (self.M * k1z)) - (n - 0.5) * np.pi

    def _compute_modes(self):
        """Compute all discrete mode eigenvalues and normalization constants."""

        # Maximum number of modes (from Pekeris theory)
        # n_max = floor(k1 * H * sqrt(1 - c1²/c2²) / pi + 0.5)
        self.n_modes = int(self.k1 * self.H * np.sqrt(1 - (self.c1/self.c2)**2) / np.pi + 0.5)

        if self.n_modes == 0:
            self.eigenvalues = np.array([])
            self.k_r = np.array([])
            self.k1_z = np.array([])
            self.k2_z = np.array([])
            self.A_sqrd = np.array([])
            return

        # Search bounds for eigenvalues: k2² < kr² < k1²
        kr_sqrd_lo = self.k2_sqrd + 1e-10
        kr_sqrd_hi = self.k1_sqrd - 1e-10

        eigenvalues = []
        for n in range(1, self.n_modes + 1):
            # Solve eigenvalue equation using Brent's method
            kr_sqrd = brentq(self._eigenvalue_equation, kr_sqrd_lo, kr_sqrd_hi, args=(n,))
            eigenvalues.append(kr_sqrd)

        self.eigenvalues = np.array(eigenvalues)

        # Compute derived quantities
        self.k_r = np.sqrt(self.eigenvalues)      # horizontal wavenumber
        self.k1_z = np.sqrt(self.k1_sqrd - self.eigenvalues)  # vertical wavenumber in water
        self.k2_z = np.sqrt(self.eigenvalues - self.k2_sqrd)  # vertical wavenumber in sediment

        # Mode normalization (amplitude squared)
        # A² = 2 / [(1/ρ1) * (H - sin(2*k1z*H)/(2*k1z)) + (1/ρ2) * sin²(k1z*H) / k2z]
        term1 = (1/self.rho1) * (self.H - np.sin(2*self.k1_z*self.H) / (2*self.k1_z))
        term2 = (1/self.rho2) * np.sin(self.k1_z*self.H)**2 / self.k2_z
        self.A_sqrd = 2.0 / (term1 + term2)

    # -------------------------------------------------------------------------
    # Continuous spectrum integrand functions
    # The integration variable x is the vertical wavenumber in the water layer (k1_z)
    # -------------------------------------------------------------------------

    def _cont_spectrum_denominator(self, x: float) -> float:
        """Common denominator for continuous spectrum integrands."""
        # k_r² = k1² - x² (for definite region) or x² - k1² (for indefinite)
        # k2_z² = k2² - k_r² = k2² - k1² + x²
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        sin_sqrd = np.sin(x * self.H)**2
        ratio = (self.M * x / k2_z)**2
        return ratio + (1 - ratio) * sin_sqrd

    def _integrand_definite_real(self, x: float, r: float, z: float) -> float:
        """Real part integrand for definite region [branch_start, k1] using J0."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return jv(0, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_definite_imag(self, x: float, r: float, z: float) -> float:
        """Imaginary part integrand for definite region [branch_start, k1] using Y0."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return yv(0, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_indefinite(self, x: float, r: float, z: float) -> float:
        """Integrand for indefinite region [k1, cutoff] using K0."""
        # In this region, k_r is imaginary: k_r = i * sqrt(x² - k1²)
        # So we use K0 which is related to H0^(2) for imaginary argument
        k_r_mag = np.sqrt(x**2 - self.k1_sqrd)  # magnitude of imaginary k_r
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        # Factor of -2/pi comes from K0(x) = -pi/2 * i * H0^(2)(ix) relationship
        return -(2/np.pi) * kv(0, k_r_mag * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_definite_real_dr(self, x: float, r: float, z: float) -> float:
        """d/dr of real part integrand for definite region."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        # d/dr of J0(k_r * r) = -k_r * J1(k_r * r)
        return -k_r * jv(1, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_definite_imag_dr(self, x: float, r: float, z: float) -> float:
        """d/dr of imaginary part integrand for definite region."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return -k_r * yv(1, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_indefinite_dr(self, x: float, r: float, z: float) -> float:
        """d/dr of integrand for indefinite region."""
        k_r_mag = np.sqrt(x**2 - self.k1_sqrd)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        # d/dr of K0(k_r_mag * r) = -k_r_mag * K1(k_r_mag * r)
        return (2/np.pi) * k_r_mag * kv(1, k_r_mag * r) * (x / k2_z) * np.sin(x * self.z_s) * np.sin(x * z) / denom

    def _integrand_definite_real_dz(self, x: float, r: float, z: float) -> float:
        """d/dz of real part integrand for definite region."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return x * jv(0, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.cos(x * z) / denom

    def _integrand_definite_imag_dz(self, x: float, r: float, z: float) -> float:
        """d/dz of imaginary part integrand for definite region."""
        k_r = np.sqrt(self.k1_sqrd - x**2)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return x * yv(0, k_r * r) * (x / k2_z) * np.sin(x * self.z_s) * np.cos(x * z) / denom

    def _integrand_indefinite_dz(self, x: float, r: float, z: float) -> float:
        """d/dz of integrand for indefinite region."""
        k_r_mag = np.sqrt(x**2 - self.k1_sqrd)
        k2_z = np.sqrt(self.k2_sqrd - self.k1_sqrd + x**2)
        denom = self._cont_spectrum_denominator(x)
        return -(2/np.pi) * x * kv(0, k_r_mag * r) * (x / k2_z) * np.sin(x * self.z_s) * np.cos(x * z) / denom

    def _continuous_spectrum_pressure(self, r: float, z: float) -> complex:
        """
        Compute the continuous spectrum contribution to pressure.

        The continuous spectrum integral is split into two parts:
        1. Definite region: [branch_start, k1] - oscillatory in water layer
        2. Indefinite region: [k1, cutoff] - evanescent in water layer
        """
        # Definite region integrals
        int_real, _ = quad(self._integrand_definite_real, self.branch_start, self.k1,
                          args=(r, z), limit=100, epsrel=self.quadrature_rtol)
        int_imag, _ = quad(self._integrand_definite_imag, self.branch_start, self.k1,
                          args=(r, z), limit=100, epsrel=self.quadrature_rtol)

        # Indefinite region integral (evanescent)
        # Cutoff chosen so that K0 argument is large enough for negligible contribution
        cutoff = np.sqrt((100.0 / r)**2 + self.k1_sqrd)
        int_indef, _ = quad(self._integrand_indefinite, self.k1, cutoff,
                           args=(r, z), limit=100, epsrel=self.quadrature_rtol)

        # Combine: the indefinite integral contributes to imaginary part
        integral = complex(int_real, int_imag + int_indef)

        # Scale factor and phase adjustment (matching Fortran)
        # The factor 2*M comes from the residue calculation
        return 2 * self.M * complex(-integral.imag, integral.real)

    def _continuous_spectrum_gradient(self, r: float, z: float) -> tuple[complex, complex]:
        """
        Compute the continuous spectrum contribution to pressure gradient.
        """
        # d/dr integrals
        int_real_dr, _ = quad(self._integrand_definite_real_dr, self.branch_start, self.k1,
                             args=(r, z), limit=100, epsrel=self.quadrature_rtol)
        int_imag_dr, _ = quad(self._integrand_definite_imag_dr, self.branch_start, self.k1,
                             args=(r, z), limit=100, epsrel=self.quadrature_rtol)
        cutoff = np.sqrt((100.0 / r)**2 + self.k1_sqrd)
        int_indef_dr, _ = quad(self._integrand_indefinite_dr, self.k1, cutoff,
                              args=(r, z), limit=100, epsrel=self.quadrature_rtol)

        integral_dr = complex(int_real_dr, int_imag_dr + int_indef_dr)
        grad_r = 2 * self.M * complex(-integral_dr.imag, integral_dr.real)

        # d/dz integrals
        int_real_dz, _ = quad(self._integrand_definite_real_dz, self.branch_start, self.k1,
                             args=(r, z), limit=100, epsrel=self.quadrature_rtol)
        int_imag_dz, _ = quad(self._integrand_definite_imag_dz, self.branch_start, self.k1,
                             args=(r, z), limit=100, epsrel=self.quadrature_rtol)
        int_indef_dz, _ = quad(self._integrand_indefinite_dz, self.k1, cutoff,
                              args=(r, z), limit=100, epsrel=self.quadrature_rtol)

        integral_dz = complex(int_real_dz, int_imag_dz + int_indef_dz)
        grad_z = 2 * self.M * complex(-integral_dz.imag, integral_dz.real)

        return grad_r, grad_z

    def _discrete_modes_pressure(self, r: float, z: float) -> complex:
        """Compute discrete modes contribution to pressure."""
        if self.n_modes == 0:
            return 0j

        p = 0j

        def hankel2_0(x):
            return jv(0, x) - 1j * yv(0, x)

        if z <= self.H:
            for n in range(self.n_modes):
                modal_amp = self.A_sqrd[n] * np.sin(self.k1_z[n] * self.z_s) * np.sin(self.k1_z[n] * z)
                p += modal_amp * hankel2_0(self.k_r[n] * r)
        else:
            for n in range(self.n_modes):
                modal_amp = (self.A_sqrd[n] * np.sin(self.k1_z[n] * self.z_s) *
                            np.sin(self.k1_z[n] * self.H) * np.exp(self.k2_z[n] * (self.H - z)))
                p += modal_amp * hankel2_0(self.k_r[n] * r)

        # Scale and phase adjustment
        return complex(-p.imag, p.real) * np.pi / self.rho1

    def _discrete_modes_gradient(self, r: float, z: float) -> tuple[complex, complex]:
        """Compute discrete modes contribution to pressure gradient."""
        if self.n_modes == 0:
            return 0j, 0j

        grad_r = 0j
        grad_z = 0j

        def hankel2_0(x):
            return jv(0, x) - 1j * yv(0, x)

        def hankel2_1(x):
            return jv(1, x) - 1j * yv(1, x)

        if z <= self.H:
            for n in range(self.n_modes):
                sin_zs = np.sin(self.k1_z[n] * self.z_s)
                sin_z = np.sin(self.k1_z[n] * z)
                cos_z = np.cos(self.k1_z[n] * z)
                H0 = hankel2_0(self.k_r[n] * r)
                H1 = hankel2_1(self.k_r[n] * r)

                grad_r += -self.k_r[n] * self.A_sqrd[n] * sin_zs * sin_z * H1
                grad_z += self.k1_z[n] * self.A_sqrd[n] * sin_zs * cos_z * H0
        else:
            for n in range(self.n_modes):
                sin_zs = np.sin(self.k1_z[n] * self.z_s)
                sin_H = np.sin(self.k1_z[n] * self.H)
                exp_decay = np.exp(self.k2_z[n] * (self.H - z))
                H0 = hankel2_0(self.k_r[n] * r)
                H1 = hankel2_1(self.k_r[n] * r)

                modal_base = self.A_sqrd[n] * sin_zs * sin_H * exp_decay
                grad_r += -self.k_r[n] * modal_base * H1
                grad_z += -self.k2_z[n] * modal_base * H0

        scale = 1j * np.pi / self.rho1
        return grad_r * scale, grad_z * scale

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
        # Discrete modes contribution
        p = self._discrete_modes_pressure(r, z)

        # Add continuous spectrum if requested
        if not self.discrete_modes_only and z <= self.H:
            p += self._continuous_spectrum_pressure(r, z)

        # Conjugate for exp(+iωt) time convention
        return np.conj(p)

    def pressure_gradient(self, r: float, z: float) -> tuple[complex, complex]:
        """
        Compute pressure gradient at position (r, z).

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
        # Discrete modes contribution
        grad_r, grad_z = self._discrete_modes_gradient(r, z)

        # Add continuous spectrum if requested
        if not self.discrete_modes_only and z <= self.H:
            cont_r, cont_z = self._continuous_spectrum_gradient(r, z)
            grad_r += cont_r
            grad_z += cont_z

        # Correct for z convention and conjugate for exp(+iωt)
        return np.conj(grad_r), np.conj(-grad_z)

    def pressure_field(self, r_array: np.ndarray, z_array: np.ndarray,
                      show_progress: bool = True) -> np.ndarray:
        """
        Compute pressure field on a grid.

        Parameters
        ----------
        r_array : np.ndarray
            1D array of radial distances (m)
        z_array : np.ndarray
            1D array of depths (m)
        show_progress : bool
            Print progress indicator (useful when including continuous spectrum)

        Returns
        -------
        np.ndarray
            2D complex array of pressure values, shape (len(z_array), len(r_array))
        """
        P = np.zeros((len(z_array), len(r_array)), dtype=complex)

        total = len(z_array) * len(r_array)
        for i, z in enumerate(z_array):
            for j, r in enumerate(r_array):
                P[i, j] = self.pressure(r, z)
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Progress: {(i+1)*len(r_array)}/{total} points")

        return P

    def __repr__(self):
        mode_str = "discrete only" if self.discrete_modes_only else "discrete + continuous"
        return (f"PekerisWaveguide(omega={self.omega}, c1={self.c1}, c2={self.c2}, "
                f"rho1={self.rho1}, rho2={self.rho2}, H={self.H}, z_s={self.z_s}, "
                f"n_modes={self.n_modes}, mode={mode_str})")


def example(show_plot=True, discrete_only=False):
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
    wg = PekerisWaveguide(omega, c1, c2, rho1, rho2, H, z_s,
                          discrete_modes_only=discrete_only)

    print(f"Pekeris Waveguide Parameters:")
    print(f"  omega = {omega} rad/s (f = {omega/(2*np.pi):.2f} Hz)")
    print(f"  c1 = {c1} m/s, c2 = {c2} m/s")
    print(f"  rho1 = {rho1} kg/m³, rho2 = {rho2} kg/m³")
    print(f"  H = {H} m, z_s = {z_s} m")
    print(f"\nNumber of discrete modes: {wg.n_modes}")
    print(f"Continuous spectrum: {'disabled' if discrete_only else 'enabled'}")
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


def plot_field(wg: PekerisWaveguide, r_max: float = 1000.0, z_max: float = None,
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

    mode_str = "discrete only" if wg.discrete_modes_only else "discrete + continuous"
    print(f"\nComputing pressure field on {nr}x{nz} grid ({mode_str})...")
    P = wg.pressure_field(r_array, z_array)

    # Compute transmission loss: TL = -20*log10(|p| * r) relative to 1m
    # For display, use magnitude in dB relative to max
    P_abs = np.abs(P)
    P_db = 20 * np.log10(P_abs / P_abs.max() + 1e-20)

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
    vlim = np.max(np.abs(P_real)) * 0.02
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pekeris waveguide solver')
    parser.add_argument('--discrete-only', action='store_true',
                       help='Use discrete modes only (skip continuous spectrum)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    args = parser.parse_args()

    example(show_plot=not args.no_plot, discrete_only=args.discrete_only)
