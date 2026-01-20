"""
Pekeris waveguide exact solution - discrete modes only.

This module computes the acoustic pressure field in a two-layer Pekeris waveguide:
- Layer 1 (water): constant sound speed c1, density rho1, depth H
- Layer 2 (sediment): constant sound speed c2 > c1, density rho2, semi-infinite

The solution uses the normal mode representation with discrete (trapped) modes.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, yv  # Bessel functions J_n and Y_n


class PekerisWaveguide:
    """
    Pekeris waveguide solver for discrete normal modes.

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

        # Wavenumbers
        self.k1 = omega / c1  # wavenumber in water
        self.k2 = omega / c2  # wavenumber in sediment
        self.k1_sqrd = self.k1**2
        self.k2_sqrd = self.k2**2

        # Density ratio
        self.M = rho2 / rho1

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

        p = 0j

        # Hankel function of second kind: H0^(2)(x) = J0(x) - i*Y0(x)
        def hankel2_0(x):
            return jv(0, x) - 1j * yv(0, x)

        if z <= self.H:
            # In water layer
            for n in range(self.n_modes):
                modal_amp = self.A_sqrd[n] * np.sin(self.k1_z[n] * self.z_s) * np.sin(self.k1_z[n] * z)
                p += modal_amp * hankel2_0(self.k_r[n] * r)
        else:
            # In sediment (exponential decay)
            for n in range(self.n_modes):
                modal_amp = (self.A_sqrd[n] * np.sin(self.k1_z[n] * self.z_s) *
                            np.sin(self.k1_z[n] * self.H) * np.exp(self.k2_z[n] * (self.H - z)))
                p += modal_amp * hankel2_0(self.k_r[n] * r)

        # Scale factor (from normal mode theory)
        p *= 1j * np.pi / self.rho1

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
        if self.n_modes == 0:
            return 0j, 0j

        grad_r = 0j
        grad_z = 0j

        # Hankel functions
        def hankel2_0(x):
            return jv(0, x) - 1j * yv(0, x)

        def hankel2_1(x):
            return jv(1, x) - 1j * yv(1, x)

        if z <= self.H:
            # In water layer
            for n in range(self.n_modes):
                sin_zs = np.sin(self.k1_z[n] * self.z_s)
                sin_z = np.sin(self.k1_z[n] * z)
                cos_z = np.cos(self.k1_z[n] * z)
                H0 = hankel2_0(self.k_r[n] * r)
                H1 = hankel2_1(self.k_r[n] * r)

                # d/dr: derivative of H0^(2)(kr*r) is -kr * H1^(2)(kr*r)
                grad_r += -self.k_r[n] * self.A_sqrd[n] * sin_zs * sin_z * H1

                # d/dz: derivative of sin(k1z*z) is k1z * cos(k1z*z)
                grad_z += self.k1_z[n] * self.A_sqrd[n] * sin_zs * cos_z * H0
        else:
            # In sediment
            for n in range(self.n_modes):
                sin_zs = np.sin(self.k1_z[n] * self.z_s)
                sin_H = np.sin(self.k1_z[n] * self.H)
                exp_decay = np.exp(self.k2_z[n] * (self.H - z))
                H0 = hankel2_0(self.k_r[n] * r)
                H1 = hankel2_1(self.k_r[n] * r)

                modal_base = self.A_sqrd[n] * sin_zs * sin_H * exp_decay

                grad_r += -self.k_r[n] * modal_base * H1

                # d/dz of exp(k2z*(H-z)) is -k2z * exp(k2z*(H-z))
                grad_z += -self.k2_z[n] * modal_base * H0

        # Scale factor
        scale = 1j * np.pi / self.rho1
        grad_r *= scale
        grad_z *= scale

        # Conjugate for exp(+iωt) time convention
        return np.conj(grad_r), np.conj(grad_z)

    def pressure_field(self, r_array: np.ndarray, z_array: np.ndarray) -> np.ndarray:
        """
        Compute pressure field on a grid.

        Parameters
        ----------
        r_array : np.ndarray
            1D array of radial distances (m)
        z_array : np.ndarray
            1D array of depths (m)

        Returns
        -------
        np.ndarray
            2D complex array of pressure values, shape (len(z_array), len(r_array))
        """
        R, Z = np.meshgrid(r_array, z_array)
        P = np.zeros_like(R, dtype=complex)

        for i, z in enumerate(z_array):
            for j, r in enumerate(r_array):
                P[i, j] = self.pressure(r, z)

        return P

    def __repr__(self):
        return (f"PekerisWaveguide(omega={self.omega}, c1={self.c1}, c2={self.c2}, "
                f"rho1={self.rho1}, rho2={self.rho2}, H={self.H}, z_s={self.z_s}, "
                f"n_modes={self.n_modes})")


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
    wg = PekerisWaveguide(omega, c1, c2, rho1, rho2, H, z_s)

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

    print(f"\nComputing pressure field on {nr}x{nz} grid...")
    P = wg.pressure_field(r_array, z_array)

    # Compute transmission loss: TL = -20*log10(|p| * r) relative to 1m
    # For display, use magnitude in dB relative to max
    P_abs = np.abs(P)
    P_db = 20 * np.log10(P_abs / P_abs.max() + 1e-20)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Magnitude (dB)
    ax1 = axes[0]
    im1 = ax1.pcolormesh(r_array, z_array, P_db, shading='auto', cmap='viridis', vmin=-40, vmax=0)
    ax1.axhline(wg.H, color='white', linestyle='--', linewidth=1, label='Seafloor')
    ax1.axhline(wg.z_s, color='red', linestyle=':', linewidth=1, label=f'Source (z={wg.z_s}m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title(f'Pressure Magnitude (dB re max) - {wg.n_modes} discrete modes')
    ax1.invert_yaxis()
    ax1.legend(loc='lower right')
    cbar1 = fig.colorbar(im1, ax=ax1, label='dB')

    # Plot 2: Real part (shows interference pattern)
    ax2 = axes[1]
    P_real = np.real(P)
    vlim = np.max(np.abs(P_real)) * 0.5
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
    plt.savefig('pekeris_field.png', dpi=150)
    print("Saved figure to pekeris_field.png")
    if not save_only:
        plt.show()


if __name__ == "__main__":
    example()
