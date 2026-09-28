"""
Example nk.py-style material file for the laser endpoint simulator.

Same conventions as the UCSB Nanofab / EMpy nk.py:
  * wavelength `wl` is in MICRONS
  * a material is a function  f(wl)  or, for an alloy,  f(x, wl)
  * optionally accept `k=False`; when called with k=True return a complex index.
    Either n - ik or n + ik is fine; the simulator uses |Im| as the loss.

Import it with "Import nk.py…" in the desktop app. Functions whose first argument is named
p / params / coeffs / base (fitting helpers like `cauchy` below) are ignored.
The values here are illustrative only; replace them with your own measured data.
"""
import numpy as np

# --- 1. A fit formula (fitting helper + material) ---------------------------------------
cauchy = lambda p, x: p[0] + p[1] / x**2 + p[2] / x**4


def My_PECVD_SiN(wl):
    """PECVD SiN, Cauchy fit (example coefficients)."""
    return cauchy([1.95, 0.012, 0.0], wl)


# --- 2. Tabulated n, k data, interpolated (e.g. from ellipsometry) ----------------------
_wl_tab = np.array([0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.10])   # microns
_n_tab = np.array([3.44, 3.40, 3.37, 3.33, 3.30, 3.28, 3.26])
_k_tab = np.array([0.020, 0.008, 0.002, 0.0, 0.0, 0.0, 0.0])


def My_InGaP_ellips(wl, k=False):
    """Lattice-matched InGaP from ellipsometry (example data), valid 0.60-1.10 um."""
    if np.any(np.atleast_1d(wl) < _wl_tab[0]) or np.any(np.atleast_1d(wl) > _wl_tab[-1]):
        raise ValueError("My_InGaP_ellips: wavelength outside 0.60-1.10 um")
    n = np.interp(wl, _wl_tab, _n_tab)
    if k:
        return n - 1j * np.interp(wl, _wl_tab, _k_tab)
    return n


# --- 3. An alloy: first argument is the composition x -----------------------------------
def My_AlGaAs_linear(x, wl):
    """Toy AlGaAs: linear mix of two single-oscillator end points (illustration only)."""
    n_gaas = np.sqrt(1 + 9.9 / (1 - (0.40 / wl) ** 2))
    n_alas = np.sqrt(1 + 7.9 / (1 - (0.30 / wl) ** 2))
    return n_gaas + (n_alas - n_gaas) * x
