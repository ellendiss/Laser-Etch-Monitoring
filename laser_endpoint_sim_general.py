#!/usr/bin/env python3
"""
Laser endpoint simulator for AlGaAs/GaAs plasma etching (offline desktop version).

Simulates the reflectance signal a laser etch monitor (670 nm or 980 nm) sees while an
AlGaAs/GaAs stack (e.g. a DBR) is etched from the top, using the thin-film transfer
matrix method at normal incidence. Same approach as the UCSB Nanofab EMpy script
(https://wiki.nanofab.ucsb.edu/wiki/Laser_Etch_Monitor_Simulation_in_Python).

Requirements: Python 3.8+, numpy, matplotlib (tkinter ships with standard Python).
    pip install numpy matplotlib
Run:
    python laser_endpoint_sim.py
    python laser_endpoint_sim_general.py --selftest            (checks the physics, no GUI)
    python laser_endpoint_sim_general.py --selftest nk.py      (also checks an nk.py material file)

Importing materials from nk.py files
    Like the UCSB Nanofab script, this tool can use material definitions from an nk.py-style
    Python file (e.g. the nk.py that ships with EMpy, or your own). Use "Import nk.py..." in the
    app. Every function in the file of the form  f(wl)  or  f(x, wl)  (wl in MICRONS, x = alloy
    fraction) that returns a refractive index becomes a selectable material for layers and the
    substrate. If a function accepts  k=True  it is called that way to get the complex index;
    the loss is taken as |Im(n)|, so both n - ik and n + ik conventions work.
    Importing runs the file's Python code, so only import files you trust.

Stack files (.json) are interchangeable with the web version of this tool (nk.py materials
are carried in the file as their stored n, k values at 670 and 980 nm).
"""

import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import warnings

import numpy as np

# =====================================================================================
# Physics
# =====================================================================================
HC = 1239.84198  # eV*nm
LASERS = (670, 980)

# Tabulated n, k (k > 0 = loss) from the UCSB/EMpy nk.py database (Filmetrics data)
REF = {
    670: {"GaAs": (3.806241, 0.165506), "AlAs": (3.096836, 0.0)},
    980: {"GaAs": (3.523308, 0.0), "AlAs": (2.967379, 0.0)},
}


def gamma_gap(x):
    """Direct (Gamma) band gap of AlxGa1-xAs in eV, as used with the Afromowitz model."""
    return 1.424 + 1.266 * x + 0.26 * x * x


def afromowitz_n(x, wl):
    """Afromowitz (1974) modified single-oscillator index, broadened (0.2 eV^2) at the gap."""
    E = HC / wl
    E0 = 3.65 + 0.871 * x + 0.179 * x * x
    Ed = 36.1 - 2.45 * x
    EG = gamma_gap(x)
    eta = math.pi * Ed / (2 * E0 ** 3 * (E0 ** 2 - EG ** 2))
    den = math.sqrt((EG ** 2 - E ** 2) ** 2 + 0.2 ** 2)
    eps = (1 + Ed / E0 + Ed * E ** 2 / E0 ** 3
           + eta / math.pi * E ** 4 * math.log((2 * E0 ** 2 - EG ** 2 - E ** 2) / den))
    return math.sqrt(eps)


def band_edge_k(x, wl):
    """Direct-gap absorption edge, alpha ~ sqrt(E - Eg), anchored to measured GaAs at 670 nm."""
    E, Eg = HC / wl, gamma_gap(x)
    if E <= Eg:
        return 0.0
    k0 = REF[670]["GaAs"][1]
    return k0 * math.sqrt((E - Eg) / (HC / 670 - gamma_gap(0))) * (wl / 670)


def algaas_nk(x, wl, model):
    """(n, k) of AlxGa1-xAs at laser wavelength wl. model: 'afro' or 'linear' (UCSB/EMpy)."""
    x = min(1.0, max(0.0, x))
    if model == "linear":
        g, a = REF[wl]["GaAs"], REF[wl]["AlAs"]
        return g[0] + (a[0] - g[0]) * x, g[1] + (a[1] - g[1]) * x
    return afromowitz_n(x, wl), band_edge_k(x, wl)


def quarter_wave(x, wl_design):
    return round(wl_design / (4 * afromowitz_n(x, wl_design)), 2)


def simulate(layers, substrate, wl, step, over):
    """
    Reflectance vs etch depth at normal incidence from vacuum.
    layers: list of dicts {n, k, d (nm), rate (nm/min)}, surface first.
    substrate: dict {n, k, rate}. Returns dict of numpy arrays.
    """
    nL = len(layers)
    N = [L["n"] - 1j * L["k"] for L in layers]  # n - ik convention (Macleod)
    Ns = substrate["n"] - 1j * substrate["k"]

    def apply(Nj, d, vB, vC):
        delta = 2 * np.pi * Nj * d / wl
        c, s = np.cos(delta), np.sin(delta)
        return c * vB + 1j * s / Nj * vC, 1j * Nj * s * vB + c * vC

    # below[j] = [B, C] of everything underneath layer j (layers j+1.. plus substrate)
    below = [None] * (nL + 1)
    below[nL] = (1.0 + 0j, Ns)
    for j in range(nL - 1, -1, -1):
        below[j] = below[nL] if j == nL - 1 else apply(N[j + 1], layers[j + 1]["d"], *below[j + 1])

    d = np.array([L["d"] for L in layers], dtype=float)
    bottoms = np.cumsum(d) if nL else np.zeros(0)
    tops = bottoms - d
    total = float(bottoms[-1]) if nL else 0.0
    end = total + over
    depth = np.unique(np.round(np.concatenate([np.arange(0, end + 1e-9, step), tops, [total, end]]), 6))
    depth = depth[depth <= end + 1e-9]
    idx = np.searchsorted(bottoms, depth + 1e-9, side="right") if nL else np.zeros(len(depth), int)

    B = np.empty(len(depth), complex)
    C = np.empty(len(depth), complex)
    n_s = np.empty(len(depth))
    k_s = np.empty(len(depth))
    time = np.empty(len(depth))
    rates = np.array([L["rate"] for L in layers], dtype=float)
    t_top = np.concatenate([[0.0], np.cumsum(d / rates * 60.0)]) if nL else np.array([0.0])
    for j in range(nL + 1):
        m = idx == j
        if not m.any():
            continue
        z = depth[m]
        if j < nL:
            B[m], C[m] = apply(N[j], bottoms[j] - z, *below[j])
            n_s[m], k_s[m] = layers[j]["n"], layers[j]["k"]
            time[m] = t_top[j] + (z - tops[j]) / rates[j] * 60.0
        else:
            B[m], C[m] = below[nL]
            n_s[m], k_s[m] = substrate["n"], substrate["k"]
            time[m] = t_top[nL] + (z - total) / substrate["rate"] * 60.0
    r = (B - C) / (B + C)
    idx = np.where(idx >= nL, -1, idx)
    return {"depth": depth, "time": time, "R": np.abs(r) ** 2, "n": n_s, "k": k_s, "idx": idx, "total": total}


# =====================================================================================
# Stack model (same JSON schema as the web version)
# =====================================================================================
def new_layer(name, x, d):
    return {"type": "layer", "name": name, "mat": "algaas", "x": x, "d": d,
            "n670": 2, "k670": 0, "n980": 2, "k980": 0, "rate": 50}


def new_custom(name, n, d, rate):
    return {"type": "layer", "name": name, "mat": "custom", "x": 0, "d": d,
            "n670": n, "k670": 0, "n980": n, "k980": 0, "rate": rate}


def new_dbr(name, periods, xa, xb, qw):
    return {"type": "dbr", "name": name, "periods": periods,
            "a": {"x": xa, "d": quarter_wave(xa, qw)}, "b": {"x": xb, "d": quarter_wave(xb, qw)}, "qw": qw}


PRESETS = {
    "850 nm VCSEL-type mesa (Al0.12 / Al0.90)": lambda: {"substrate": {"x": 0}, "blocks": [
        new_layer("GaAs contact cap", 0, 10),
        new_dbr("Top DBR", 20, 0.12, 0.90, 850),
        new_layer("Cavity (1λ, Al0.30)", 0.30, round(850 / afromowitz_n(0.30, 850), 2)),
        new_dbr("Bottom DBR", 6, 0.90, 0.12, 850)]},
    "980 nm DBR (GaAs / Al0.90), 20 pairs": lambda: {"substrate": {"x": 0}, "blocks": [
        new_dbr("DBR", 20, 0, 0.90, 980)]},
    "UCSB wiki example stack (EMpy)": lambda: {"substrate": {"x": 0}, "blocks": [
        new_custom("Si3N4", 1.997, 300, 50),
        {"type": "dbr", "name": "Upper DBR", "periods": 5, "a": {"x": 0, "d": 79.36}, "b": {"x": 0.95, "d": 92.51}, "qw": 1100},
        new_layer("GaAs core", 0, 551.75),
        {"type": "dbr", "name": "Lower DBR", "periods": 5, "a": {"x": 0.95, "d": 92.51}, "b": {"x": 0, "d": 79.36}, "qw": 1100},
        new_layer("AlGaAs spacer", 0.95, 500),
        new_layer("GaAs spacer", 0, 300)]},
    "Single GaAs layer on Al0.90": lambda: {"substrate": {"x": 0.9}, "blocks": [new_layer("GaAs", 0, 1000)]},
}


def fnum(v, default=0.0):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def clampx(x):
    return min(1.0, max(0.0, fnum(x)))


# =====================================================================================
# nk.py material import (UCSB/EMpy-style material files)
# =====================================================================================
class NkUnavailable(Exception):
    """An imported material cannot give an index at the requested wavelength."""


# Fitting helpers in nk.py files that are not materials themselves
NK_HELPERS = {"check_wl", "fillfraction", "cauchy", "sellmeier", "sellmeier1", "sellmeier5",
              "lorentzian", "wave", "wavelength", "interp"}


class NkLibrary:
    """Materials discovered in one or more imported nk.py-style files."""

    def __init__(self):
        self.files = []       # absolute paths, in import order
        self.materials = {}   # key -> dict(name, func, alloy, has_k, file, doc)
        self._cache = {}

    def load(self, path):
        """Import an nk.py file. Returns (added_keys, skipped[(name, reason)]). Re-importing replaces it."""
        path = os.path.abspath(path)
        modname = "nkimport_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            raise ImportError("not a Python file")
        mod = importlib.util.module_from_spec(spec)
        folder = os.path.dirname(path)
        added_path = folder not in sys.path
        if added_path:  # let the file import its own helper modules from the same folder
            sys.path.insert(0, folder)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec.loader.exec_module(mod)
        finally:
            if added_path and folder in sys.path:
                sys.path.remove(folder)

        self.materials = {k: v for k, v in self.materials.items() if v["file"] != path}
        self._cache.clear()
        stem = os.path.splitext(os.path.basename(path))[0]
        added, skipped = [], []
        for name, obj in list(vars(mod).items()):
            if name.startswith("_") or name in NK_HELPERS or not callable(obj) or inspect.isclass(obj):
                continue
            if getattr(obj, "__module__", None) != modname:
                continue  # numpy etc. imported into the file
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            required = [q for q in sig.parameters.values()
                        if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD) and q.default is q.empty]
            has_k = "k" in sig.parameters or any(q.kind == q.VAR_KEYWORD for q in sig.parameters.values())
            if len(required) == 1:
                alloy = False
            elif len(required) == 2 and required[0].name.lower() not in ("p", "params", "coeffs", "base"):
                alloy = True
            else:
                continue
            doc = (inspect.getdoc(obj) or "").strip().splitlines()
            mat = {"name": name, "func": obj, "alloy": alloy, "has_k": has_k, "file": path, "doc": doc[0] if doc else ""}
            reasons = []
            for wl in LASERS:
                try:
                    self._call(mat, wl, 0.5 if alloy else None)
                    break
                except NkUnavailable as e:
                    reasons.append(str(e))
            else:
                skipped.append((name, reasons[-1] if reasons else "no usable value"))
                continue
            key = name if name not in self.materials else f"{stem}.{name}"
            mat["key"] = key
            self.materials[key] = mat
            added.append(key)
        if path not in self.files:
            self.files.append(path)
        return added, skipped

    @staticmethod
    def _call(mat, wl_nm, x):
        wl = wl_nm / 1000.0  # nk.py convention: wavelength in microns
        args = (x, wl) if mat["alloy"] else (wl,)

        def invoke():
            if mat["has_k"]:
                try:
                    return mat["func"](*args, k=True)
                except TypeError:
                    pass
            return mat["func"](*args)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                val = invoke()
            except Exception as e:
                msg = str(e).split("\n")[0]
                raise NkUnavailable(f"{mat['name']} has no value at {wl_nm} nm ({msg[:140]})")
        try:
            arr = np.asarray(val, dtype=complex).ravel()
            v = complex(arr[0])
        except Exception:
            raise NkUnavailable(f"{mat['name']} did not return a number at {wl_nm} nm")
        n, k = v.real, abs(v.imag)
        if not (math.isfinite(n) and math.isfinite(k)) or not 1.0 <= n <= 10.0 or k > 20:
            raise NkUnavailable(f"{mat['name']} returned an implausible index ({v:.4g}) at {wl_nm} nm")
        notes = [str(w.message) for w in caught]
        return n, k, notes

    def eval(self, key, wl, x=None):
        """(n, k, notes) for material `key` at laser wavelength wl (nm). Raises NkUnavailable."""
        mat = self.materials.get(key)
        if mat is None:
            raise NkUnavailable(f"{key} is not loaded; import its nk.py file")
        xx = clampx(x) if mat["alloy"] else None
        ck = (key, wl, xx)
        if ck not in self._cache:
            try:
                self._cache[ck] = self._call(mat, wl, xx)
            except NkUnavailable as e:
                self._cache[ck] = e
        r = self._cache[ck]
        if isinstance(r, Exception):
            raise r
        return r


def refresh_nk_cache(stack):
    """Store current n, k of every nk.py material in the stack itself (portable stack files)."""
    items = [b for b in stack.blocks if b.get("type") == "layer" and b.get("mat") == "nk"]
    if stack.substrate.get("mat") == "nk":
        items.append(stack.substrate)
    for b in items:
        key = b.get("nkfunc")
        if stack.lib is None or key not in stack.lib.materials:
            continue
        b["nkalloy"] = stack.lib.materials[key]["alloy"]
        vals = {}
        try:
            for wl in LASERS:
                n, k, _ = stack.lib.eval(key, wl, b.get("nkx"))
                vals[f"n{wl}"], vals[f"k{wl}"] = round(n, 6), round(k, 6)
        except NkUnavailable:
            continue
        b.update(vals)
        b["nkcached"] = True


def mat_label(m):
    if m.get("mat") == "nk":
        name = m.get("nkfunc") or "nk.py material"
        return f"{name} (x={clampx(m.get('nkx')):g})" if m.get("nkalloy") else name
    if m.get("mat") == "custom":
        return "Custom"
    x = clampx(m.get("x", 0))
    if x == 0:
        return "GaAs"
    if x == 1:
        return "AlAs"
    return f"Al{x:g}Ga{round(1 - x, 3):g}As"


class Stack:
    def __init__(self):
        self.model = "afro"
        self.rateG = 100.0
        self.rateA = 100.0
        self.substrate = {"x": 0}
        self.blocks = []
        self.lib = None  # NkLibrary for nk.py materials

    def nk(self, m, wl):
        if m.get("mat") == "nk":
            key = m.get("nkfunc")
            if self.lib is not None and key in self.lib.materials:
                n, k, _ = self.lib.eval(key, wl, m.get("nkx"))
                return n, k
            if m.get("nkcached") and m.get(f"n{wl}") is not None:  # file not loaded: use stored values
                return max(0.01, fnum(m.get(f"n{wl}"), 1.0)), max(0.0, fnum(m.get(f"k{wl}")))
            raise NkUnavailable(f"{key} is not loaded; import its nk.py file")
        if m.get("mat") == "custom":
            return max(0.01, fnum(m.get(f"n{wl}"), 1.0)), max(0.0, fnum(m.get(f"k{wl}")))
        return algaas_nk(clampx(m.get("x")), wl, self.model)

    def rate(self, m):
        if m.get("mat") in ("custom", "nk"):
            return max(0.01, fnum(m.get("rate"), 1.0))
        x = clampx(m.get("x"))
        g, a = max(0.01, fnum(self.rateG, 1)), max(0.01, fnum(self.rateA, 1))
        return g + (a - g) * x

    def flatten(self):
        out = []
        for bi, b in enumerate(self.blocks):
            if b["type"] == "dbr":
                P = max(0, min(500, int(round(fnum(b.get("periods"))))))
                for p in range(P):
                    for key, tag in (("a", "A"), ("b", "B")):
                        m = {"mat": "algaas", "x": clampx(b[key]["x"])}
                        d = fnum(b[key]["d"])
                        if d > 0:
                            out.append({**m, "d": d, "block": bi,
                                        "label": f"{b.get('name') or 'DBR'}, pair {p + 1}{tag}, {mat_label(m)}"})
            else:
                d = fnum(b.get("d"))
                if d > 0:
                    m = {k: b.get(k) for k in ("mat", "x", "n670", "k670", "n980", "k980", "rate",
                                               "nkfunc", "nkx", "nkalloy", "nkcached")}
                    m["x"] = clampx(m["x"])
                    out.append({**m, "d": d, "block": bi, "label": f"{b.get('name') or 'Layer'}, {mat_label(b)}"})
        return out


def run_all(stack, step, over):
    """Simulate both lasers. A laser whose materials can't be evaluated gets sims[wl]['invalid'] = reason."""
    flat = stack.flatten()
    if stack.substrate.get("mat") == "nk":
        sub = dict(stack.substrate)
    else:
        sub = {"mat": "algaas", "x": clampx(stack.substrate.get("x"))}
    sims = {}
    for wl in LASERS:
        invalid = None
        try:
            L = [dict(zip(("n", "k"), stack.nk(m, wl)), d=m["d"], rate=stack.rate(m)) for m in flat]
            ns, ks = stack.nk(sub, wl)
        except NkUnavailable as e:  # keep the geometry (depth/time) but mark the trace unusable
            invalid = str(e)
            L = [dict(n=1.0, k=0.0, d=m["d"], rate=stack.rate(m)) for m in flat]
            ns, ks = 1.0, 0.0
        sims[wl] = simulate(L, {"n": ns, "k": ks, "rate": stack.rate(sub)}, wl, step, over)
        sims[wl]["invalid"] = invalid
    return flat, sub, sims


# =====================================================================================
# Self-test (no GUI): compare against the UCSB EMpy reference at a few depths
# =====================================================================================
def selftest(nk_path=None):
    s = Stack()
    s.model = "linear"
    s.__dict__.update(PRESETS["UCSB wiki example stack (EMpy)"]())
    # exact EMpy example thicknesses
    for b in s.blocks:
        if b["type"] == "dbr":
            for key in ("a", "b"):
                b[key]["d"] = 79.35620377815292 if b[key]["x"] == 0 else 92.50806354194499
    _, _, sims = run_all(s, 10, 0)
    r = sims[670]
    # Reference values from running the UCSB/EMpy ex_laser_etch_monitor.py script
    empy = {0: 21.589606, 300: 30.384570, 1000: 28.288907, 2000: 32.493645, 3000: 24.723039}
    print("UCSB wiki example stack, 670 nm, UCSB linear model")
    print("  depth (nm)   this tool (%)   EMpy (%)")
    worst = 0.0
    for z, ref in empy.items():
        i = int(np.argmin(np.abs(r["depth"] - z)))
        worst = max(worst, abs(100 * r["R"][i] - ref))
        print(f"  {z:10.0f}   {100 * r['R'][i]:13.5f}   {ref:8.5f}")
    print(f"  max difference: {worst:.2e} %  ->  {'PASS' if worst < 1e-3 else 'FAIL'}")
    print("\nAfromowitz + band-edge optical constants:")
    for x in (0, 0.12, 0.3, 0.9, 1):
        print(f"  x={x:<4}  670 nm: n={afromowitz_n(x, 670):.4f} k={band_edge_k(x, 670):.4f}   "
              f"980 nm: n={afromowitz_n(x, 980):.4f} k={band_edge_k(x, 980):.4f}")
    if not nk_path:
        return
    lib = NkLibrary()
    added, skipped = lib.load(nk_path)
    print(f"\nImported {len(added)} materials from {nk_path} ({len(skipped)} skipped)")
    for key in added:
        m = lib.materials[key]
        vals = []
        for wl in LASERS:
            try:
                n, k, notes = lib.eval(key, wl, 0.5 if m["alloy"] else None)
                vals.append(f"{wl}: {n:.4f}{'-' if k else '+'}{k:.4f}i{' (out of range)' if notes else ''}")
            except NkUnavailable:
                vals.append(f"{wl}: n/a")
        print(f"  {key + (' (x=0.5)' if m['alloy'] else ''):28s} " + "   ".join(vals))
    if "GaAs_interp" in lib.materials and "AlGaAs_interp" in lib.materials:
        # Rebuild the UCSB example entirely from the imported file's materials; must match EMpy
        g = lambda d: {"type": "layer", "name": "GaAs", "mat": "nk", "nkfunc": "GaAs_interp", "d": d, "rate": 100}
        a = lambda d: {"type": "layer", "name": "AlGaAs", "mat": "nk", "nkfunc": "AlGaAs_interp", "nkx": 0.95, "d": d, "rate": 100}
        G, A = 79.35620377815292, 92.50806354194499
        s2 = Stack()
        s2.lib = lib
        s2.blocks = ([new_custom("Si3N4", 1.997, 300, 50)] + [g(G), a(A)] * 5 + [g(551.75)] + [a(A), g(G)] * 5
                     + [a(500), g(300)])
        s2.substrate = {"mat": "nk", "nkfunc": "GaAs_interp", "rate": 100}
        _, _, sims2 = run_all(s2, 10, 0)
        r2 = sims2[670]
        worst = max(abs(100 * r2["R"][int(np.argmin(np.abs(r2["depth"] - z)))] - ref) for z, ref in empy.items())
        print(f"\nUCSB example built from imported GaAs_interp / AlGaAs_interp vs EMpy: "
              f"max difference {worst:.2e} %  ->  {'PASS' if worst < 1e-3 else 'FAIL'}")


# =====================================================================================
# GUI
# =====================================================================================
def main_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    if sys.platform == "win32":  # crisp text on high-DPI Windows displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    LASER_COLOR = {670: "#C8102E", 980: "#5B3FA8"}
    GAAS, ALAS, CUSTOM, NKCOLOR = (137, 149, 160), (207, 228, 221), "#E4D7B5", "#B9CCE8"
    AUTOSAVE = os.path.join(os.path.expanduser("~"), ".laser_endpoint_sim_general.json")

    def mat_color(m):
        if m.get("mat") == "nk":
            return NKCOLOR
        if m.get("mat") == "custom":
            return CUSTOM
        x = clampx(m.get("x"))
        return "#%02x%02x%02x" % tuple(round(a + (b - a) * x) for a, b in zip(GAAS, ALAS))

    MAT_CHOICES = ("AlGaAs / GaAs", "Custom n, k", "From nk.py")
    MAT_OF_CHOICE = {"AlGaAs / GaAs": "algaas", "Custom n, k": "custom", "From nk.py": "nk"}
    CHOICE_OF_MAT = {v: k for k, v in MAT_OF_CHOICE.items()}

    def fmt_time(s):
        m = int(s // 60)
        return f"{m} min {round(s - 60 * m)} s" if m else f"{round(s)} s"

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Laser endpoint simulator: AlGaAs/GaAs etch monitor")
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{min(1320, sw - 40)}x{min(840, sh - 90)}+10+10")
            self.minsize(1000, 640)
            self.stack = Stack()
            self.lib = NkLibrary()
            self.stack.lib = self.lib
            self._nk_trust_ok = False
            self._libwin = None
            self.stack.__dict__.update(PRESETS["850 nm VCSEL-type mesa (Al0.12 / Al0.90)"]())
            self.last = None
            self._job = None
            self._loading = False
            self.sel = None

            self.v_laser = tk.StringVar(value="670")
            self.v_model = tk.StringVar(value="afro")
            self.v_axis = tk.StringVar(value="depth")
            self.v_yfit = tk.StringVar(value="fit")
            self._full_x = None
            self._drawn_axis = None
            self.v_target = tk.StringVar(value="")
            self.v_rateG = tk.StringVar(value="100")
            self.v_rateA = tk.StringVar(value="100")
            self.v_step = tk.StringVar(value="1")
            self.v_over = tk.StringVar(value="300")
            self.v_subx = tk.StringVar(value="0")
            self.v_subnkx = tk.StringVar(value="0")
            self.v_status = tk.StringVar(value="Move the mouse over the plot to read values. Click to set the stop depth.")

            self._build()
            self._autoload()
            self._sync_globals_to_vars()
            self.refresh_tree()
            if self.lib.files:
                self.after_lib_change()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.schedule()

        # ---------------- layout ----------------
        def _build(self):
            style = ttk.Style(self)
            style.configure("Hint.TLabel", foreground="#566370")
            from tkinter import font as tkfont
            fam = tkfont.nametofont("TkDefaultFont").actual("family")
            style.configure("Big.TLabel", font=(fam, 15, "bold"))

            top = ttk.Frame(self, padding=(10, 8))
            top.pack(fill="x")
            ttk.Label(top, text="Laser:").pack(side="left")
            for val, txt in (("670", "670 nm"), ("980", "980 nm"), ("both", "Both")):
                ttk.Radiobutton(top, text=txt, value=val, variable=self.v_laser, command=self.schedule).pack(side="left", padx=3)
            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
            ttk.Label(top, text="Optical model:").pack(side="left")
            ttk.Radiobutton(top, text="Afromowitz + band edge", value="afro", variable=self.v_model, command=self.schedule).pack(side="left", padx=3)
            ttk.Radiobutton(top, text="UCSB linear", value="linear", variable=self.v_model, command=self.schedule).pack(side="left", padx=3)
            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
            ttk.Label(top, text="X axis:").pack(side="left")
            ttk.Radiobutton(top, text="Depth", value="depth", variable=self.v_axis, command=self.schedule).pack(side="left", padx=3)
            ttk.Radiobutton(top, text="Time", value="time", variable=self.v_axis, command=self.schedule).pack(side="left", padx=3)
            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
            ttk.Label(top, text="Reflectance axis:").pack(side="left")
            ttk.Radiobutton(top, text="Fit to trace", value="fit", variable=self.v_yfit, command=self.refit_y).pack(side="left", padx=3)
            ttk.Radiobutton(top, text="From 0", value="zero", variable=self.v_yfit, command=self.refit_y).pack(side="left", padx=3)
            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
            ttk.Label(top, text="Stop at depth (nm):").pack(side="left")
            e = ttk.Entry(top, textvariable=self.v_target, width=9, justify="right")
            e.pack(side="left", padx=4)
            self.v_target.trace_add("write", lambda *a: self.schedule())

            body = ttk.PanedWindow(self, orient="horizontal")
            body.pack(fill="both", expand=True, padx=8, pady=(0, 4))
            holder = ttk.Frame(body)
            right = ttk.Frame(body, padding=(4, 4))
            body.add(holder, weight=0)
            body.add(right, weight=1)
            # scrollable left panel so the editor fits on small laptop screens
            cv = tk.Canvas(holder, highlightthickness=0, width=455)
            vs = ttk.Scrollbar(holder, orient="vertical", command=cv.yview)
            cv.configure(yscrollcommand=vs.set)
            vs.pack(side="right", fill="y")
            cv.pack(side="left", fill="both", expand=True)
            left = ttk.Frame(cv, padding=(4, 4))
            win = cv.create_window((0, 0), window=left, anchor="nw")
            left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
            cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))

            def _wheel(e):
                if e.widget.winfo_class() in ("Treeview", "TCombobox"):
                    return
                if cv.winfo_height() >= left.winfo_reqheight():
                    return
                delta = -1 if (getattr(e, "num", 0) == 4 or e.delta > 0) else 1
                cv.yview_scroll(delta, "units")
            holder.bind_all("<MouseWheel>", lambda e: _wheel(e) if str(e.widget).startswith(str(holder)) else None)
            holder.bind_all("<Button-4>", lambda e: _wheel(e) if str(e.widget).startswith(str(holder)) else None)
            holder.bind_all("<Button-5>", lambda e: _wheel(e) if str(e.widget).startswith(str(holder)) else None)
            self._build_left(left)
            self._build_right(right)

            ttk.Label(self, textvariable=self.v_status, style="Hint.TLabel", padding=(12, 2, 12, 6)).pack(fill="x")

        def _build_left(self, f):
            row = ttk.Frame(f)
            row.pack(fill="x")
            self.cb_preset = ttk.Combobox(row, values=list(PRESETS), state="readonly")
            self.cb_preset.set("Choose a preset…")
            self.cb_preset.pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Load preset", command=self.load_preset).pack(side="left", padx=(4, 0))

            ttk.Label(f, text="Stack, surface first (etch starts at the top)", style="Hint.TLabel").pack(anchor="w", pady=(10, 2))
            tf = ttk.Frame(f)
            tf.pack(fill="both", expand=True)
            self.tree = ttk.Treeview(tf, columns=("type", "comp", "thk"), show="tree headings", height=7, selectmode="browse")
            self.tree.heading("#0", text="Name")
            self.tree.heading("type", text="Type")
            self.tree.heading("comp", text="Composition")
            self.tree.heading("thk", text="Total (nm)")
            self.tree.column("#0", width=140)
            self.tree.column("type", width=50, anchor="center")
            self.tree.column("comp", width=135)
            self.tree.column("thk", width=70, anchor="e")
            sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            sb.pack(side="left", fill="y")
            self.tree.bind("<<TreeviewSelect>>", self.on_select)

            bb = ttk.Frame(f)
            bb.pack(fill="x", pady=4)
            for txt, cmd in (("Add layer", self.add_layer), ("Add DBR", self.add_dbr), ("Duplicate", self.dup),
                             ("Remove", self.remove), ("Up", lambda: self.move(-1)), ("Down", lambda: self.move(1))):
                ttk.Button(bb, text=txt, command=cmd, width=len(txt) + 1).pack(side="left", padx=1)

            self.editor = ttk.LabelFrame(f, text="Selected item", padding=8)
            self.editor.pack(fill="x", pady=(6, 4))
            self._build_editors()

            sf = ttk.LabelFrame(f, text="Substrate and etch", padding=8)
            sf.pack(fill="x", pady=4)
            subf = ttk.Frame(sf)
            subf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
            ttk.Label(subf, text="Substrate").grid(row=0, column=0, sticky="w")
            self.cb_submat = ttk.Combobox(subf, values=("AlGaAs / GaAs", "From nk.py"), state="readonly", width=16)
            self.cb_submat.grid(row=0, column=1, sticky="w", pady=1)
            self.cb_submat.set("AlGaAs / GaAs")
            self.cb_submat.bind("<<ComboboxSelected>>", self.on_submat)
            self.lbl_subx = ttk.Label(subf, text="Al fraction x")
            self.ent_subx = ttk.Entry(subf, textvariable=self.v_subx, width=9, justify="right")
            self.lbl_subx.grid(row=1, column=0, sticky="w")
            self.ent_subx.grid(row=1, column=1, sticky="w", pady=1)
            self.lbl_subfunc = ttk.Label(subf, text="nk.py material")
            self.cb_subfunc = ttk.Combobox(subf, state="readonly", width=24)
            self.cb_subfunc.bind("<<ComboboxSelected>>", lambda e: (self.layout_substrate(), self.schedule()))
            self.lbl_subfunc.grid(row=2, column=0, sticky="w")
            self.cb_subfunc.grid(row=2, column=1, sticky="w", pady=1)
            self.lbl_subnkx = ttk.Label(subf, text="Alloy fraction x")
            self.ent_subnkx = ttk.Entry(subf, textvariable=self.v_subnkx, width=9, justify="right")
            self.lbl_subnkx.grid(row=3, column=0, sticky="w")
            self.ent_subnkx.grid(row=3, column=1, sticky="w", pady=1)
            subf.columnconfigure(0, minsize=120)
            for var in (self.v_subx, self.v_subnkx):
                var.trace_add("write", lambda *a: self.schedule())
            grid = [("GaAs etch rate (nm/min)", self.v_rateG),
                    ("AlAs etch rate (nm/min)", self.v_rateA), ("Depth step (nm)", self.v_step),
                    ("Etch past last layer (nm)", self.v_over)]
            for r, (lab, var) in enumerate(grid, start=1):
                ttk.Label(sf, text=lab).grid(row=r, column=0, sticky="w", pady=1)
                ttk.Entry(sf, textvariable=var, width=10, justify="right").grid(row=r, column=1, sticky="e", pady=1)
                var.trace_add("write", lambda *a: self.schedule())
            sf.columnconfigure(0, weight=1)
            ttk.Label(sf, text="Substrate is semi-infinite (no backside reflection). AlGaAs etch rate is\n"
                               "interpolated linearly in x; rates only affect the time axis.",
                      style="Hint.TLabel").grid(row=len(grid) + 1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            self.step_note = ttk.Label(sf, text="", foreground="#9A5B00", wraplength=380, justify="left")
            self.step_note.grid(row=len(grid) + 2, column=0, columnspan=2, sticky="w")

            ff = ttk.LabelFrame(f, text="Files and materials", padding=8)
            ff.pack(fill="x", pady=4)
            buttons = (("Load stack", self.load_json), ("Save stack", self.save_json), ("Export CSV", self.export_csv),
                       ("Save PNG", self.save_png), ("Import nk.py…", self.import_nk), ("Material library…", self.show_library))
            for i, (txt, cmd) in enumerate(buttons):
                ttk.Button(ff, text=txt, command=cmd).grid(row=i // 3, column=i % 3, sticky="ew", padx=1, pady=1)
            for c in range(3):
                ff.columnconfigure(c, weight=1)
            self.nk_status = ttk.Label(ff, text="No nk.py file imported.", style="Hint.TLabel", wraplength=380, justify="left")
            self.nk_status.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        def _build_editors(self):
            self.ed_vars = {k: tk.StringVar() for k in
                            ("name", "mat", "x", "d", "n670", "k670", "n980", "k980", "rate", "nkx",
                             "periods", "a.x", "a.d", "b.x", "b.d", "qw")}
            for key, var in self.ed_vars.items():
                var.trace_add("write", lambda *a, k=key: self.on_edit(k))

            self.ed_empty = ttk.Label(self.editor, text="Select a layer or DBR in the list to edit it.", style="Hint.TLabel")

            L = self.ed_layer = ttk.Frame(self.editor)
            ttk.Label(L, text="Name").grid(row=0, column=0, sticky="w")
            ttk.Entry(L, textvariable=self.ed_vars["name"]).grid(row=0, column=1, columnspan=3, sticky="ew", pady=1)
            ttk.Label(L, text="Material").grid(row=1, column=0, sticky="w")
            self.cb_mat = ttk.Combobox(L, values=MAT_CHOICES, state="readonly", width=16)
            self.cb_mat.grid(row=1, column=1, columnspan=3, sticky="w", pady=1)
            self.cb_mat.bind("<<ComboboxSelected>>", self.on_mat)
            self.lbl_x = ttk.Label(L, text="Al fraction x")
            self.ent_x = ttk.Entry(L, textvariable=self.ed_vars["x"], width=9, justify="right")
            self.lbl_x.grid(row=2, column=0, sticky="w")
            self.ent_x.grid(row=2, column=1, sticky="w", pady=1)
            ttk.Label(L, text="Thickness (nm)").grid(row=2, column=2, sticky="w", padx=(10, 0))
            ttk.Entry(L, textvariable=self.ed_vars["d"], width=9, justify="right").grid(row=2, column=3, sticky="w", pady=1)
            self.custom_frame = ttk.Frame(L)
            self.custom_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
            for c, (lab, key) in enumerate((("n 670", "n670"), ("k 670", "k670"), ("n 980", "n980"), ("k 980", "k980"), ("Rate nm/min", "rate"))):
                ttk.Label(self.custom_frame, text=lab).grid(row=0, column=c, sticky="w", padx=2)
                ttk.Entry(self.custom_frame, textvariable=self.ed_vars[key], width=8, justify="right").grid(row=1, column=c, padx=2)
            self.nk_frame = ttk.Frame(L)
            self.nk_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0))
            ttk.Label(self.nk_frame, text="nk.py material").grid(row=0, column=0, sticky="w")
            self.cb_nkfunc = ttk.Combobox(self.nk_frame, state="readonly", width=26)
            self.cb_nkfunc.grid(row=0, column=1, columnspan=3, sticky="w", pady=1)
            self.cb_nkfunc.bind("<<ComboboxSelected>>", self.on_nkfunc)
            self.lbl_nkx = ttk.Label(self.nk_frame, text="Alloy fraction x")
            self.ent_nkx = ttk.Entry(self.nk_frame, textvariable=self.ed_vars["nkx"], width=8, justify="right")
            self.lbl_nkx.grid(row=1, column=0, sticky="w")
            self.ent_nkx.grid(row=1, column=1, sticky="w", pady=1)
            ttk.Label(self.nk_frame, text="Etch rate (nm/min)").grid(row=1, column=2, sticky="w", padx=(10, 0))
            ttk.Entry(self.nk_frame, textvariable=self.ed_vars["rate"], width=8, justify="right").grid(row=1, column=3, sticky="w", pady=1)
            self.nk_info = ttk.Label(self.nk_frame, text="", style="Hint.TLabel", wraplength=370, justify="left")
            self.nk_info.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
            L.columnconfigure(1, weight=1)

            D = self.ed_dbr = ttk.Frame(self.editor)
            ttk.Label(D, text="Name").grid(row=0, column=0, sticky="w")
            ttk.Entry(D, textvariable=self.ed_vars["name"]).grid(row=0, column=1, columnspan=2, sticky="ew", pady=1)
            ttk.Label(D, text="Pairs").grid(row=1, column=0, sticky="w")
            ttk.Entry(D, textvariable=self.ed_vars["periods"], width=9, justify="right").grid(row=1, column=1, sticky="w", pady=1)
            ttk.Label(D, text="Al fraction x", style="Hint.TLabel").grid(row=2, column=1, sticky="w", pady=(6, 0))
            ttk.Label(D, text="Thickness (nm)", style="Hint.TLabel").grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
            for r, (lab, key) in enumerate((("Upper layer", "a"), ("Lower layer", "b")), start=3):
                ttk.Label(D, text=lab).grid(row=r, column=0, sticky="w")
                ttk.Entry(D, textvariable=self.ed_vars[key + ".x"], width=9, justify="right").grid(row=r, column=1, sticky="w", pady=1)
                ttk.Entry(D, textvariable=self.ed_vars[key + ".d"], width=9, justify="right").grid(row=r, column=2, sticky="w", padx=(8, 0), pady=1)
            ttk.Label(D, text="Quarter-wave at (nm)").grid(row=5, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(D, textvariable=self.ed_vars["qw"], width=9, justify="right").grid(row=5, column=1, sticky="w", pady=(6, 0))
            ttk.Button(D, text="Set thicknesses", command=self.set_qw).grid(row=5, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
            self.dbr_info = ttk.Label(D, text="", style="Hint.TLabel", wraplength=340, justify="left")
            self.dbr_info.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
            self.show_editor(None)

        def _build_right(self, f):
            vp = ttk.PanedWindow(f, orient="vertical")
            vp.pack(fill="both", expand=True)
            pf = ttk.Frame(vp)
            vp.add(pf, weight=3)
            self.fig = Figure(figsize=(5, 3.5), dpi=100, layout="constrained")
            gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 3.2])
            self.ax_n = self.fig.add_subplot(gs[0])
            self.ax_r = self.fig.add_subplot(gs[1], sharex=self.ax_n)
            self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
            self.toolbar = NavigationToolbar2Tk(self.canvas, pf, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side="bottom", fill="x")
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvas.mpl_connect("motion_notify_event", self.on_hover)
            self.canvas.mpl_connect("button_press_event", self.on_click)

            bf = ttk.Frame(vp, padding=(0, 6))
            vp.add(bf, weight=1)
            self.readout = ttk.Frame(bf)
            self.readout.pack(fill="x")
            self.ro_labels = []
            for c in range(4):
                fr = ttk.Frame(self.readout, padding=(8, 2))
                fr.grid(row=0, column=c, sticky="nw")
                self.readout.columnconfigure(c, weight=1)
                a = ttk.Label(fr, text="", style="Hint.TLabel")
                b = ttk.Label(fr, text="", style="Big.TLabel")
                cc = ttk.Label(fr, text="", style="Hint.TLabel")
                a.pack(anchor="w"); b.pack(anchor="w"); cc.pack(anchor="w")
                self.ro_labels.append((a, b, cc))
            self.ro_layer = ttk.Label(bf, text="", padding=(8, 2))
            self.ro_layer.pack(anchor="w")

            cols = ("n670", "k670", "f670", "n980", "k980", "f980", "rate")
            self.nk_tree = ttk.Treeview(bf, columns=cols, show="tree headings", height=5)
            self.nk_tree.heading("#0", text="Material")
            for c, t in zip(cols, ("n 670", "k 670", "nm per fringe, 670", "n 980", "k 980", "nm per fringe, 980", "Rate nm/min")):
                self.nk_tree.heading(c, text=t)
                self.nk_tree.column(c, width=55 if c[0] in "nk" else (70 if c == "rate" else 125), minwidth=40, anchor="e", stretch=True)
            self.nk_tree.column("#0", width=150, minwidth=80)
            self.nk_tree.pack(fill="both", expand=True, pady=(6, 0))

        # ---------------- state <-> widgets ----------------
        def _sync_globals_to_vars(self):
            s = self.stack
            self.v_model.set(s.model)
            self.v_rateG.set(f"{fnum(s.rateG, 100):g}")
            self.v_rateA.set(f"{fnum(s.rateA, 100):g}")
            self.v_subx.set(f"{clampx(s.substrate.get('x')):g}")
            nk = s.substrate.get("mat") == "nk"
            self.cb_submat.set("From nk.py" if nk else "AlGaAs / GaAs")
            self.cb_subfunc.set(s.substrate.get("nkfunc", "") if nk else "")
            self.v_subnkx.set(f"{clampx(s.substrate.get('nkx')):g}")
            self.layout_substrate()

        def _read_globals(self):
            s = self.stack
            s.model = self.v_model.get()
            s.rateG = fnum(self.v_rateG.get(), 100) or 100
            s.rateA = fnum(self.v_rateA.get(), 100) or 100
            if self.cb_submat.get() == "From nk.py" and self.cb_subfunc.get():
                key = self.cb_subfunc.get()
                new = {"mat": "nk", "x": clampx(self.v_subx.get()), "nkfunc": key, "nkx": clampx(self.v_subnkx.get()),
                       "rate": fnum(self.v_rateG.get(), 100) or 100}
                old = s.substrate
                if old.get("mat") == "nk" and old.get("nkfunc") == key:  # keep stored values for portability
                    for k in ("n670", "k670", "n980", "k980", "nkalloy", "nkcached"):
                        if k in old:
                            new[k] = old[k]
                s.substrate = new
            else:
                s.substrate = {"x": clampx(self.v_subx.get())}

        def block_summary(self, b):
            if b["type"] == "dbr":
                P = int(round(fnum(b.get("periods"))))
                a, c = b["a"], b["b"]
                comp = f"{P} × {mat_label({'x': a['x']})} / {mat_label({'x': c['x']})}"
                tot = P * (fnum(a["d"]) + fnum(c["d"]))
                return "DBR", comp, f"{tot:.1f}"
            comp = mat_label(b)
            return "Layer", comp, f"{fnum(b.get('d')):.1f}"

        def refresh_tree(self, keep=None):
            self.tree.delete(*self.tree.get_children())
            for i, b in enumerate(self.stack.blocks):
                t, comp, thk = self.block_summary(b)
                self.tree.insert("", "end", iid=str(i), text=b.get("name", ""), values=(t, comp, thk))
            if keep is not None and 0 <= keep < len(self.stack.blocks):
                self.tree.selection_set(str(keep))
                self.tree.see(str(keep))
            else:
                self.show_editor(None)

        def refresh_row(self, i):
            b = self.stack.blocks[i]
            t, comp, thk = self.block_summary(b)
            self.tree.item(str(i), text=b.get("name", ""), values=(t, comp, thk))

        def show_editor(self, b):
            for w in (self.ed_empty, self.ed_layer, self.ed_dbr):
                w.pack_forget()
            if b is None:
                self.sel = None
                self.ed_empty.pack(anchor="w")
                return
            self._loading = True
            v = self.ed_vars
            v["name"].set(b.get("name", ""))
            if b["type"] == "dbr":
                v["periods"].set(f"{fnum(b.get('periods')):g}")
                for k in ("a", "b"):
                    v[k + ".x"].set(f"{fnum(b[k]['x']):g}")
                    v[k + ".d"].set(f"{fnum(b[k]['d']):g}")
                v["qw"].set(f"{fnum(b.get('qw'), 980):g}")
                self.ed_dbr.pack(fill="x")
                self.update_dbr_info(b)
            else:
                for k in ("x", "d", "n670", "k670", "n980", "k980", "rate", "nkx"):
                    v[k].set(f"{fnum(b.get(k)):g}")
                self.cb_mat.set(CHOICE_OF_MAT.get(b.get("mat", "algaas"), "AlGaAs / GaAs"))
                self.cb_nkfunc.configure(values=sorted(self.lib.materials, key=str.lower))
                self.cb_nkfunc.set(b.get("nkfunc", "") if b.get("mat") == "nk" else "")
                self.layout_layer_editor(b)
                self.ed_layer.pack(fill="x")
            self._loading = False

        def layout_layer_editor(self, b):
            mat = b.get("mat", "algaas")
            for w in (self.lbl_x, self.ent_x):
                w.grid() if mat == "algaas" else w.grid_remove()
            self.custom_frame.grid() if mat == "custom" else self.custom_frame.grid_remove()
            if mat == "nk":
                self.nk_frame.grid()
                alloy = self.is_alloy(b)
                for w in (self.lbl_nkx, self.ent_nkx):
                    w.grid() if alloy else w.grid_remove()
                self.update_nk_info(b)
            else:
                self.nk_frame.grid_remove()

        def is_alloy(self, m):
            mat = self.lib.materials.get(m.get("nkfunc"))
            return mat["alloy"] if mat else bool(m.get("nkalloy"))

        def nk_summary(self, m):
            """One-line description of an nk.py material's values at both lasers."""
            key = m.get("nkfunc")
            if key not in self.lib.materials:
                if m.get("nkcached"):
                    vals = ", ".join(f"{wl} nm: n {fnum(m.get(f'n{wl}')):.3f}, k {fnum(m.get(f'k{wl}')):.4f}" for wl in LASERS)
                    return f"{key} is not loaded, so the stored values are used ({vals}). Import its nk.py file to recalculate."
                return f"{key} is not loaded. Import its nk.py file."
            parts, out_of_range = [], []
            for wl in LASERS:
                try:
                    n, k, notes = self.lib.eval(key, wl, m.get("nkx"))
                    parts.append(f"{wl} nm: n {n:.3f}, k {k:.4f}")
                    if notes:
                        out_of_range.append(str(wl))
                except NkUnavailable:
                    parts.append(f"{wl} nm: not available")
            txt = "   ".join(parts)
            if out_of_range:
                txt += f"\nOutside the file's stated wavelength range at {' and '.join(out_of_range)} nm, so treat with care."
            return txt

        def update_nk_info(self, b):
            self.nk_info.configure(text=self.nk_summary(b) if b.get("nkfunc") else "Choose a material from the imported file(s).")

        def update_dbr_info(self, b):
            per = fnum(b["a"]["d"]) + fnum(b["b"]["d"])
            txt = f"Period {per:.2f} nm, total {per * int(round(fnum(b.get('periods')))):.1f} nm."
            E = HC / max(1.0, fnum(b.get("qw"), 980))
            if any(gamma_gap(clampx(b[k]["x"])) <= E for k in ("a", "b")):
                txt += " At this design wavelength one layer is above its band gap, so quarter-wave values are approximate."
            self.dbr_info.configure(text=txt)

        # ---------------- events ----------------
        def on_select(self, _=None):
            s = self.tree.selection()
            if not s:
                return
            self.sel = int(s[0])
            self.show_editor(self.stack.blocks[self.sel])

        def on_edit(self, key):
            if self._loading or self.sel is None or self.sel >= len(self.stack.blocks):
                return
            b = self.stack.blocks[self.sel]
            val = self.ed_vars[key].get()
            if key == "name":
                b["name"] = val
            elif "." in key:
                k1, k2 = key.split(".")
                if val.strip() == "" or not math.isfinite(fnum(val, float("nan"))):
                    return
                b[k1][k2] = fnum(val)
            else:
                if val.strip() == "" or not math.isfinite(fnum(val, float("nan"))):
                    return
                b[key] = fnum(val)
            self.refresh_row(self.sel)
            if b["type"] == "dbr":
                self.update_dbr_info(b)
            elif b.get("mat") == "nk" and key == "nkx":
                self.update_nk_info(b)
            self.schedule()

        def on_mat(self, _=None):
            if self.sel is None:
                return
            b = self.stack.blocks[self.sel]
            new = MAT_OF_CHOICE.get(self.cb_mat.get(), "algaas")
            if new == "nk" and b.get("mat") != "nk":
                if not self.lib.materials:
                    messagebox.showinfo("nk.py materials", "Import an nk.py file first (Files and materials, Import nk.py…).")
                    self.cb_mat.set(CHOICE_OF_MAT.get(b.get("mat", "algaas")))
                    return
                if b.get("nkfunc") not in self.lib.materials:
                    b["nkfunc"] = sorted(self.lib.materials, key=str.lower)[0]
                b.setdefault("nkx", 0)
                b["nkalloy"] = self.lib.materials[b["nkfunc"]]["alloy"]
                b.pop("nkcached", None)
            if new == "custom" and b.get("mat") != "custom":
                for wl in LASERS:  # seed custom values from the material currently used
                    try:
                        n, k = self.stack.nk(b, wl)
                    except NkUnavailable:
                        n, k = algaas_nk(clampx(b.get("x")), wl, self.stack.model)
                    b[f"n{wl}"], b[f"k{wl}"] = round(n, 4), round(k, 5)
            b["mat"] = new
            self.show_editor(b)
            self.refresh_row(self.sel)
            self.schedule()

        def on_nkfunc(self, _=None):
            if self.sel is None:
                return
            b = self.stack.blocks[self.sel]
            b["nkfunc"] = self.cb_nkfunc.get()
            b["nkalloy"] = self.is_alloy(b)
            b.pop("nkcached", None)
            self.layout_layer_editor(b)
            self.refresh_row(self.sel)
            self.schedule()

        # ---------------- substrate ----------------
        def on_submat(self, _=None):
            if self.cb_submat.get() == "From nk.py" and not self.lib.materials:
                messagebox.showinfo("nk.py materials", "Import an nk.py file first (Files and materials, Import nk.py…).")
                self.cb_submat.set("AlGaAs / GaAs")
            if self.cb_submat.get() == "From nk.py" and self.cb_subfunc.get() not in self.lib.materials:
                self.cb_subfunc.set(sorted(self.lib.materials, key=str.lower)[0])
            self.layout_substrate()
            self.schedule()

        def layout_substrate(self):
            nk = self.cb_submat.get() == "From nk.py"
            self.cb_subfunc.configure(values=sorted(self.lib.materials, key=str.lower))
            for w in (self.lbl_subx, self.ent_subx):
                w.grid_remove() if nk else w.grid()
            for w in (self.lbl_subfunc, self.cb_subfunc):
                w.grid() if nk else w.grid_remove()
            alloy = nk and self.is_alloy({"nkfunc": self.cb_subfunc.get(), "nkalloy": self.stack.substrate.get("nkalloy")})
            for w in (self.lbl_subnkx, self.ent_subnkx):
                w.grid() if alloy else w.grid_remove()

        # ---------------- nk.py import ----------------
        def import_nk(self):
            p = filedialog.askopenfilename(title="Import nk.py material file",
                                           filetypes=[("Python material file", "*.py"), ("All files", "*.*")])
            if not p:
                return
            if not self._nk_trust_ok:
                if not messagebox.askokcancel(
                        "Import nk.py",
                        "Importing runs the Python code in this file, just like `import nk` in the UCSB script.\n\n"
                        "Only import files you trust. Continue?"):
                    return
                self._nk_trust_ok = True
            self._load_nk_files([p], report=True)

        def _load_nk_files(self, paths, report=False):
            msgs = []
            for p in paths:
                name = os.path.basename(p)
                try:
                    added, skipped = self.lib.load(p)
                except Exception as e:
                    msgs.append(f"Could not import {name}: {type(e).__name__}: {e}")
                    continue
                m = f"Imported {len(added)} materials from {name}."
                if skipped:
                    m += f" Skipped {len(skipped)} that give no value at 670 or 980 nm: " + ", ".join(s[0] for s in skipped[:8])
                    m += "…" if len(skipped) > 8 else ""
                msgs.append(m)
            self.after_lib_change()
            if report and msgs:
                messagebox.showinfo("Import nk.py", "\n\n".join(msgs))

        def after_lib_change(self):
            files = [os.path.basename(f) for f in self.lib.files]
            self.nk_status.configure(text=(f"Imported: {', '.join(files)} ({len(self.lib.materials)} materials)"
                                           if files else "No nk.py file imported."))
            self.layout_substrate()
            if self.sel is not None and self.sel < len(self.stack.blocks):
                self.show_editor(self.stack.blocks[self.sel])
            for i in range(len(self.stack.blocks)):
                self.refresh_row(i)
            self._fill_library()
            self.schedule()

        def show_library(self):
            if self._libwin is not None and self._libwin.winfo_exists():
                self._libwin.lift()
                self._fill_library()
                return
            w = self._libwin = tk.Toplevel(self)
            w.title("nk.py material library")
            w.geometry("1060x460")
            top = ttk.Frame(w, padding=8)
            top.pack(fill="x")
            self._lib_head = ttk.Label(top, text="", wraplength=760, justify="left")
            self._lib_head.pack(side="left")
            ttk.Button(top, text="Import nk.py…", command=self.import_nk).pack(side="right")
            cols = ("type", "n670", "k670", "n980", "k980", "notes")
            tv = self._libtree = ttk.Treeview(w, columns=cols, show="tree headings")
            tv.heading("#0", text="Material")
            for c, txt, wd, anc in (("type", "Type", 90, "w"), ("n670", "n 670", 125, "e"), ("k670", "k 670", 125, "e"),
                                    ("n980", "n 980", 125, "e"), ("k980", "k 980", 125, "e"), ("notes", "Notes", 330, "w")):
                tv.heading(c, text=txt)
                tv.column(c, width=wd, anchor=anc)
            tv.column("#0", width=170)
            sb = ttk.Scrollbar(w, orient="vertical", command=tv.yview)
            tv.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y", pady=(0, 8))
            tv.pack(fill="both", expand=True, padx=(8, 0), pady=(0, 8))
            ttk.Label(w, text="Alloys show x = 0 → x = 1. Values are what the simulator uses at each laser.",
                      style="Hint.TLabel", padding=(8, 0, 8, 8)).pack(anchor="w")
            self._fill_library()

        def _fill_library(self):
            if self._libwin is None or not self._libwin.winfo_exists():
                return
            tv = self._libtree
            tv.delete(*tv.get_children())
            self._lib_head.configure(text=("Files: " + "; ".join(self.lib.files)) if self.lib.files
                                     else "No nk.py file imported yet.")
            for key in sorted(self.lib.materials, key=str.lower):
                m = self.lib.materials[key]
                xs = (0.0, 1.0) if m["alloy"] else (None,)
                cells, oor = {}, set()
                for wl in LASERS:
                    ns, ks = [], []
                    for x in xs:
                        try:
                            n, k, notes = self.lib.eval(key, wl, x)
                            ns.append(f"{n:.3f}"); ks.append(f"{k:.4f}")
                            if notes:
                                oor.add(str(wl))
                        except NkUnavailable:
                            ns.append("n/a"); ks.append("n/a")
                    cells[wl] = (" → ".join(ns), " → ".join(ks))
                note = f"outside stated range at {' & '.join(sorted(oor))} nm. " if oor else ""
                note += m["doc"][:120]
                tv.insert("", "end", text=key, values=("alloy (x, wl)" if m["alloy"] else "material",
                                                       cells[670][0], cells[670][1], cells[980][0], cells[980][1], note))

        def set_qw(self):
            if self.sel is None:
                return
            b = self.stack.blocks[self.sel]
            wl = fnum(self.ed_vars["qw"].get())
            if not 600 <= wl <= 2000:
                messagebox.showinfo("Quarter-wave", "Enter a design wavelength between 600 and 2000 nm.")
                return
            b["qw"] = wl
            for k in ("a", "b"):
                b[k]["d"] = quarter_wave(clampx(b[k]["x"]), wl)
            self.show_editor(b)
            self.refresh_row(self.sel)
            self.schedule()

        def add_layer(self):
            self.stack.blocks.append(new_layer("New layer", 0, 100))
            self.refresh_tree(len(self.stack.blocks) - 1); self.schedule()

        def add_dbr(self):
            self.stack.blocks.append(new_dbr("DBR", 10, 0, 0.9, 980))
            self.refresh_tree(len(self.stack.blocks) - 1); self.schedule()

        def dup(self):
            if self.sel is None:
                return
            self.stack.blocks.insert(self.sel + 1, copy.deepcopy(self.stack.blocks[self.sel]))
            self.refresh_tree(self.sel + 1); self.schedule()

        def remove(self):
            if self.sel is None:
                return
            del self.stack.blocks[self.sel]
            nxt = min(self.sel, len(self.stack.blocks) - 1)
            self.sel = None
            self.refresh_tree(nxt if nxt >= 0 else None); self.schedule()

        def move(self, dirn):
            i = self.sel
            if i is None or not 0 <= i + dirn < len(self.stack.blocks):
                return
            B = self.stack.blocks
            B[i], B[i + dirn] = B[i + dirn], B[i]
            self.refresh_tree(i + dirn); self.schedule()

        def load_preset(self):
            p = PRESETS.get(self.cb_preset.get())
            if not p:
                return
            self.stack.__dict__.update(p())
            self._sync_globals_to_vars()
            self.v_target.set("")
            self.sel = None
            self._full_x = None  # new structure: show the full trace
            self.refresh_tree(); self.schedule()

        # ---------------- compute + plot ----------------
        def schedule(self):
            if self._job:
                self.after_cancel(self._job)
            self._job = self.after(120, self.run)

        def run(self):
            self._job = None
            self._read_globals()
            refresh_nk_cache(self.stack)
            flat = self.stack.flatten()
            total = sum(m["d"] for m in flat)
            over = max(0.0, fnum(self.v_over.get(), 300))
            step = max(0.1, fnum(self.v_step.get(), 1))
            note = ""
            if (total + over) / step > 60000:
                step = math.ceil((total + over) / 60000 * 10) / 10
                note = f"Step raised to {step} nm to keep the calculation fast."
            self.step_note.configure(text=note)
            if not flat and over == 0:
                self.last = None
                self.ax_n.clear(); self.ax_r.clear()
                self.ax_r.text(0.5, 0.5, "Add a layer to see the monitor trace.", ha="center", va="center", transform=self.ax_r.transAxes)
                self.canvas.draw_idle()
                return
            flat, sub, sims = run_all(self.stack, step, over)
            bad = [f"{wl} nm trace unavailable: {sims[wl]['invalid']}" for wl in LASERS if sims[wl]["invalid"]]
            if bad:
                self.step_note.configure(text="\n".join(([note] if note else []) + bad))
            bounds, z, t = [], 0.0, 0.0
            for m in flat:
                dt = m["d"] / self.stack.rate(m) * 60
                bounds.append((z, z + m["d"], t, t + dt, mat_color(m), m["label"]))
                z += m["d"]; t += dt
            bounds.append((z, z + over, t, t + over / self.stack.rate(sub) * 60, mat_color(sub), "Substrate, " + mat_label(sub)))
            self.last = {"flat": flat, "sub": sub, "sims": sims, "bounds": bounds, "total": total}
            self.title(f"Laser endpoint simulator: {len(flat)} layers, {total / 1000:.2f} µm, {fmt_time(t)} to substrate")
            self.draw()
            self.update_readout()
            self.update_nk_table()

        def lasers_on(self):
            v = self.v_laser.get()
            chosen = list(LASERS) if v == "both" else [int(v)]
            if self.last:  # hide traces whose materials can't be evaluated at that wavelength
                chosen = [wl for wl in chosen if not self.last["sims"][wl].get("invalid")]
            return chosen

        def target_index(self):
            if not self.last:
                return None
            tv = fnum(self.v_target.get(), float("nan"))
            if not math.isfinite(tv):
                return None
            dep = self.last["sims"][670]["depth"]
            return int(np.argmin(np.abs(dep - max(0.0, tv))))

        def refit_y(self):
            """Scale the reflectance axis to the visible laser trace(s) inside the current x window."""
            if not self.last:
                return
            use_t = self.v_axis.get() == "time"
            x0, x1 = sorted(self.ax_r.get_xlim())
            mn, mx = np.inf, -np.inf
            for wl in self.lasers_on():
                s = self.last["sims"][wl]
                X = s["time"] if use_t else s["depth"]
                m = (X >= x0) & (X <= x1)
                if m.any():
                    R = s["R"][m] * 100
                    mn, mx = min(mn, R.min()), max(mx, R.max())
            if not np.isfinite(mn):
                return
            if self.v_yfit.get() == "zero":
                lo, hi = 0.0, mx + max(0.5, mx * 0.05)
            else:
                pad = max(0.25, (mx - mn) * 0.06)
                lo, hi = max(0.0, mn - pad), mx + pad
            self.ax_r.set_ylim(lo, hi)
            self.canvas.draw_idle()

        def draw(self):
            L = self.last
            use_t = self.v_axis.get() == "time"
            # keep the user's x zoom across redraws (e.g. when switching lasers), unless the axis units changed
            keep = None
            if self._drawn_axis == self.v_axis.get() and self._full_x is not None:
                cur = self.ax_r.get_xlim()
                if any(abs(a - b) > 1e-9 for a, b in zip(cur, self._full_x)):
                    keep = cur
            for ax in (self.ax_n, self.ax_r):
                ax.clear()
                for i, (z0, z1, t0, t1, col, _) in enumerate(L["bounds"]):
                    last = i == len(L["bounds"]) - 1
                    ax.axvspan(t0 if use_t else z0, t1 if use_t else z1, color=col, alpha=0.2 if last else 0.38, lw=0, zorder=0)
                ax.grid(True, alpha=0.35, zorder=1)
            for wl in self.lasers_on():
                s = L["sims"][wl]
                X = s["time"] if use_t else s["depth"]
                self.ax_n.step(X, s["n"], where="post", color=LASER_COLOR[wl], lw=1.4, zorder=3)
                self.ax_r.plot(X, s["R"] * 100, color=LASER_COLOR[wl], lw=1.6, label=f"{wl} nm", zorder=3)
            ti = self.target_index()
            if ti is not None:
                s = L["sims"][670]
                xv = s["time"][ti] if use_t else s["depth"][ti]
                for ax in (self.ax_n, self.ax_r):
                    ax.axvline(xv, color="#16202A", ls="--", lw=1.2, zorder=4)
            self.ax_n.set_ylabel("n at surface")
            self.ax_r.set_ylabel("Reflectance (%)")
            self.ax_r.set_xlabel("Etch time (s)" if use_t else "Etch depth (nm)")
            self.ax_r.set_xlim(left=0)
            self._full_x = self.ax_r.get_xlim()
            self._drawn_axis = self.v_axis.get()
            self.ax_n.tick_params(labelbottom=False)
            handles, labels = self.ax_r.get_legend_handles_labels()
            if not handles:
                self.ax_r.text(0.5, 0.5, "The selected laser has no valid trace; see the note under Substrate and etch.",
                               ha="center", va="center", transform=self.ax_r.transAxes, wrap=True)
            else:
                self.ax_n.legend(handles, labels, loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2,
                                 frameon=False, borderaxespad=0)
            self.toolbar.update()        # reset zoom history to the new data
            self.toolbar.push_current()  # "Home" = full trace
            if keep is not None:
                self.ax_r.set_xlim(keep)
            # ax.clear() drops callbacks, so reconnect: refit y whenever x is zoomed or panned
            self.ax_r.callbacks.connect("xlim_changed", lambda ax: self.after_idle(self.refit_y))
            self.refit_y()

        def label_at(self, i):
            j = self.last["sims"][670]["idx"][i]
            return self.last["bounds"][-1][5] if j < 0 else self.last["flat"][j]["label"]

        def update_readout(self):
            for a, b, c in self.ro_labels:
                a.configure(text=""); b.configure(text=""); c.configure(text="")
            L, ti = self.last, self.target_index()
            if ti is None:
                for col, wl in enumerate(self.lasers_on()):
                    R = L["sims"][wl]["R"] * 100
                    a, b, c = self.ro_labels[col]
                    a.configure(text=f"{wl} nm signal range", foreground=LASER_COLOR[wl])
                    b.configure(text=f"{R.min():.1f} – {R.max():.1f} %")
                    c.configure(text=f"Start {R[0]:.1f} %, end {R[-1]:.1f} %")
                self.ro_layer.configure(text="Click the plot, or type a stop depth above, to read the signal at a stop point.")
                return
            s0 = L["sims"][670]
            a, b, c = self.ro_labels[0]
            a.configure(text="Stop depth", foreground="")
            b.configure(text=f"{s0['depth'][ti]:.1f} nm")
            c.configure(text=f"{fmt_time(s0['time'][ti])} of etching")
            for col, wl in enumerate(self.lasers_on(), start=1):
                s = L["sims"][wl]
                i0, i1 = max(0, ti - 1), min(len(s["R"]) - 1, ti + 1)
                sl = (s["R"][i1] - s["R"][i0]) * 100
                trend = "flat" if abs(sl) < 1e-3 else ("rising" if sl > 0 else "falling")
                a, b, c = self.ro_labels[col]
                a.configure(text=f"{wl} nm at stop", foreground=LASER_COLOR[wl])
                b.configure(text=f"{s['R'][ti] * 100:.2f} %")
                c.configure(text=f"Signal {trend} as the etch continues")
            j = s0["idx"][ti]
            z = s0["depth"][ti]
            if j >= 0:
                z0, z1 = L["bounds"][j][0], L["bounds"][j][1]
                within = f"{z - z0:.1f} nm into this layer, {z1 - z:.1f} nm left"
            else:
                within = f"{z - L['total']:.1f} nm into substrate"
            self.ro_layer.configure(text=f"Layer at stop: {self.label_at(ti)}   ({within})")

        def update_nk_table(self):
            self.nk_tree.delete(*self.nk_tree.get_children())
            seen = {}
            for m in self.last["flat"] + [dict(self.last["sub"], label="Substrate")]:
                if m.get("mat") == "custom":
                    key = ("c", m.get("n670"), m.get("k670"), m.get("n980"), m.get("k980"))
                    name = m["label"].split(",")[0] + " (custom)"
                elif m.get("mat") == "nk":
                    key = ("n", m.get("nkfunc"), clampx(m.get("nkx")) if m.get("nkalloy") else None)
                    name = mat_label(m) + " (nk.py)"
                else:
                    key, name = ("a", clampx(m.get("x"))), mat_label(m)
                if key not in seen:
                    seen[key] = (m, name)
            for m, name in seen.values():
                vals = []
                for wl in LASERS:
                    try:
                        n, k = self.stack.nk(m, wl)
                    except NkUnavailable:
                        vals += ["n/a", "n/a", "not available"]
                        continue
                    fr = f"absorbing, 1/α {wl / (4 * math.pi * k):.0f} nm" if k > 1e-6 else f"{wl / (2 * n):.1f} nm"
                    vals += [f"{n:.3f}", f"{k:.4f}", fr]
                vals.append(f"{self.stack.rate(m):.1f}")
                self.nk_tree.insert("", "end", text=name, values=vals)

        def x_to_index(self, xv):
            s = self.last["sims"][670]
            arr = s["time"] if self.v_axis.get() == "time" else s["depth"]
            return int(np.argmin(np.abs(arr - xv)))

        def on_hover(self, ev):
            if not self.last or ev.inaxes not in (self.ax_n, self.ax_r) or ev.xdata is None:
                return
            i = self.x_to_index(ev.xdata)
            s = self.last["sims"]
            parts = [f"Depth {s[670]['depth'][i]:.1f} nm", f"t {fmt_time(s[670]['time'][i])}"]
            parts += [f"R{wl} {s[wl]['R'][i] * 100:.2f} %  (n {s[wl]['n'][i]:.3f})" for wl in self.lasers_on()]
            parts.append(self.label_at(i))
            self.v_status.set("   |   ".join(parts))

        def on_click(self, ev):
            if not self.last or ev.inaxes not in (self.ax_n, self.ax_r) or ev.xdata is None or ev.button != 1:
                return
            mode = getattr(self.toolbar, "mode", "")
            if getattr(mode, "value", mode):
                return  # zoom or pan is active
            i = self.x_to_index(ev.xdata)
            self.v_target.set(f"{self.last['sims'][670]['depth'][i]:.1f}")

        # ---------------- files ----------------
        def to_json(self):
            self._read_globals()
            s = self.stack
            return {"app": "laser-endpoint-sim", "version": 1, "laser": self.v_laser.get(), "model": s.model,
                    "axis": self.v_axis.get(), "step": fnum(self.v_step.get(), 1), "over": fnum(self.v_over.get(), 300),
                    "rateG": s.rateG, "rateA": s.rateA,
                    "target": fnum(self.v_target.get(), None) if self.v_target.get().strip() else None,
                    "substrate": s.substrate, "blocks": s.blocks, "nk_files": list(self.lib.files)}

        def from_json(self, d):
            if not isinstance(d.get("blocks"), list) or "substrate" not in d:
                raise ValueError("not a stack file")
            s = self.stack
            s.blocks, s.substrate = d["blocks"], d["substrate"]
            s.model = d.get("model", s.model)
            s.rateG, s.rateA = d.get("rateG", s.rateG), d.get("rateA", s.rateA)
            for var, key in ((self.v_laser, "laser"), (self.v_axis, "axis"), (self.v_step, "step"), (self.v_over, "over")):
                if d.get(key) is not None:
                    var.set(f"{d[key]:g}" if isinstance(d[key], (int, float)) else str(d[key]))
            self.v_target.set("" if d.get("target") is None else f"{d['target']:g}")
            wanted = [os.path.abspath(f) for f in d.get("nk_files", []) if isinstance(f, str)]
            return [f for f in wanted if f not in self.lib.files]  # nk.py files this stack uses that aren't loaded

        def save_json(self):
            p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Stack file", "*.json")], initialfile="etch-stack.json")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(self.to_json(), f, indent=2)

        def load_json(self):
            p = filedialog.askopenfilename(filetypes=[("Stack file", "*.json"), ("All files", "*.*")])
            if not p:
                return
            try:
                with open(p, encoding="utf-8") as f:
                    missing = self.from_json(json.load(f))
            except Exception:
                messagebox.showerror("Load stack", "That file is not a saved stack from this tool.")
                return
            self._sync_globals_to_vars()
            self.sel = None
            self.refresh_tree(); self.schedule()
            if missing:
                found = [f for f in missing if os.path.isfile(f)]
                lost = [f for f in missing if not os.path.isfile(f)]
                if found and messagebox.askyesno(
                        "nk.py materials",
                        "This stack uses materials from:\n\n" + "\n".join(found) +
                        "\n\nImport these files now? This runs their Python code, so only do it if you trust them."):
                    self._nk_trust_ok = True
                    self._load_nk_files(found, report=False)
                if lost:
                    messagebox.showinfo("nk.py materials",
                                        "These nk.py files weren't found on this computer, so the n, k values stored in the "
                                        "stack are used instead:\n\n" + "\n".join(lost))

        def export_csv(self):
            if not self.last:
                return
            p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="laser-endpoint-trace.csv")
            if not p:
                return
            a, b = self.last["sims"][670], self.last["sims"][980]
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["depth_nm", "time_s", "layer", "n_670", "k_670", "R_670_pct", "n_980", "k_980", "R_980_pct"])
                for i in range(len(a["depth"])):
                    row = [f"{a['depth'][i]:.3f}", f"{a['time'][i]:.3f}", self.label_at(i)]
                    for s in (a, b):
                        row += (["", "", ""] if s.get("invalid") else
                                [f"{s['n'][i]:.4f}", f"{s['k'][i]:.5f}", f"{s['R'][i] * 100:.4f}"])
                    w.writerow(row)

        def save_png(self):
            p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")], initialfile="laser-endpoint-plot.png")
            if p:
                self.fig.savefig(p, dpi=200)

        def _autoload(self):
            try:
                with open(AUTOSAVE, encoding="utf-8") as f:
                    missing = self.from_json(json.load(f))
            except Exception:
                return
            # re-import the nk.py files you had loaded last session (you imported them yourself)
            found = [f for f in missing if os.path.isfile(f)]
            if found:
                self._nk_trust_ok = True
                for f in found:
                    try:
                        self.lib.load(f)
                    except Exception:
                        pass

        def _on_close(self):
            try:
                with open(AUTOSAVE, "w", encoding="utf-8") as f:
                    json.dump(self.to_json(), f)
            except Exception:
                pass
            self.destroy()

    App().mainloop()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest"]
        selftest(rest[0] if rest else None)
    else:
        main_gui()
