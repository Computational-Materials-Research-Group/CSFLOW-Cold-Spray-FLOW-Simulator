"""
CSFLOW | Cold Spray FLOW simulator - shared physics library
============================================================
Gas and powder property databases, nozzle geometry, quasi-1D gas dynamics,
compressible particle drag (Henderson 1976), particle heat transfer
(Ranz-Marshall with recovery temperature) and critical/erosion velocity
(Schmidt et al. 2006, Assadi et al. 2003).

Units: SI throughout (m, s, K, Pa, kg) unless a name says otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq

T_REF = 293.15  # reference temperature for critical velocity (K)

# --------------------------------------------------------------------------
# Property databases
# --------------------------------------------------------------------------
GASES = {
    # gamma, R (J/kg/K), Sutherland mu0 (Pa s) at Tmu0 (K), S (K), Pr, molar mass
    "Nitrogen (N2)": dict(gamma=1.40, R=296.8, mu0=1.663e-5, Tmu0=273.15,
                          S=106.7, Pr=0.71, W=28.013),
    "Helium (He)": dict(gamma=1.667, R=2077.1, mu0=1.870e-5, Tmu0=273.15,
                        S=79.4, Pr=0.67, W=4.003),
    "Air": dict(gamma=1.40, R=287.05, mu0=1.716e-5, Tmu0=273.15,
                S=110.4, Pr=0.71, W=28.966),
}

MATERIALS = {
    # density (kg/m3), cp (J/kg/K), melting point (K), UTS (Pa)
    "Copper (Cu)": dict(rho=8960, cp=385, Tm=1358, uts=220e6),
    "Aluminium (Al)": dict(rho=2700, cp=900, Tm=933, uts=90e6),
    "Al 6061": dict(rho=2700, cp=896, Tm=855, uts=310e6),
    "Titanium (CP-Ti)": dict(rho=4510, cp=520, Tm=1941, uts=345e6),
    "Ti-6Al-4V": dict(rho=4430, cp=526, Tm=1878, uts=950e6),
    "Nickel (Ni)": dict(rho=8900, cp=444, Tm=1728, uts=317e6),
    "Stainless 316L": dict(rho=7990, cp=500, Tm=1673, uts=485e6),
    "Zinc (Zn)": dict(rho=7140, cp=388, Tm=693, uts=110e6),
}


@dataclass
class Gas:
    name: str
    gamma: float
    R: float
    mu0: float
    Tmu0: float
    S: float
    Pr: float
    W: float

    @classmethod
    def from_db(cls, name, **overrides):
        d = dict(GASES[name]) if name in GASES else {}
        d.update(overrides)
        return cls(name=name, **d)

    @property
    def cp(self):
        return self.gamma * self.R / (self.gamma - 1.0)

    def mu(self, T):
        T = np.maximum(T, 1.0)
        return self.mu0 * (T / self.Tmu0) ** 1.5 * (self.Tmu0 + self.S) / (T + self.S)

    def k(self, T):
        return self.mu(T) * self.cp / self.Pr

    # Sutherland coefficients in OpenFOAM form mu = As*sqrt(T)/(1+Ts/T)
    @property
    def As(self):
        return self.mu0 * (self.Tmu0 + self.S) / self.Tmu0 ** 1.5


@dataclass
class Powder:
    name: str
    rho: float
    cp: float
    Tm: float
    uts: float

    @classmethod
    def from_db(cls, name, **overrides):
        d = dict(MATERIALS[name]) if name in MATERIALS else {}
        d.update(overrides)
        return cls(name=name, **d)


# --------------------------------------------------------------------------
# Nozzle geometry (conical converging + conical diverging, de Laval)
# --------------------------------------------------------------------------
@dataclass
class Nozzle:
    d_in: float = 8.0e-3      # inlet (chamber) diameter
    d_throat: float = 2.0e-3  # throat diameter
    d_exit: float = 6.0e-3    # exit diameter
    L_conv: float = 20e-3     # converging length
    L_div: float = 130e-3     # diverging length
    standoff: float = 30e-3   # nozzle exit to substrate distance

    @property
    def area_ratio(self):
        return (self.d_exit / self.d_throat) ** 2

    def radius(self, x):
        """Wall radius at axial position x (throat at x = 0)."""
        x = np.asarray(x, dtype=float)
        ri, rt, re = self.d_in / 2, self.d_throat / 2, self.d_exit / 2
        r = np.where(
            x < 0,
            rt + (ri - rt) * np.clip(-x / self.L_conv, 0, 1),
            rt + (re - rt) * np.clip(x / self.L_div, 0, 1),
        )
        return r

    @property
    def x_in(self):
        return -self.L_conv

    @property
    def x_exit(self):
        return self.L_div

    @property
    def x_sub(self):
        return self.L_div + self.standoff


# --------------------------------------------------------------------------
# Isentropic and normal-shock relations
# --------------------------------------------------------------------------
def T_ratio(M, g):          # T/T0
    return 1.0 / (1.0 + 0.5 * (g - 1) * M * M)


def p_ratio(M, g):          # p/p0
    return T_ratio(M, g) ** (g / (g - 1))


def area_ratio(M, g):       # A/A*
    return (1.0 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M * M)) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(Ar, g, supersonic):
    if Ar <= 1.0 + 1e-12:
        return 1.0
    if supersonic:
        return brentq(lambda M: area_ratio(M, g) - Ar, 1.0 + 1e-9, 50.0)
    return brentq(lambda M: area_ratio(M, g) - Ar, 1e-6, 1.0 - 1e-9)


def normal_shock(M1, g):
    """Return M2, p2/p1, T2/T1, p02/p01 across a normal shock."""
    M2 = math.sqrt((1 + 0.5 * (g - 1) * M1 ** 2) / (g * M1 ** 2 - 0.5 * (g - 1)))
    pr = 1 + 2 * g / (g + 1) * (M1 ** 2 - 1)
    Tr = pr * (2 + (g - 1) * M1 ** 2) / ((g + 1) * M1 ** 2)
    p0r = ((g + 1) * M1 ** 2 / ((g - 1) * M1 ** 2 + 2)) ** (g / (g - 1)) * \
          ((g + 1) / (2 * g * M1 ** 2 - (g - 1))) ** (1 / (g - 1))
    return M2, pr, Tr, p0r


# --------------------------------------------------------------------------
# Particle drag and heat transfer
# --------------------------------------------------------------------------
def henderson_cd(Re, M, gamma, Tp_over_T):
    """Henderson (1976) drag coefficient for a sphere, valid over continuum,
    slip and free-molecular regimes and from incompressible to hypersonic."""
    Re = max(Re, 1e-8)
    M = max(M, 1e-8)
    Tw = max(Tp_over_T, 1e-3)

    def cd_sub(M):
        S = M * math.sqrt(gamma / 2.0)
        sre = math.sqrt(Re)
        t1 = 24.0 / (Re + S * (4.33 + (3.65 - 1.53 * Tw) / (1 + 0.353 * Tw)
                                * math.exp(-0.247 * Re / S)))
        t2 = math.exp(-0.5 * M / sre) * (
            (4.5 + 0.38 * (0.03 * Re + 0.48 * sre)) / (1 + 0.03 * Re + 0.48 * sre)
            + 0.1 * M ** 2 + 0.2 * M ** 8)
        t3 = 0.6 * S * (1 - math.exp(-M / Re))
        return t1 + t2 + t3

    def cd_sup(M):
        S = M * math.sqrt(gamma / 2.0)
        a = 1.86 * math.sqrt(M / Re)
        num = 0.9 + 0.34 / M ** 2 + a * (2 + 2 / S ** 2 + 1.058 / S * math.sqrt(Tw) - 1 / S ** 4)
        return num / (1 + a)

    if M <= 1.0:
        return cd_sub(M)
    if M >= 1.75:
        return cd_sup(M)
    c1, c2 = cd_sub(1.0), cd_sup(1.75)
    return c1 + (4.0 / 3.0) * (M - 1.0) * (c2 - c1)


def stokes_cd(Re, *_):
    """Standard (incompressible) sphere drag, Schiller-Naumann + Newton."""
    Re = max(Re, 1e-8)
    return 24 / Re * (1 + 0.15 * Re ** 0.687) if Re < 1000 else 0.44


DRAG_MODELS = {"Henderson (compressible)": henderson_cd,
               "Schiller-Naumann (incompressible)": stokes_cd}


def particle_rhs(urel_vec, Tg, rho_g, gas: Gas, dp, Tp, powder: Powder, drag_fn):
    """Acceleration vector and dTp/dt for one particle.

    urel_vec : gas velocity minus particle velocity (array, any dimension)
    """
    urel = float(np.linalg.norm(urel_vec))
    mu = float(gas.mu(Tg))
    a_sound = math.sqrt(gas.gamma * gas.R * Tg)
    Re = rho_g * urel * dp / mu
    Mrel = urel / a_sound
    Cd = drag_fn(Re, Mrel, gas.gamma, Tp / Tg)
    mp = powder.rho * math.pi * dp ** 3 / 6
    Ap = math.pi * dp ** 2 / 4
    acc = 0.5 * rho_g * Cd * Ap * urel * np.asarray(urel_vec) / mp
    # heat transfer: Ranz-Marshall against adiabatic recovery temperature
    Nu = 2.0 + 0.6 * math.sqrt(Re) * gas.Pr ** (1 / 3)
    h = Nu * float(gas.k(Tg)) / dp
    Taw = Tg + gas.Pr ** (1 / 3) * urel ** 2 / (2 * gas.cp)
    dT = h * math.pi * dp ** 2 * (Taw - Tp) / (mp * powder.cp)
    return acc, dT, Re, Mrel, Cd


# --------------------------------------------------------------------------
# Critical and erosion velocity
# --------------------------------------------------------------------------
def v_crit_schmidt(powder: Powder, Ti, F1=1.2, F2=0.3):
    """Schmidt et al. (2006) critical velocity (m/s) at impact temperature Ti."""
    Ti = min(Ti, powder.Tm - 1)
    term1 = F1 * 4 * powder.uts * (1 - (Ti - T_REF) / (powder.Tm - T_REF)) / powder.rho
    term2 = F2 * powder.cp * (powder.Tm - Ti)
    return math.sqrt(max(term1 + term2, 0.0))


def v_crit_assadi(powder: Powder, Ti):
    """Assadi et al. (2003) empirical fit (rho g/cm3, Tm and Ti in C, UTS MPa)."""
    return (667 - 14 * powder.rho / 1000 + 0.08 * (powder.Tm - 273.15)
            + 0.1 * powder.uts / 1e6 - 0.4 * (Ti - 273.15))


def v_erosion(powder: Powder, Ti):
    """Erosion velocity, taken as about twice the critical velocity (Schmidt)."""
    return 2.0 * v_crit_schmidt(powder, Ti)


CRIT_MODELS = {"Schmidt et al. (2006)": v_crit_schmidt,
               "Assadi et al. (2003)": v_crit_assadi}


# --------------------------------------------------------------------------
# Particle size distribution (Rosin-Rammler, mass based)
# --------------------------------------------------------------------------
def rosin_rammler_bins(d50, n, dmin, dmax, nbins):
    """Return diameter bin centres and mass fractions for a Rosin-Rammler
    distribution with median d50 and spread n (mass cumulative)."""
    dbar = d50 / (math.log(2) ** (1 / n))
    edges = np.linspace(dmin, dmax, nbins + 1)
    F = 1 - np.exp(-(edges / dbar) ** n)
    w = np.diff(F)
    w = w / w.sum()
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, w


def to_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
