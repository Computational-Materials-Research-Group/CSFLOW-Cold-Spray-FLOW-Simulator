"""
CSFLOW quick mode: quasi-1D nozzle gas dynamics + free jet + bow-shock layer,
with Lagrangian particle acceleration and heating along the centreline.

Runs in seconds, so it is meant for parametric sweeps. The full OpenFOAM mode
(csflow_foam.py) resolves the 2D axisymmetric flow, shocks and boundary layers.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from csflow_physics import (Gas, Powder, Nozzle, T_ratio, p_ratio, area_ratio,
                            mach_from_area, normal_shock, particle_rhs,
                            DRAG_MODELS, CRIT_MODELS, v_erosion,
                            rosin_rammler_bins)


# --------------------------------------------------------------------------
# Gas phase
# --------------------------------------------------------------------------
def nozzle_flow(noz: Nozzle, gas: Gas, p0, T0, p_amb, nx=600):
    """Quasi-1D flow in the nozzle. Detects the operating regime
    (subsonic, normal shock inside, over-, ideally or under-expanded)."""
    g = gas.gamma
    At = math.pi * noz.d_throat ** 2 / 4
    Ae = math.pi * noz.d_exit ** 2 / 4
    x = np.concatenate([np.linspace(noz.x_in, 0, nx // 4, endpoint=False),
                        np.linspace(0, noz.x_exit, nx - nx // 4)])
    A = math.pi * noz.radius(x) ** 2
    Ar = Ae / At
    Me_sub = mach_from_area(Ar, g, False)
    Me_sup = mach_from_area(Ar, g, True)
    pe_sub = p0 * p_ratio(Me_sub, g)
    pe_sup = p0 * p_ratio(Me_sup, g)
    pe_shock_at_exit = pe_sup * normal_shock(Me_sup, g)[1]

    M = np.empty_like(x)
    p0_local = np.full_like(x, p0)
    x_shock = None

    if p_amb >= p0:
        raise ValueError("Stagnation pressure must exceed ambient pressure.")

    if p_amb > pe_sub:
        # not choked: subsonic everywhere, exit pressure = ambient
        regime = "Subsonic (not choked)"
        Me = math.sqrt(2 / (g - 1) * ((p0 / p_amb) ** ((g - 1) / g) - 1))
        Astar = Ae / area_ratio(Me, g)
        for i, a in enumerate(A):
            M[i] = mach_from_area(a / Astar, g, False)
        mdot = None
    else:
        choked = True
        if p_amb > pe_shock_at_exit:
            # normal shock inside the diverging section
            def resid(As):
                M1 = mach_from_area(As / At, g, True)
                _, _, _, p0r = normal_shock(M1, g)
                Astar2 = At / p0r
                Me2 = mach_from_area(Ae / Astar2, g, False)
                return p0 * p0r * p_ratio(Me2, g) - p_amb
            As = brentq(resid, At * (1 + 1e-9), Ae * (1 - 1e-9))
            M1 = mach_from_area(As / At, g, True)
            _, _, _, p0r = normal_shock(M1, g)
            Astar2 = At / p0r
            x_shock = float(np.interp(As, A[x >= 0], x[x >= 0]))
            regime = "Normal shock inside nozzle"
            for i, (xi, a) in enumerate(zip(x, A)):
                if xi < 0:
                    M[i] = mach_from_area(a / At, g, False)
                elif xi < x_shock:
                    M[i] = mach_from_area(a / At, g, True)
                else:
                    M[i] = mach_from_area(max(a / Astar2, 1.0), g, False)
                    p0_local[i] = p0 * p0r
        else:
            for i, (xi, a) in enumerate(zip(x, A)):
                M[i] = mach_from_area(a / At, g, xi >= 0)
            ratio = pe_sup / p_amb
            if abs(ratio - 1) < 0.03:
                regime = "Ideally expanded"
            elif ratio < 1:
                regime = "Over-expanded (oblique shocks at exit)"
            else:
                regime = "Under-expanded (expansion fans at exit)"
        mdot = p0 * At * math.sqrt(g / (gas.R * T0)) * \
            ((g + 1) / 2) ** (-(g + 1) / (2 * (g - 1)))

    T = T0 * T_ratio(M, g)
    p = p0_local * p_ratio(M, g)
    rho = p / (gas.R * T)
    u = M * np.sqrt(g * gas.R * T)
    if mdot is None:
        mdot = float(rho[-1] * u[-1] * Ae)
    info = dict(regime=regime, x_shock=x_shock, Me_design=Me_sup,
                pe_design=pe_sup, mdot=mdot, Me=float(M[-1]),
                pe=float(p[-1]), ue=float(u[-1]), Te=float(T[-1]))
    return dict(x=x, M=M, T=T, p=p, rho=rho, u=u, p0=p0_local), info


def jet_flow(noz: Nozzle, gas: Gas, T0, p_amb, exit_state, info,
             bow_shock_k=0.25, nx=300):
    """Centreline free jet from nozzle exit to the substrate, with a
    normal-shock compressed layer (bow shock) in front of the substrate."""
    g = gas.gamma
    de = noz.d_exit
    xe, xs = noz.x_exit, noz.x_sub
    x = np.linspace(xe, xs, nx)
    Me, pe = info["Me"], info["pe"]
    p0e = exit_state["p0"][-1]

    # fully expanded jet Mach number (pressure matched to ambient)
    if Me > 1 and p0e > p_amb:
        Mj = math.sqrt(max(2 / (g - 1) * ((p0e / p_amb) ** ((g - 1) / g) - 1), 1e-6))
    else:
        Mj = Me
    # supersonic core length, empirical (Lau et al.-type fit)
    L_core = de * (4.2 + 1.1 * Mj ** 2)

    s = x - xe
    blend = np.clip(s / de, 0, 1)
    Mc = Me + (Mj - Me) * blend          # core Mach number
    Tc = T0 * T_ratio(Mc, g)
    uc = Mc * np.sqrt(g * gas.R * Tc)
    decay = np.where(s > L_core, L_core / np.maximum(s, 1e-12), 1.0)
    u = uc * decay
    T = T0 - u ** 2 / (2 * gas.cp)       # stagnation temperature conserved
    M = u / np.sqrt(g * gas.R * T)
    p = np.where(s < de, pe + (p_amb - pe) * blend, p_amb)
    rho = p / (gas.R * T)

    # bow shock / stagnation layer
    delta = bow_shock_k * de
    x_bs = xs - delta
    i_bs = int(np.searchsorted(x, x_bs))
    M1 = float(np.interp(x_bs, x, M))
    p1 = float(np.interp(x_bs, x, p))
    T1 = float(np.interp(x_bs, x, T))
    if M1 > 1:
        M2, pr, Tr, _ = normal_shock(M1, g)
        p2, T2 = p1 * pr, T1 * Tr
    else:
        M2, p2, T2 = M1, p1, T1
    p02 = p2 / p_ratio(M2, g)
    u2 = M2 * math.sqrt(g * gas.R * T2)
    xl = x[i_bs:]
    frac = np.clip((xs - xl) / delta, 0, 1)
    u[i_bs:] = u2 * frac                 # linear stagnation-point deceleration
    T[i_bs:] = T0 - u[i_bs:] ** 2 / (2 * gas.cp)
    M[i_bs:] = u[i_bs:] / np.sqrt(g * gas.R * T[i_bs:])
    p[i_bs:] = p02 * p_ratio(M[i_bs:], g)
    rho = p / (gas.R * T)
    jinfo = dict(Mj=Mj, L_core=L_core, x_bow_shock=x_bs, M_before_bs=M1,
                 bow_shock=M1 > 1, p_stag_substrate=p02)
    return dict(x=x, M=M, T=T, p=p, rho=rho, u=u), jinfo


def centreline(cfg):
    noz = Nozzle(**cfg["nozzle"])
    gas = Gas.from_db(cfg["gas"]["name"], **cfg["gas"].get("overrides", {}))
    p0, T0, pa = cfg["bc"]["p0"], cfg["bc"]["T0"], cfg["bc"]["p_amb"]
    nf, info = nozzle_flow(noz, gas, p0, T0, pa)
    jf, jinfo = jet_flow(noz, gas, T0, pa, nf, info,
                         cfg["model"].get("bow_shock_k", 0.25))
    keys = ["x", "M", "T", "p", "rho", "u"]
    field = {k: np.concatenate([nf[k], jf[k][1:]]) for k in keys}
    info.update(jinfo)
    return noz, gas, field, info


# --------------------------------------------------------------------------
# Particle phase (1D along the axis)
# --------------------------------------------------------------------------
def track_particle(dp, field, noz, gas, powder, drag_fn, x0, v0, Tp0,
                   n_out=200):
    xg, ug, Tg, rg = field["x"], field["u"], field["T"], field["rho"]
    x_end = noz.x_sub

    def rhs(t, y):
        x, v, Tp = y
        u = np.interp(x, xg, ug)
        T = np.interp(x, xg, Tg)
        rho = np.interp(x, xg, rg)
        acc, dT, *_ = particle_rhs(np.array([u - v]), T, rho, gas, dp, Tp,
                                   powder, drag_fn)
        return [v, acc[0], dT]

    def hit(t, y):
        return y[0] - x_end
    hit.terminal, hit.direction = True, 1

    def stall(t, y):
        return y[1] - 0.5
    stall.terminal, stall.direction = True, -1

    L = x_end - x0
    t_max = 50 * L / max(v0, 1.0) + 0.05
    sol = solve_ivp(rhs, (0, t_max), [x0, v0, Tp0], method="RK45",
                    events=[hit, stall], max_step=L / 400, rtol=1e-6, atol=1e-9)
    impacted = len(sol.t_events[0]) > 0
    idx = np.linspace(0, len(sol.t) - 1, min(n_out, len(sol.t))).astype(int)
    # drag/Re/Mrel histories for reporting
    Re_h, M_h, Cd_h = [], [], []
    for x, v, Tp in sol.y.T[idx]:
        u = np.interp(x, xg, ug)
        T = np.interp(x, xg, Tg)
        rho = np.interp(x, xg, rg)
        _, _, Re, Mr, Cd = particle_rhs(np.array([u - v]), T, rho, gas, dp,
                                        Tp, powder, drag_fn)
        Re_h.append(Re); M_h.append(Mr); Cd_h.append(Cd)
    return dict(dp=dp, t=sol.t[idx], x=sol.y[0][idx], v=sol.y[1][idx],
                Tp=sol.y[2][idx], Re=np.array(Re_h), Mrel=np.array(M_h),
                Cd=np.array(Cd_h), impacted=impacted,
                v_imp=float(sol.y[1][-1]) if impacted else 0.0,
                T_imp=float(sol.y[2][-1]), t_flight=float(sol.t[-1]))


def deposition_summary(particles, weights, powder, crit_fn):
    rows = []
    de_mass = 0.0
    for p, w in zip(particles, weights):
        vcr = crit_fn(powder, p["T_imp"])
        vero = v_erosion(powder, p["T_imp"])
        if not p["impacted"] or p["v_imp"] < vcr:
            outcome = "Rebound" if p["impacted"] else "No impact"
        elif p["v_imp"] > vero:
            outcome = "Erosion"
        else:
            outcome = "Deposit"
            de_mass += w
        rows.append(dict(dp_um=p["dp"] * 1e6, mass_frac=w, v_imp=p["v_imp"],
                         T_imp=p["T_imp"], v_crit=vcr, v_erosion=vero,
                         eta=p["v_imp"] / vcr if vcr > 0 else 0.0,
                         t_flight_ms=p["t_flight"] * 1e3, outcome=outcome))
    return rows, de_mass


def run_quick(cfg, progress=None):
    """Run the whole quick model. cfg is the CSFLOW model dictionary."""
    noz, gas, field, info = centreline(cfg)
    powder = Powder.from_db(cfg["powder"]["name"], **cfg["powder"].get("overrides", {}))
    psd = cfg["powder"]["psd"]
    d, w = rosin_rammler_bins(psd["d50"], psd["n"], psd["dmin"], psd["dmax"],
                              psd["nbins"])
    drag_fn = DRAG_MODELS[cfg["model"]["drag"]]
    crit_fn = CRIT_MODELS[cfg["model"]["crit"]]
    inj = cfg["injection"]
    x0 = noz.x_in + inj["x_from_inlet"]
    particles = []
    for i, dp in enumerate(d):
        particles.append(track_particle(dp, field, noz, gas, powder, drag_fn,
                                        x0, inj["v0"], inj["Tp0"]))
        if progress:
            progress((i + 1) / len(d), f"particle {i + 1}/{len(d)}  d = {dp * 1e6:.1f} um")
    rows, de = deposition_summary(particles, w, powder, crit_fn)
    info["deposition_efficiency"] = de
    info["mean_v_imp"] = float(np.sum([r["v_imp"] * r["mass_frac"] for r in rows]))
    info["mean_T_imp"] = float(np.sum([r["T_imp"] * r["mass_frac"] for r in rows]))
    wall = dict(x=field["x"], r=np.where(field["x"] <= noz.x_exit,
                                         noz.radius(field["x"]), np.nan))
    return dict(mode="quick", field=field, info=info, particles=particles,
                table=rows, wall=wall)


if __name__ == "__main__":
    from csflow_model import default_model
    import time
    t = time.time()
    res = run_quick(default_model())
    print(res["info"])
    for r in res["table"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("time", time.time() - t)
