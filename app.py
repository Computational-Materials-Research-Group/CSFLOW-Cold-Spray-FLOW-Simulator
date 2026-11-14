"""
CSFLOW | Cold Spray FLOW Simulator
==================================
Browser-based CFD environment for cold spray nozzles, built like CSIM:
Part -> Material -> Mesh -> Boundary conditions -> Step -> Job -> Visualization.

Run (WSL / Linux):
    streamlit run app.py --server.headless true
then open http://localhost:8501
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from csflow_model import default_model, dump_model, load_model  # noqa: E402
from csflow_physics import (GASES, MATERIALS, DRAG_MODELS, CRIT_MODELS, Gas,  # noqa: E402
                            Powder, Nozzle, rosin_rammler_bins, v_erosion)
import csflow_quick as quick  # noqa: E402
import csflow_foam as foam  # noqa: E402

RUNS = APP_DIR / "runs"
RUNS.mkdir(exist_ok=True)
C2K = 273.15

st.set_page_config(page_title="CSFLOW | Cold Spray FLOW Simulator",
                   page_icon="🌀", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3.2rem;}
.csf-bar {background: linear-gradient(90deg,#0b3954,#087e8b); color:#fff;
          padding: 10px 18px; border-radius: 8px; margin-bottom: 14px;
          display:flex; justify-content:space-between; align-items:center;}
.csf-bar b {font-size: 1.35rem; letter-spacing: .5px;}
.csf-bar span {opacity:.85; font-size:.9rem;}
.csf-module {font-size:1.25rem; font-weight:600; margin: 0 0 .4rem 0;}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = default_model()
M = st.session_state.model


def show_plot(fig, key=None, where=None):
    where = where or st
    cfg = {"scrollZoom": True, "displaylogo": False,
           "toImageButtonOptions": {"format": "png", "scale": 2}}
    try:
        where.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except Exception:
        where.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def num(label, value, key, scale=1.0, offset=0.0, **kw):
    """number_input in display units; returns SI value."""
    v = st.number_input(label, value=float(value * scale + offset), key=key, **kw)
    return (v - offset) / scale


def foam_available():
    if shutil.which("blockMesh"):
        return True
    rc = M["foam"].get("bashrc", "")
    return bool(rc) and Path(rc).exists()


def jobs():
    out = []
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or not (d / "model.json").exists():
            continue
        st_ = foam.get_status(d)
        meta = json.loads((d / "job.json").read_text()) if (d / "job.json").exists() else {}
        out.append(dict(name=d.name, path=d, mode=meta.get("mode", "?"),
                        state=st_.get("state", "written"), stage=st_.get("stage", ""),
                        created=meta.get("created", ""), error=st_.get("error", "")))
    return out


@st.cache_data(show_spinner=False)
def _load_results(path, mtime):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def results(job_dir):
    p = Path(job_dir) / "results.pkl"
    if not p.exists():
        return None
    return _load_results(str(p), p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def preview_1d(model_json):
    m = json.loads(model_json)
    noz, gas, field, info = quick.centreline(m)
    return field, info


def nozzle_shapes(noz: Nozzle, R_plot, scale=1e3, wall_t=None):
    """Plotly shapes for the nozzle walls (mirrored) and substrate."""
    wall_t = wall_t or 0.6 * noz.d_throat
    xs = np.linspace(noz.x_in, noz.x_exit, 200)
    rw = noz.radius(xs)
    shapes = []
    for sgn in (1, -1):
        px = np.concatenate([xs, xs[::-1]]) * scale
        py = sgn * np.concatenate([rw, rw[::-1] + wall_t]) * scale
        path = "M " + " L ".join(f"{a},{b}" for a, b in zip(px, py)) + " Z"
        shapes.append(dict(type="path", path=path, fillcolor="#5b6770",
                           line=dict(color="#2f363b", width=1)))
    shapes.append(dict(type="rect", x0=noz.x_sub * scale, x1=(noz.x_sub + 0.004) * scale,
                       y0=-R_plot * scale, y1=R_plot * scale,
                       fillcolor="#8d6e63", line=dict(color="#4e342e", width=1)))
    return shapes


def grade(n, G):
    """blockMesh simpleGrading node fractions for n cells, expansion G."""
    n = int(n)
    if abs(G - 1) < 1e-9 or n == 1:
        return np.linspace(0, 1, n + 1)
    q = G ** (1 / (n - 1))
    sizes = q ** np.arange(n)
    return np.concatenate([[0], np.cumsum(sizes) / sizes.sum()])


FIELDS = {"Mach number": ("M", "", 1.0),
          "Axial velocity": ("ux", "m/s", 1.0),
          "Static pressure": ("p", "kPa", 1e-3),
          "Static temperature": ("T", "°C", None),
          "Density": ("rho", "kg/m³", 1.0)}


def conv(key, arr):
    if key == "T":
        return arr - C2K
    if key == "p":
        return arr * 1e-3
    return arr


# --------------------------------------------------------------------------
# Title bar and model tree
# --------------------------------------------------------------------------
st.markdown("<div class='csf-bar'><b>CSFLOW</b><span>Cold Spray FLOW Simulator · "
            "nozzle gas dynamics &amp; particle acceleration</span></div>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model tree")
    M["name"] = st.text_input("Model name", M["name"])
    module = st.radio("Module", ["Part", "Material", "Mesh", "Boundary conditions",
                                 "Step", "Job", "Visualization", "Compare"],
                      label_visibility="collapsed")
    st.divider()
    st.download_button("💾 Save model (.json)", dump_model(M),
                       file_name=f"{M['name']}.json", mime="application/json",
                       width="stretch")
    up = st.file_uploader("Open model", type="json", label_visibility="collapsed")
    if up is not None and st.session_state.get("_loaded") != up.name + str(up.size):
        st.session_state.model = load_model(up.read().decode())
        st.session_state["_loaded"] = up.name + str(up.size)
        st.rerun()
    if st.button("↺ Reset to defaults", width="stretch"):
        st.session_state.model = default_model()
        st.rerun()
    st.caption("OpenFOAM: " + ("✅ found" if foam_available() else
                               "not found (quick mode only)"))

noz = Nozzle(**M["nozzle"])
gas = Gas.from_db(M["gas"]["name"], **M["gas"].get("overrides", {}))
powder = Powder.from_db(M["powder"]["name"], **M["powder"].get("overrides", {}))


# ==========================================================================
# PART
# ==========================================================================
if module == "Part":
    st.markdown("<div class='csf-module'>Part · nozzle and substrate geometry</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.2])
    with c1:
        n = M["nozzle"]
        st.markdown("**Nozzle (de Laval, conical)**")
        n["d_in"] = num("Inlet diameter (mm)", n["d_in"], "d_in", 1e3, min_value=0.5, step=0.5)
        n["d_throat"] = num("Throat diameter (mm)", n["d_throat"], "d_th", 1e3, min_value=0.2, step=0.1)
        n["d_exit"] = num("Exit diameter (mm)", n["d_exit"], "d_ex", 1e3, min_value=0.3, step=0.5)
        n["L_conv"] = num("Converging length (mm)", n["L_conv"], "L_c", 1e3, min_value=1.0, step=1.0)
        n["L_div"] = num("Diverging length (mm)", n["L_div"], "L_d", 1e3, min_value=1.0, step=5.0)
        st.markdown("**Substrate**")
        n["standoff"] = num("Standoff distance (mm)", n["standoff"], "sod", 1e3, min_value=2.0, step=5.0)
        if n["d_throat"] >= min(n["d_in"], n["d_exit"]):
            st.error("Throat must be smaller than inlet and exit.")
    noz = Nozzle(**M["nozzle"])
    with c2:
        R = noz.d_exit * 1.5
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[noz.x_in * 1e3, noz.x_sub * 1e3], y=[0, 0], mode="lines",
                                 line=dict(dash="dashdot", color="#999"), name="axis",
                                 hoverinfo="skip"))
        fig.update_layout(shapes=nozzle_shapes(noz, R), height=330,
                          margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          title="Axisymmetric section (radial direction stretched)",
                          xaxis=dict(title="x (mm), throat at 0",
                                     range=[noz.x_in * 1e3 - 5, noz.x_sub * 1e3 + 12]),
                          yaxis=dict(title="r (mm)", range=[-R * 1.15e3, R * 1.15e3]))
        for xx, lab in [(noz.x_in, "inlet"), (0, "throat"), (noz.x_exit, "exit"),
                        (noz.x_sub, "substrate")]:
            fig.add_annotation(x=xx * 1e3, y=R * 1e3 * 1.08, text=lab, showarrow=False,
                               font=dict(size=11, color="#555"))
        show_plot(fig)
        g = gas.gamma
        from csflow_physics import mach_from_area, p_ratio
        Me = mach_from_area(noz.area_ratio, g, True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expansion ratio Ae/A*", f"{noz.area_ratio:.2f}")
        k2.metric(f"Design exit Mach ({gas.name.split()[0]})", f"{Me:.2f}")
        k3.metric("Divergent half-angle", f"{np.degrees(np.arctan((noz.d_exit - noz.d_throat) / 2 / noz.L_div)):.2f}°")
        k4.metric("Total length", f"{(noz.L_conv + noz.L_div) * 1e3:.0f} mm")


# ==========================================================================
# MATERIAL
# ==========================================================================
elif module == "Material":
    st.markdown("<div class='csf-module'>Material · process gas and powder</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Process gas**")
        names = list(GASES)
        M["gas"]["name"] = st.selectbox("Gas", names, index=names.index(M["gas"]["name"]))
        base = GASES[M["gas"]["name"]]
        ov = M["gas"].setdefault("overrides", {})
        with st.expander("Gas properties", expanded=False):
            for k, lab in [("gamma", "γ (cp/cv)"), ("R", "R (J/kg·K)"),
                           ("mu0", "Sutherland μ₀ (Pa·s)"), ("Tmu0", "Sutherland T₀ (K)"),
                           ("S", "Sutherland S (K)"), ("Pr", "Prandtl number")]:
                v = st.number_input(lab, value=float(ov.get(k, base[k])), key=f"g_{M['gas']['name']}_{k}",
                                    format="%.4g")
                if abs(v - base[k]) > 1e-12 * max(abs(base[k]), 1):
                    ov[k] = v
                else:
                    ov.pop(k, None)
        gas = Gas.from_db(M["gas"]["name"], **ov)
        st.caption(f"cp = {gas.cp:.0f} J/kg·K · μ(300 K) = {gas.mu(300.0):.3e} Pa·s")
    with c2:
        st.markdown("**Powder material**")
        names = list(MATERIALS)
        M["powder"]["name"] = st.selectbox("Powder", names, index=names.index(M["powder"]["name"]))
        base = MATERIALS[M["powder"]["name"]]
        ov = M["powder"].setdefault("overrides", {})
        with st.expander("Powder properties", expanded=True):
            for k, lab, sc in [("rho", "Density (kg/m³)", 1), ("cp", "Specific heat (J/kg·K)", 1),
                               ("Tm", "Melting point (K)", 1), ("uts", "Ultimate tensile strength (MPa)", 1e-6)]:
                v = st.number_input(lab, value=float(ov.get(k, base[k]) * sc),
                                    key=f"p_{M['powder']['name']}_{k}", format="%.4g") / sc
                if abs(v - base[k]) > 1e-9 * max(abs(base[k]), 1):
                    ov[k] = v
                else:
                    ov.pop(k, None)
        powder = Powder.from_db(M["powder"]["name"], **ov)
        from csflow_physics import v_crit_schmidt
        st.caption(f"Critical velocity at 20 °C (Schmidt): {v_crit_schmidt(powder, 293.15):.0f} m/s")
    with c3:
        st.markdown("**Particle size distribution (Rosin-Rammler)**")
        psd = M["powder"]["psd"]
        psd["d50"] = num("Median d₅₀ (µm)", psd["d50"], "d50", 1e6, min_value=1.0)
        psd["n"] = st.number_input("Spread n", value=float(psd["n"]), min_value=0.5, step=0.5)
        a, b = st.columns(2)
        psd["dmin"] = a.number_input("d min (µm)", value=psd["dmin"] * 1e6, min_value=0.5) * 1e-6
        psd["dmax"] = b.number_input("d max (µm)", value=psd["dmax"] * 1e6, min_value=1.0) * 1e-6
        psd["nbins"] = int(st.slider("Size classes", 3, 30, int(psd["nbins"])))
        d, w = rosin_rammler_bins(psd["d50"], psd["n"], psd["dmin"], psd["dmax"], psd["nbins"])
        fig = go.Figure(go.Bar(x=d * 1e6, y=w * 100, marker_color="#087e8b"))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="d (µm)", yaxis_title="mass %")
        show_plot(fig)
    st.divider()
    st.markdown("**Powder injection**")
    inj = M["injection"]
    a, b, c, d_, e = st.columns(5)
    inj["x_from_inlet"] = a.number_input("Distance from inlet (mm)", value=inj["x_from_inlet"] * 1e3,
                                         min_value=0.0, help="0 = upstream (high-pressure) axial injection") * 1e-3
    inj["v0"] = b.number_input("Injection velocity (m/s)", value=float(inj["v0"]), min_value=0.0)
    inj["Tp0"] = c.number_input("Powder temperature (°C)", value=inj["Tp0"] - C2K) + C2K
    inj["n_radial"] = int(d_.number_input("Radial start points (full mode)", value=int(inj["n_radial"]),
                                          min_value=1, max_value=20))
    inj["r_frac_max"] = e.number_input("Max start radius / wall radius", value=float(inj["r_frac_max"]),
                                       min_value=0.0, max_value=0.95, step=0.1)


# ==========================================================================
# MESH
# ==========================================================================
elif module == "Mesh":
    st.markdown("<div class='csf-module'>Mesh · 2D axisymmetric wedge (full mode)</div>",
                unsafe_allow_html=True)
    me = M["mesh"]
    c1, c2 = st.columns([1, 2.4])
    with c1:
        st.markdown("**Seeds**")
        me["n_conv"] = int(st.number_input("Converging section, axial cells", value=int(me["n_conv"]), min_value=5))
        me["n_div"] = int(st.number_input("Diverging section, axial cells", value=int(me["n_div"]), min_value=10))
        me["n_jet"] = int(st.number_input("Jet region, axial cells", value=int(me["n_jet"]), min_value=5))
        me["n_rad"] = int(st.number_input("Nozzle / jet core, radial cells", value=int(me["n_rad"]), min_value=4))
        me["n_rad_out"] = int(st.number_input("Outer jet region, radial cells", value=int(me["n_rad_out"]), min_value=4))
        st.markdown("**Bias and domain**")
        me["wall_grading"] = st.number_input("Wall grading (last/first radial cell)", value=float(me["wall_grading"]),
                                             min_value=0.02, max_value=1.0, step=0.05,
                                             help="< 1 clusters cells at the nozzle wall")
        me["substrate_grading"] = st.number_input("Substrate grading (last/first axial cell)",
                                                  value=float(me["substrate_grading"]), min_value=0.02,
                                                  max_value=1.0, step=0.05)
        me["domain_r_factor"] = st.number_input("Domain radius / exit radius", value=float(me["domain_r_factor"]),
                                                min_value=2.0, max_value=20.0, step=0.5)
    noz = Nozzle(**M["nozzle"])
    with c2:
        ri, rt, re_ = noz.d_in / 2, noz.d_throat / 2, noz.d_exit / 2
        Rd = re_ * me["domain_r_factor"]
        blocks = [(noz.x_in, 0.0, me["n_conv"], 0.5, lambda x: noz.radius(x), 0.0, me["n_rad"], me["wall_grading"]),
                  (0.0, noz.x_exit, me["n_div"], 2.0, lambda x: noz.radius(x), 0.0, me["n_rad"], me["wall_grading"]),
                  (noz.x_exit, noz.x_sub, me["n_jet"], me["substrate_grading"], lambda x: re_ + 0 * x, 0.0, me["n_rad"], me["wall_grading"]),
                  (noz.x_exit, noz.x_sub, me["n_jet"], me["substrate_grading"], lambda x: Rd + 0 * x, re_, me["n_rad_out"], 4.0)]
        xs_l, ys_l = [], []
        dmin = 1e9
        for xa, xb, nx, gx, rtop, rbot, nr, gr in blocks:
            fx = xa + (xb - xa) * grade(nx, gx)
            fr = grade(nr, gr)
            for xx in fx:
                top = float(rtop(xx))
                xs_l += [xx, xx, None]
                ys_l += [rbot, top, None]
            for ff in fr:
                xx = np.linspace(xa, xb, 40)
                yy = rbot + (rtop(xx) - rbot) * ff
                xs_l += list(xx) + [None]
                ys_l += list(yy) + [None]
            dmin = min(dmin, np.min(np.diff(fx)), np.min(np.diff(fr)) * (float(rtop(xa)) - rbot),
                       np.min(np.diff(fr)) * (float(rtop(xb)) - rbot))
        fig = go.Figure(go.Scattergl(x=np.array(xs_l, dtype=float) * 1e3,
                                     y=np.array(ys_l, dtype=float) * 1e3, mode="lines",
                                     line=dict(color="#0b3954", width=0.6), hoverinfo="skip"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          title="Mesh preview (upper half, radial direction stretched; scroll to zoom)",
                          xaxis_title="x (mm)", yaxis=dict(title="r (mm)", range=[0, Rd * 1.03e3]))
        show_plot(fig)
        ncell = (me["n_conv"] + me["n_div"] + me["n_jet"]) * me["n_rad"] + me["n_jet"] * me["n_rad_out"]
        try:
            field, info = preview_1d(json.dumps(M))
            smax = float(np.max(field["u"] + np.sqrt(gas.gamma * gas.R * field["T"])))
        except Exception:
            smax = 1200.0
        # rhoCentralFoam includes the wedge front/back faces in its Courant number, so the
        # first cell on the axis at the throat (only 2·r·sin 2.5° wide) usually sets Δt.
        dr0 = rt * grade(me["n_rad"], me["wall_grading"])[1]
        d_wedge = 2 * np.sin(np.radians(foam.WEDGE_DEG)) * dr0 * 0.9
        c_star = np.sqrt(gas.gamma * gas.R * M["bc"]["T0"] * 2 / (gas.gamma + 1))
        dt = M["step"]["max_co"] * min(dmin / smax, d_wedge / c_star)
        steps = M["step"]["end_time"] / dt
        est = steps * ncell * 4e-6 / max(1, int(M["step"]["n_procs"]))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cells", f"{ncell:,}")
        k2.metric("Smallest cell", f"{dmin * 1e6:.0f} µm")
        k3.metric("Estimated Δt", f"{dt:.1e} s")
        k4.metric("Estimated run time", f"{est / 60:.0f} min", help=f"{steps:,.0f} time steps; "
                  "rough estimate at ~4 µs per cell per step per core")
        if st.button("Run blockMesh + checkMesh", disabled=not foam_available()):
            with st.spinner("Meshing..."):
                tmp = Path(tempfile.mkdtemp(prefix="csflow_mesh_"))
                foam.write_case(M, tmp)
                env = dict(os.environ)
                if M["foam"].get("bashrc"):
                    env["CSFLOW_FOAM_BASHRC"] = M["foam"]["bashrc"]
                cmd = ('[ -n "$CSFLOW_FOAM_BASHRC" ] && source "$CSFLOW_FOAM_BASHRC" >/dev/null 2>&1; '
                       'blockMesh > log.blockMesh 2>&1 && checkMesh 2>&1')
                out = subprocess.run(["bash", "-c", cmd], cwd=tmp, env=env, capture_output=True, text=True)
                txt = out.stdout + out.stderr
                lines = [l for l in txt.splitlines() if any(s in l for s in
                         ("cells:", "points:", "Max aspect", "non-orthogonality", "skewness",
                          "Mesh OK", "Failed", "***", "Wedge"))]
                (st.success if "Mesh OK" in txt else st.error)("\n\n".join(lines) or txt[-2000:])
                shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# BOUNDARY CONDITIONS
# ==========================================================================
elif module == "Boundary conditions":
    st.markdown("<div class='csf-module'>Boundary conditions · process parameters</div>",
                unsafe_allow_html=True)
    bc = M["bc"]
    c1, c2 = st.columns([1, 2])
    with c1:
        bc["p0"] = st.number_input("Stagnation (gun) pressure (MPa)", value=bc["p0"] / 1e6,
                                   min_value=0.2, max_value=10.0, step=0.1) * 1e6
        bc["T0"] = st.number_input("Stagnation (gun) temperature (°C)", value=bc["T0"] - C2K,
                                   min_value=-50.0, max_value=1100.0, step=25.0) + C2K
        bc["p_amb"] = st.number_input("Ambient pressure (kPa)", value=bc["p_amb"] / 1e3,
                                      min_value=1.0, step=1.0) * 1e3
        bc["T_amb"] = st.number_input("Ambient temperature (°C)", value=bc["T_amb"] - C2K) + C2K
    with c2:
        st.markdown("**Patches written to the OpenFOAM case**")
        st.dataframe(pd.DataFrame([
            ["inlet", "totalPressure p₀", "totalTemperature T₀", "pressureInletOutletVelocity"],
            ["nozzleWall", "zeroGradient", "zeroGradient (adiabatic)", "noSlip"],
            ["substrate", "zeroGradient", "zeroGradient (adiabatic)", "noSlip"],
            ["ambient", "totalPressure p_amb", "inletOutlet T_amb", "pressureInletOutletVelocity"],
            ["front / back", "wedge", "wedge", "wedge"],
            ["axis", "empty", "empty", "empty"]],
            columns=["Patch", "p", "T", "U"]), hide_index=True, width="stretch")
        try:
            field, info = preview_1d(json.dumps(M))
            st.markdown("**Quasi-1D prediction**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Nozzle regime", info["regime"].split(" (")[0])
            k2.metric("Exit Mach", f"{info['Me']:.2f}")
            k3.metric("Exit gas velocity", f"{info['ue']:.0f} m/s")
            k4.metric("Gas mass flow", f"{info['mdot'] * 1e3:.2f} g/s")
            st.caption(f"Exit pressure {info['pe'] / 1e3:.1f} kPa vs ambient {bc['p_amb'] / 1e3:.1f} kPa · "
                       f"exit temperature {info['Te'] - C2K:.0f} °C · "
                       f"bow shock: {'yes' if info['bow_shock'] else 'no'}")
        except Exception as e:
            st.error(f"Cannot evaluate this operating point: {e}")


# ==========================================================================
# STEP
# ==========================================================================
elif module == "Step":
    st.markdown("<div class='csf-module'>Step · physics models and solver controls</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Particle models (both modes)**")
        md = M["model"]
        md["drag"] = st.selectbox("Drag law", list(DRAG_MODELS), index=list(DRAG_MODELS).index(md["drag"]),
                                  help="Henderson includes compressibility and rarefaction; "
                                       "use it for supersonic flow.")
        md["crit"] = st.selectbox("Critical velocity model", list(CRIT_MODELS),
                                  index=list(CRIT_MODELS).index(md["crit"]))
        md["bow_shock_k"] = st.number_input("Quick mode: bow-shock layer thickness / exit diameter",
                                            value=float(md["bow_shock_k"]), min_value=0.05, max_value=2.0,
                                            step=0.05)
        st.caption("Heat transfer: Ranz-Marshall to the adiabatic recovery temperature. "
                   "Deposition window: v_crit ≤ v_impact ≤ v_erosion (= 2 v_crit).")
    with c2:
        st.markdown("**OpenFOAM solver (full mode)**")
        fo, sp = M["foam"], M["step"]
        fl = list(foam.FLAVOURS)
        fo["flavour"] = st.selectbox("OpenFOAM version", fl, index=fl.index(fo["flavour"]))
        fo["bashrc"] = st.text_input("OpenFOAM bashrc (blank if already sourced)", fo.get("bashrc", ""),
                                     placeholder="/usr/lib/openfoam/openfoam2412/etc/bashrc")
        sp["turbulence"] = st.selectbox("Turbulence", ["kOmegaSST", "laminar"],
                                        index=["kOmegaSST", "laminar"].index(sp["turbulence"]))
        a, b = st.columns(2)
        sp["end_time"] = a.number_input("End time (ms)", value=sp["end_time"] * 1e3, min_value=0.01,
                                        step=0.1, format="%.3f") * 1e-3
        sp["write_interval"] = b.number_input("Write interval (ms)", value=sp["write_interval"] * 1e3,
                                              min_value=0.001, step=0.05, format="%.3f") * 1e-3
        sp["max_co"] = a.number_input("Max Courant number", value=float(sp["max_co"]), min_value=0.05,
                                      max_value=0.9, step=0.05)
        sp["n_procs"] = int(b.number_input("CPU cores (MPI)", value=int(sp["n_procs"]), min_value=1,
                                           max_value=128))
        sp["init_nozzle"] = st.checkbox("Initialise converging section at p₀, T₀ (faster start-up)",
                                        value=bool(sp["init_nozzle"]))
        st.caption("Gas transit through the nozzle takes ~0.2 ms; 0.8–1.5 ms usually reaches a steady jet. "
                   "Check that the centreline Mach number stops changing between the last frames.")


# ==========================================================================
# JOB
# ==========================================================================
elif module == "Job":
    st.markdown("<div class='csf-module'>Job · submit, monitor and manage runs</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        jname = st.text_input("Job name", f"{M['name']}")
        mode = st.radio("Analysis type", ["Quick (quasi-1D, seconds)", "Full CFD (OpenFOAM)"],
                        index=0 if not foam_available() else 0)
        full = mode.startswith("Full")
        jdir = RUNS / jname.strip().replace(" ", "_")
        overwrite = st.checkbox("Overwrite if the job exists", value=False)
        b1, b2 = st.columns(2)
        submit = b1.button("▶ Submit", type="primary", width="stretch",
                           disabled=full and not foam_available())
        write_only = b2.button("Write input only", width="stretch", disabled=not full,
                               help="Writes the OpenFOAM case and offers it as a .zip")
        if full and not foam_available():
            st.info("OpenFOAM not found. Source its bashrc before starting the app or set the "
                    "path in Step. You can still write the case and run it yourself.")
        if submit or write_only:
            if jdir.exists() and not overwrite:
                st.error("A job with this name exists. Rename it or tick overwrite.")
            else:
                if jdir.exists():
                    if foam.get_status(jdir).get("state") == "running":
                        foam.kill_case(jdir)
                    shutil.rmtree(jdir)
                jdir.mkdir(parents=True)
                (jdir / "job.json").write_text(json.dumps(dict(mode="full" if full else "quick",
                                                               created=datetime.now().strftime("%Y-%m-%d %H:%M"))))
                if full:
                    foam.write_case(M, jdir)
                    if write_only:
                        foam.set_status(jdir, state="written", stage="")
                        buf = io.BytesIO()
                        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                            for p in jdir.rglob("*"):
                                if p.is_file():
                                    z.write(p, p.relative_to(jdir.parent))
                        st.session_state["_zip"] = (jdir.name, buf.getvalue())
                    else:
                        log = open(jdir / "job.log", "w")
                        proc = subprocess.Popen([sys.executable, str(APP_DIR / "csflow_foam.py"), "run", str(jdir)],
                                                cwd=APP_DIR, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                        foam.set_status(jdir, state="running", stage="starting", pid=proc.pid,
                                        started=time.time())
                        st.success(f"Job {jdir.name} submitted.")
                else:
                    (jdir / "model.json").write_text(dump_model(M))
                    bar = st.progress(0.0, "starting")
                    t0 = time.time()
                    try:
                        res = quick.run_quick(M, progress=lambda f_, msg: bar.progress(f_, msg))
                        res["info"]["wall_time"] = time.time() - t0
                        with open(jdir / "results.pkl", "wb") as fh:
                            pickle.dump(res, fh)
                        foam.set_status(jdir, state="done", stage="done", finished=time.time())
                        st.success(f"Completed in {time.time() - t0:.1f} s · deposition efficiency "
                                   f"{res['info']['deposition_efficiency'] * 100:.0f} %. "
                                   "Open Visualization to see the results.")
                    except Exception as e:
                        foam.set_status(jdir, state="failed", error=str(e))
                        st.error(f"Job failed: {e}")
                st.session_state["monitor"] = jdir.name
        if "_zip" in st.session_state:
            nm, data = st.session_state["_zip"]
            st.download_button(f"⬇ {nm}.zip (OpenFOAM case)", data, file_name=f"{nm}.zip")
            st.caption("Run it with `./Allrun` from a shell where OpenFOAM is sourced.")

    with c2:
        st.markdown("**Job manager**")
        J = jobs()
        if not J:
            st.caption("No jobs yet.")
        else:
            df = pd.DataFrame([dict(Job=j["name"], Type=j["mode"], Status=j["state"],
                                    Stage=j["stage"], Created=j["created"]) for j in J])
            st.dataframe(df, hide_index=True, width="stretch")
            names = [j["name"] for j in J]
            sel = st.selectbox("Monitor job", names,
                               index=names.index(st.session_state["monitor"])
                               if st.session_state.get("monitor") in names else 0)
            st.session_state["monitor"] = sel
            a, b, c = st.columns(3)
            if a.button("■ Kill", width="stretch"):
                foam.kill_case(RUNS / sel)
                st.rerun()
            if b.button("↻ Re-run post-processing", width="stretch",
                        help="Re-reads the OpenFOAM results and re-tracks particles"):
                with st.spinner("Post-processing..."):
                    foam.post_process(RUNS / sel)
                    foam.set_status(RUNS / sel, state="done", stage="done")
                st.rerun()
            if c.button("🗑 Delete", width="stretch"):
                if foam.get_status(RUNS / sel).get("state") == "running":
                    foam.kill_case(RUNS / sel)
                shutil.rmtree(RUNS / sel, ignore_errors=True)
                st.rerun()

            @st.fragment(run_every=3)
            def monitor(name):
                jd = RUNS / name
                stt = foam.get_status(jd)
                meta = json.loads((jd / "job.json").read_text()) if (jd / "job.json").exists() else {}
                state = stt.get("state", "written")
                if meta.get("mode") == "quick":
                    st.info(f"Quick job · {state}")
                    return
                m_ = json.loads((jd / "model.json").read_text())
                hist = foam.parse_log(jd)
                tnow = hist["time"][-1] if hist and hist["time"] else 0.0
                frac = min(tnow / m_["step"]["end_time"], 1.0)
                elapsed = (stt.get("finished") or time.time()) - stt.get("started", time.time())
                label = {"running": f"Running · {stt.get('stage', '')}", "done": "Completed",
                         "failed": "Failed", "killed": "Killed", "written": "Input written"}.get(state, state)
                st.progress(frac, f"{label} · t = {tnow * 1e3:.3f} / {m_['step']['end_time'] * 1e3:.3f} ms · "
                                  f"elapsed {elapsed / 60:.1f} min")
                if state == "failed":
                    st.error(stt.get("error", "Job failed."))
                if hist and hist["time"]:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Initial residuals", "Δt and Courant"),
                                        specs=[[{}, {"secondary_y": True}]])
                    for nm_, pts in hist["res"].items():
                        pts = np.array(pts)
                        if len(pts) > 1500:
                            pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
                        fig.add_trace(go.Scatter(x=pts[:, 0] * 1e3, y=pts[:, 1], name=nm_, mode="lines"), 1, 1)
                    fig.add_trace(go.Scatter(x=np.array(hist["time"]) * 1e3, y=hist["dt"], name="Δt",
                                             mode="lines", line=dict(color="#555")), 1, 2)
                    fig.add_trace(go.Scatter(x=np.array(hist["time"]) * 1e3, y=hist["co"], name="max Co",
                                             mode="lines", line=dict(color="#e07a5f", dash="dot")),
                                  1, 2, secondary_y=True)
                    fig.update_yaxes(type="log", row=1, col=1)
                    fig.update_xaxes(title_text="t (ms)")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                    show_plot(fig, key=f"mon_{name}_{len(hist['time'])}")
                    with st.expander("Solver log (tail)"):
                        st.code(hist["tail"], language=None)
                elif (jd / "job.log").exists():
                    st.code((jd / "job.log").read_text()[-2000:] or "starting...", language=None)
                if state not in ("running",):
                    st.caption("Monitoring paused (job not running).")

            monitor(sel)


# ==========================================================================
# VISUALIZATION
# ==========================================================================
elif module == "Visualization":
    st.markdown("<div class='csf-module'>Visualization · results</div>", unsafe_allow_html=True)
    done = [j for j in jobs() if (j["path"] / "results.pkl").exists()]
    if not done:
        st.info("No completed jobs yet. Submit one in the Job module.")
        st.stop()
    names = [j["name"] for j in done]
    sel = st.selectbox("Job", names, index=names.index(st.session_state["monitor"])
                       if st.session_state.get("monitor") in names else 0)
    jd = RUNS / sel
    R_ = results(jd)
    mj = json.loads((jd / "model.json").read_text())
    nz = Nozzle(**mj["nozzle"])
    gz = Gas.from_db(mj["gas"]["name"], **mj["gas"].get("overrides", {}))
    pw = Powder.from_db(mj["powder"]["name"], **mj["powder"].get("overrides", {}))
    info = R_["info"]

    k = st.columns(6)
    k[0].metric("Exit Mach (axis)", f"{info['Me']:.2f}")
    k[1].metric("Exit gas velocity", f"{info['ue']:.0f} m/s")
    k[2].metric("Gas mass flow", f"{info['mdot'] * 1e3:.2f} g/s")
    k[3].metric("Mean impact velocity", f"{info['mean_v_imp']:.0f} m/s")
    k[4].metric("Mean impact temperature", f"{info['mean_T_imp'] - C2K:.0f} °C")
    k[5].metric("Deposition efficiency", f"{info['deposition_efficiency'] * 100:.0f} %")
    st.caption(f"{'Full CFD (OpenFOAM, ' + format(info.get('n_cells', 0), ',') + ' cells, t = ' + format(info.get('t_final', 0) * 1e3, '.3f') + ' ms)' if R_['mode'] == 'full' else 'Quick mode (quasi-1D)'}"
               f" · {mj['gas']['name']} at {mj['bc']['p0'] / 1e6:.2f} MPa, {mj['bc']['T0'] - C2K:.0f} °C · "
               f"{mj['powder']['name']} · regime: {info['regime']}")

    tabs = st.tabs(["Contour", "Animation", "XY data", "Particles", "Report & export"])

    # ---------------- Contour ----------------
    with tabs[0]:
        a, b, c = st.columns([1.3, 1, 1])
        fname = a.selectbox("Field", list(FIELDS))
        key, unit, _ = FIELDS[fname]
        tracks = b.checkbox("Particle tracks", value=True)
        true_aspect = b.checkbox("True aspect ratio", value=False,
                                 help="Off: radial direction stretched so the flow is easier to read")
        cmap = c.selectbox("Colour map", ["Turbo", "Jet", "Viridis", "Inferno", "RdBu_r"])
        if R_["mode"] == "full":
            G = R_["grid"]
            xg, rg = G["x"], G["r"]
            Z = conv(key, G[key]).T
        else:
            F = R_["field"]
            xg = F["x"]
            Rmax = nz.d_exit / 2 * 2.2
            rg = np.linspace(0, Rmax, 120)
            src = {"M": F["M"], "ux": F["u"], "p": F["p"], "T": F["T"], "rho": F["rho"]}[key]
            Xq, Rq = np.meshgrid(xg, rg)
            rb = np.where(xg <= nz.x_exit, nz.radius(xg),
                          nz.d_exit / 2 + np.tan(np.radians(6)) * (xg - nz.x_exit))
            Z = np.where(Rq <= rb[None, :], conv(key, src)[None, :], np.nan)
            st.caption("Quick mode assumes uniform properties across each section "
                       "(the jet boundary is drawn with a 6° spreading half-angle).")
        Zm = np.vstack([Z[::-1], Z[1:]])
        rm = np.concatenate([-rg[::-1], rg[1:]])
        fig = go.Figure(go.Heatmap(x=xg * 1e3, y=rm * 1e3, z=Zm, colorscale=cmap, zsmooth="best",
                                   colorbar=dict(title=unit or fname), hovertemplate=
                                   "x %{x:.1f} mm<br>r %{y:.2f} mm<br>" + fname + " %{z:.4g}<extra></extra>"))
        if tracks:
            P = R_["particles"]
            vmax = max(float(np.max(p_["v"])) for p_ in P)
            pick = P if R_["mode"] == "full" else P
            for i, p_ in enumerate(pick):
                rr = p_["r"] if "r" in p_ else np.zeros_like(p_["x"])
                sgn = 1 if (R_["mode"] == "full" or i % 2 == 0) else -1
                fig.add_trace(go.Scatter(x=p_["x"] * 1e3, y=sgn * rr * 1e3, mode="lines",
                                         line=dict(width=1, color="rgba(255,255,255,0.55)"),
                                         hovertemplate=f"d = {p_['dp'] * 1e6:.1f} µm<br>x %{{x:.1f}} mm"
                                                       "<extra></extra>", showlegend=False))
        fig.update_layout(shapes=nozzle_shapes(nz, rg[-1]), height=430,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="x (mm)", range=[nz.x_in * 1e3, nz.x_sub * 1e3 + 5]),
                          yaxis=dict(title="r (mm)", range=[-rg[-1] * 1e3, rg[-1] * 1e3],
                                     **(dict(scaleanchor="x", scaleratio=1) if true_aspect else {})),
                          plot_bgcolor="#e9edf0")
        show_plot(fig, key="contour")
        st.caption("Mirrored axisymmetric section · scroll to zoom, double-click to reset. "
                   "Tip: zoom into the throat or the substrate stand-off region to see the shocks.")

    # ---------------- Animation ----------------
    with tabs[1]:
        if R_["mode"] == "full" and R_.get("frames"):
            fr = R_["frames"]
            fg = R_["frame_grid"]
            afield = st.radio("Animated field", ["M", "p", "T"], horizontal=True,
                              format_func=lambda s: {"M": "Mach", "p": "Pressure (kPa)", "T": "Temperature (°C)"}[s])
            mats = [conv(afield, f_[afield]).T for f_ in fr]
            zmin = float(np.nanmin([np.nanmin(z) for z in mats]))
            zmax = float(np.nanmax([np.nanmax(z) for z in mats]))
            rr = fg["r"]
            rm = np.concatenate([-rr[::-1], rr[1:]])

            def mirror(z):
                return np.vstack([z[::-1], z[1:]])
            fig = go.Figure(
                data=[go.Heatmap(x=fg["x"] * 1e3, y=rm * 1e3, z=mirror(mats[0]), zmin=zmin, zmax=zmax,
                                 colorscale="Turbo", zsmooth="best")],
                frames=[go.Frame(data=[go.Heatmap(z=mirror(z))], name=f"{f_['t'] * 1e3:.3f}")
                        for z, f_ in zip(mats, fr)])
            steps_ = [dict(method="animate", label=f"{f_['t'] * 1e3:.2f}",
                           args=[[f"{f_['t'] * 1e3:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                      for f_ in fr]
            fig.update_layout(
                shapes=nozzle_shapes(nz, rr[-1]), height=430, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="x (mm)"), yaxis=dict(title="r (mm)", range=[-rr[-1] * 1e3, rr[-1] * 1e3]),
                plot_bgcolor="#e9edf0",
                updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", direction="left", buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True)]),
                    dict(label="❚❚ Pause", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
                sliders=[dict(steps=steps_, x=0.15, len=0.85, y=-0.05, currentvalue=dict(prefix="t (ms) = "))])
            show_plot(fig, key="anim_full")
            st.caption("Start-up transient: the starting shock travels through the nozzle and the jet "
                       "establishes against the substrate.")
        else:
            st.markdown("**Particles in flight** (each line of dots is one size class)")
        # particle flight animation (both modes)
        P = R_["particles"]
        sizes = sorted(set(p_["dp"] for p_ in P))
        if R_["mode"] == "full":
            P_show = [min((q for q in P if q["dp"] == d_), key=lambda q: q["r0"]) for d_ in sizes]
        else:
            P_show = P
        tmax = max(p_["t"][-1] for p_ in P_show)
        tt = np.linspace(0, tmax, 80)
        nsz = len(P_show)
        lanes = np.linspace(-nz.d_in / 2 * 0.7, nz.d_in / 2 * 0.7, nsz)
        vmax = max(float(np.max(p_["v"])) for p_ in P_show)

        def state(t):
            xs, ys, vs, txt = [], [], [], []
            for lane, p_ in zip(lanes, P_show):
                if t <= p_["t"][-1]:
                    x = np.interp(t, p_["t"], p_["x"])
                    v = np.interp(t, p_["t"], p_["v"])
                else:
                    x, v = p_["x"][-1], p_["v"][-1]
                rw = float(nz.radius(min(x, nz.x_exit)))
                xs.append(x * 1e3)
                ys.append(lane / (nz.d_in / 2) * rw * 1e3 if x <= nz.x_exit else lane / (nz.d_in / 2) * nz.d_exit / 2 * 1e3)
                vs.append(v)
                txt.append(f"d = {p_['dp'] * 1e6:.1f} µm<br>v = {v:.0f} m/s")
            return xs, ys, vs, txt

        xs, ys, vs, txt = state(0)
        sz = [4 + 10 * (p_["dp"] - sizes[0]) / max(sizes[-1] - sizes[0], 1e-12) for p_ in P_show]
        mk = dict(size=sz, color=vs, cmin=0, cmax=vmax, colorscale="Turbo",
                  colorbar=dict(title="v (m/s)"), line=dict(width=0.5, color="#222"))
        figp = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="markers", marker=mk, text=txt,
                                          hovertemplate="%{text}<extra></extra>")],
                         frames=[go.Frame(data=[go.Scatter(x=s[0], y=s[1], text=s[3],
                                                           marker=dict(mk, color=s[2]))],
                                          name=f"{t * 1e3:.3f}")
                                 for t in tt for s in [state(t)]])
        figp.update_layout(
            shapes=nozzle_shapes(nz, nz.d_exit * 1.2), height=360, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="x (mm)", range=[nz.x_in * 1e3 - 2, nz.x_sub * 1e3 + 6]),
            yaxis=dict(title="r (mm)", range=[-nz.d_exit * 1.2e3, nz.d_exit * 1.2e3]),
            updatemenus=[dict(type="buttons", x=0, y=-0.15, xanchor="left", direction="left", buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=False), fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(steps=[dict(method="animate", label=f"{t * 1e3:.2f}",
                                      args=[[f"{t * 1e3:.3f}"], dict(mode="immediate", frame=dict(duration=0))])
                                 for t in tt], x=0.15, len=0.85, y=-0.08,
                          currentvalue=dict(prefix="t (ms) = "))])
        show_plot(figp, key="anim_particles")
        st.caption("Marker size scales with particle diameter; colour is particle speed. "
                   "Large particles lag behind; small ones accelerate fast but slow down in the bow-shock layer.")

    # ---------------- XY data ----------------
    with tabs[2]:
        F = R_["field"]
        a, b = st.columns(2)
        with a:
            gsel = st.multiselect("Gas, along the axis", ["Mach number", "Velocity (m/s)", "Temperature (°C)",
                                                          "Pressure (kPa)", "Density (kg/m³)"],
                                  default=["Mach number", "Velocity (m/s)"])
            gmap = {"Mach number": F["M"], "Velocity (m/s)": F["u"], "Temperature (°C)": F["T"] - C2K,
                    "Pressure (kPa)": F["p"] / 1e3, "Density (kg/m³)": F["rho"]}
            fig = make_subplots(rows=max(len(gsel), 1), cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for i, gname in enumerate(gsel):
                fig.add_trace(go.Scatter(x=F["x"] * 1e3, y=gmap[gname], name=gname, mode="lines"), i + 1, 1)
                fig.update_yaxes(title_text=gname, row=i + 1, col=1)
                for xx in (0, nz.x_exit * 1e3):
                    fig.add_vline(x=xx, line=dict(color="#aaa", dash="dot"), row=i + 1, col=1)
            fig.update_xaxes(title_text="x (mm)", row=max(len(gsel), 1), col=1)
            fig.update_layout(height=220 * max(len(gsel), 1) + 40, margin=dict(l=10, r=10, t=10, b=10),
                              showlegend=False)
            show_plot(fig, key="xy_gas")
        with b:
            P = R_["particles"]
            sizes = sorted(set(p_["dp"] for p_ in P))
            labels = [f"{d_ * 1e6:.1f} µm" for d_ in sizes]
            pick = st.multiselect("Particles, along their path", labels,
                                  default=[labels[0], labels[len(labels) // 2], labels[-1]])
            pq = st.radio("Quantity", ["Velocity", "Temperature"], horizontal=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=F["x"] * 1e3, y=F["u"] if pq == "Velocity" else F["T"] - C2K,
                                     name="gas (axis)", line=dict(color="#999", dash="dash")))
            for lab in pick:
                d_ = sizes[labels.index(lab)]
                cand = [q for q in P if q["dp"] == d_]
                q = min(cand, key=lambda q: q.get("r0", 0))
                fig.add_trace(go.Scatter(x=q["x"] * 1e3, y=q["v"] if pq == "Velocity" else q["Tp"] - C2K,
                                         name=lab, mode="lines"))
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="x (mm)",
                              yaxis_title="m/s" if pq == "Velocity" else "°C",
                              legend=dict(orientation="h", y=-0.18))
            show_plot(fig, key="xy_part")
        if R_["mode"] == "full":
            S = R_["substrate"]
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Substrate pressure", "Substrate gas temperature"))
            fig.add_trace(go.Scatter(x=S["r"] * 1e3, y=S["p"] / 1e3, mode="lines"), 1, 1)
            fig.add_trace(go.Scatter(x=S["r"] * 1e3, y=S["T"] - C2K, mode="lines"), 1, 2)
            fig.update_xaxes(title_text="r (mm)")
            fig.update_yaxes(title_text="kPa", row=1, col=1)
            fig.update_yaxes(title_text="°C", row=1, col=2)
            fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            show_plot(fig, key="xy_sub")

    # ---------------- Particles ----------------
    with tabs[3]:
        T_ = pd.DataFrame(R_["table"])
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            colors = {"Deposit": "#2a9d8f", "Rebound": "#e76f51", "Erosion": "#6d597a", "No impact": "#999"}
            fig.add_trace(go.Scatter(x=T_["dp_um"], y=T_["v_erosion"], mode="lines", name="v erosion",
                                     line=dict(color="#6d597a", dash="dot")))
            fig.add_trace(go.Scatter(x=T_["dp_um"], y=T_["v_crit"], mode="lines", name="v critical",
                                     line=dict(color="#e76f51", dash="dash"),
                                     fill="tonexty", fillcolor="rgba(42,157,143,0.10)"))
            if "v_imp_min" in T_:
                fig.add_trace(go.Scatter(x=list(T_["dp_um"]) + list(T_["dp_um"][::-1]),
                                         y=list(T_["v_imp_max"]) + list(T_["v_imp_min"][::-1]),
                                         fill="toself", fillcolor="rgba(11,57,84,0.15)", line=dict(width=0),
                                         name="radial spread", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=T_["dp_um"], y=T_["v_imp"], mode="lines+markers", name="impact velocity",
                                     line=dict(color="#0b3954"),
                                     marker=dict(size=9, color=[colors[o] for o in T_["outcome"]])))
            fig.update_layout(height=380, title="Impact velocity vs particle size (shaded: deposition window)",
                              xaxis_title="d (µm)", yaxis_title="m/s", margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h", y=-0.2))
            show_plot(fig, key="vimp")
        with b:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=T_["dp_um"], y=T_["mass_frac"] * 100, name="mass %",
                                 marker_color=[colors[o] for o in T_["outcome"]], opacity=0.7))
            fig.add_trace(go.Scatter(x=T_["dp_um"], y=T_["T_imp"] - C2K, name="impact T (°C)",
                                     mode="lines+markers", line=dict(color="#d62828")), secondary_y=True)
            fig.update_yaxes(title_text="mass %", secondary_y=False)
            fig.update_yaxes(title_text="°C", secondary_y=True, tickformat=".0f")
            fig.update_layout(height=380, title="Size distribution coloured by outcome · impact temperature",
                              xaxis_title="d (µm)", margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h", y=-0.2))
            show_plot(fig, key="timp")
        show = T_.rename(columns={"dp_um": "d (µm)", "mass_frac": "mass fraction", "v_imp": "v impact (m/s)",
                                  "v_imp_min": "v min (m/s)", "v_imp_max": "v max (m/s)",
                                  "T_imp": "T impact (K)", "v_crit": "v crit (m/s)", "v_erosion": "v erosion (m/s)",
                                  "eta": "v/v crit", "r_imp_mm": "impact radius (mm)",
                                  "t_flight_ms": "flight time (ms)", "outcome": "outcome"})
        st.dataframe(show.style.format(precision=2), hide_index=True, width="stretch")

    # ---------------- Report & export ----------------
    with tabs[4]:
        rep = {"Job": sel, "Analysis": "Full CFD" if R_["mode"] == "full" else "Quick (quasi-1D)",
               "Gas": mj["gas"]["name"], "Powder": mj["powder"]["name"],
               "p0 (MPa)": mj["bc"]["p0"] / 1e6, "T0 (°C)": mj["bc"]["T0"] - C2K,
               "Throat / exit diameter (mm)": f"{nz.d_throat * 1e3:.2f} / {nz.d_exit * 1e3:.2f}",
               "Expansion ratio": nz.area_ratio, "Standoff (mm)": nz.standoff * 1e3,
               "Nozzle regime": info["regime"], "Exit Mach (axis)": info["Me"],
               "Exit velocity (m/s)": info["ue"], "Exit temperature (°C)": info["Te"] - C2K,
               "Gas mass flow (g/s)": info["mdot"] * 1e3,
               "Mean impact velocity (m/s)": info["mean_v_imp"],
               "Mean impact temperature (°C)": info["mean_T_imp"] - C2K,
               "Deposition efficiency (%)": info["deposition_efficiency"] * 100}
        st.dataframe(pd.DataFrame([(k_, f"{v_:.4g}" if isinstance(v_, (float, np.floating)) else str(v_))
                                   for k_, v_ in rep.items()], columns=["Quantity", "Value"]),
                     hide_index=True, width="stretch")
        F = R_["field"]
        cl = pd.DataFrame({"x_mm": F["x"] * 1e3, "Mach": F["M"], "u_m_s": F["u"], "T_K": F["T"],
                           "p_Pa": F["p"], "rho_kg_m3": F["rho"]})
        a, b, c, d_ = st.columns(4)
        a.download_button("⬇ Centreline (.csv)", cl.to_csv(index=False), f"{sel}_centreline.csv", width="stretch")
        b.download_button("⬇ Particle table (.csv)", pd.DataFrame(R_["table"]).to_csv(index=False),
                          f"{sel}_particles.csv", width="stretch")
        handoff = {"source": "CSFLOW", "job": sel, "material": mj["powder"]["name"],
                   "gas": mj["gas"]["name"],
                   "particles": [dict(diameter_um=r_["dp_um"], radius_um=r_["dp_um"] / 2,
                                      impact_velocity_m_s=r_["v_imp"], impact_temperature_K=r_["T_imp"],
                                      impact_temperature_C=r_["T_imp"] - C2K, mass_fraction=r_["mass_frac"],
                                      critical_velocity_m_s=r_["v_crit"], outcome=r_["outcome"])
                                 for r_ in R_["table"]]}
        c.download_button("⬇ Send to CSIM (.json)", json.dumps(handoff, indent=2), f"{sel}_to_CSIM.json",
                          width="stretch", help="Impact velocity, temperature and radius per size class, "
                                                "ready to use as CSIM impact inputs")
        if R_["mode"] == "full":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in jd.rglob("*"):
                    if p.is_file() and "processor" not in p.parts[-3:] and p.suffix != ".pkl":
                        z.write(p, p.relative_to(jd.parent))
            d_.download_button("⬇ OpenFOAM case (.zip)", buf.getvalue(), f"{sel}_case.zip", width="stretch",
                               help="Open case.foam in ParaView")
        st.caption("The CSIM file lists one entry per size class. Pick the class you want to simulate "
                   "and enter its impact velocity, radius and temperature in CSIM.")


# ==========================================================================
# COMPARE
# ==========================================================================
elif module == "Compare":
    st.markdown("<div class='csf-module'>Compare · parametric studies</div>", unsafe_allow_html=True)
    done = [j for j in jobs() if (j["path"] / "results.pkl").exists()]
    if len(done) < 1:
        st.info("Run at least two jobs to compare them.")
        st.stop()
    names = [j["name"] for j in done]
    pick = st.multiselect("Jobs", names, default=names[:min(4, len(names))])
    if not pick:
        st.stop()
    rows = []
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Centreline Mach number", "Centreline gas velocity (m/s)"))
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Impact velocity (m/s)", "Impact temperature (°C)"))
    pal = ["#0b3954", "#e76f51", "#2a9d8f", "#f4a261", "#6d597a", "#d62828", "#3a86ff", "#8338ec"]
    for i, nm in enumerate(pick):
        jd = RUNS / nm
        R_ = results(jd)
        mj = json.loads((jd / "model.json").read_text())
        col = pal[i % len(pal)]
        F = R_["field"]
        fig1.add_trace(go.Scatter(x=F["x"] * 1e3, y=F["M"], name=nm, line=dict(color=col), legendgroup=nm), 1, 1)
        fig1.add_trace(go.Scatter(x=F["x"] * 1e3, y=F["u"], name=nm, line=dict(color=col), legendgroup=nm,
                                  showlegend=False), 1, 2)
        T_ = pd.DataFrame(R_["table"])
        fig2.add_trace(go.Scatter(x=T_["dp_um"], y=T_["v_imp"], name=nm, line=dict(color=col), legendgroup=nm),
                       1, 1)
        fig2.add_trace(go.Scatter(x=T_["dp_um"], y=T_["v_crit"], name=nm + " v crit",
                                  line=dict(color=col, dash="dot", width=1), legendgroup=nm, showlegend=False), 1, 1)
        fig2.add_trace(go.Scatter(x=T_["dp_um"], y=T_["T_imp"] - C2K, name=nm, line=dict(color=col),
                                  legendgroup=nm, showlegend=False), 1, 2)
        inf = R_["info"]
        rows.append({"Job": nm, "Type": R_["mode"], "Gas": mj["gas"]["name"], "Powder": mj["powder"]["name"],
                     "p0 (MPa)": mj["bc"]["p0"] / 1e6, "T0 (°C)": mj["bc"]["T0"] - C2K,
                     "Ae/A*": (mj["nozzle"]["d_exit"] / mj["nozzle"]["d_throat"]) ** 2,
                     "Exit Mach": inf["Me"], "Exit u (m/s)": inf["ue"], "ṁ (g/s)": inf["mdot"] * 1e3,
                     "Mean v imp (m/s)": inf["mean_v_imp"], "Mean T imp (°C)": inf["mean_T_imp"] - C2K,
                     "DE (%)": inf["deposition_efficiency"] * 100})
    for f_ in (fig1, fig2):
        f_.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    fig1.update_xaxes(title_text="x (mm)")
    fig2.update_xaxes(title_text="d (µm)")
    show_plot(fig1, key="cmp1")
    show_plot(fig2, key="cmp2")
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format(precision=2), hide_index=True, width="stretch")
    st.download_button("⬇ Comparison (.csv)", df.to_csv(index=False), "csflow_comparison.csv")
