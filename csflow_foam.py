"""
CSFLOW full mode: OpenFOAM case generator, job runner and post-processor.

* Writes a 2D axisymmetric (5 degree wedge) case for the nozzle, free jet and
  substrate, solved with the density-based Kurganov-Tadmor scheme
  (ESI: rhoCentralFoam, Foundation v11+: foamRun -solver shockFluid).
* Runs the case (optionally in parallel) and records progress in status.json.
* Reads the mesh and fields back without any OpenFOAM tools, resamples them
  onto an (x, r) grid and tracks powder particles through the converged gas
  field with the same compressible drag and heat transfer as quick mode
  (one-way coupling).

Command line (used by the app to run jobs in the background):
    python csflow_foam.py run  <case_dir>
    python csflow_foam.py post <case_dir>
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from csflow_physics import (Gas, Powder, Nozzle, particle_rhs, DRAG_MODELS,
                            CRIT_MODELS, rosin_rammler_bins)

WEDGE_DEG = 2.5          # half angle of the wedge (5 degree total)

FLAVOURS = {
    "ESI (openfoam.com): rhoCentralFoam":
        dict(solver="rhoCentralFoam", thermo_file="thermophysicalProperties",
             turb_file="turbulenceProperties", org=False),
    "Foundation (openfoam.org v11+): foamRun -solver shockFluid":
        dict(solver="foamRun -solver shockFluid", thermo_file="physicalProperties",
             turb_file="momentumTransport", org=True),
}


# --------------------------------------------------------------------------
# Dictionary writing helpers
# --------------------------------------------------------------------------
def header(cls, obj, loc=None):
    loc_line = f'    location    "{loc}";\n' if loc else ""
    return ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "  CSFLOW | Cold Spray FLOW simulator - generated case file\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{loc_line}    object      {obj};\n}}\n\n")


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def f(x):
    return f"{x:.8g}"


# --------------------------------------------------------------------------
# Mesh (blockMeshDict)
# --------------------------------------------------------------------------
def block_mesh_dict(m):
    noz = Nozzle(**m["nozzle"])
    me = m["mesh"]
    a = math.radians(WEDGE_DEG)
    ca, sa = math.cos(a), math.sin(a)
    ri, rt, re_ = noz.d_in / 2, noz.d_throat / 2, noz.d_exit / 2
    R = re_ * me["domain_r_factor"]
    x0, xt, xe, xs = noz.x_in, 0.0, noz.x_exit, noz.x_sub

    verts = []
    index = {}

    def v(x, r, side):
        """Vertex index; side = 'b' (back, z<0) or 'f' (front, z>0).
        Axis vertices (r = 0) are shared by both sides (collapsed hex)."""
        key = (round(x, 12), round(r, 12), "a" if r == 0 else side)
        if key not in index:
            z = 0.0 if r == 0 else (r * sa if side == "f" else -r * sa)
            index[key] = len(verts)
            verts.append((x, r * ca, z))
        return index[key]

    def hexv(xa, xb, ra_a, ra_b, rb_a, rb_b):
        """Block between axial stations xa, xb; inner radius ra_*, outer rb_*
        (suffix _a at xa, _b at xb)."""
        return [v(xa, ra_a, "b"), v(xb, ra_b, "b"), v(xb, rb_b, "b"), v(xa, rb_a, "b"),
                v(xa, ra_a, "f"), v(xb, ra_b, "f"), v(xb, rb_b, "f"), v(xa, rb_a, "f")]

    gw = me["wall_grading"]
    gs = me["substrate_grading"]
    blocks = [
        # name, vertices, cells (nx nr), grading (gx gr)
        ("conv", hexv(x0, xt, 0, 0, ri, rt), (me["n_conv"], me["n_rad"]), (0.5, gw)),
        ("div", hexv(xt, xe, 0, 0, rt, re_), (me["n_div"], me["n_rad"]), (2.0, gw)),
        ("jet", hexv(xe, xs, 0, 0, re_, re_), (me["n_jet"], me["n_rad"]), (gs, gw)),
        ("outer", hexv(xe, xs, re_, re_, R, R), (me["n_jet"], me["n_rad_out"]), (gs, 4.0)),
    ]

    def face(b, which):
        h = b[1]
        return {"xmin": (h[0], h[4], h[7], h[3]), "xmax": (h[1], h[2], h[6], h[5]),
                "rmin": (h[0], h[1], h[5], h[4]), "rmax": (h[3], h[7], h[6], h[2]),
                "back": (h[0], h[3], h[2], h[1]), "front": (h[4], h[5], h[6], h[7])}[which]

    B = {b[0]: b for b in blocks}
    patches = {
        "inlet": ("patch", [face(B["conv"], "xmin")]),
        "nozzleWall": ("wall", [face(B["conv"], "rmax"), face(B["div"], "rmax")]),
        "substrate": ("wall", [face(B["jet"], "xmax"), face(B["outer"], "xmax")]),
        "ambient": ("patch", [face(B["outer"], "xmin"), face(B["outer"], "rmax")]),
        "front": ("wedge", [face(b, "front") for b in blocks]),
        "back": ("wedge", [face(b, "back") for b in blocks]),
        "axis": ("empty", [face(B[n], "rmin") for n in ("conv", "div", "jet")]),
    }

    s = header("dictionary", "blockMeshDict")
    s += "scale 1;\n\nvertices\n(\n"
    for i, p in enumerate(verts):
        s += f"    ({f(p[0])} {f(p[1])} {f(p[2])})  // {i}\n"
    s += ");\n\nblocks\n(\n"
    for name, h, (nx, nr), (gx, gr) in blocks:
        s += (f"    hex ({' '.join(map(str, h))}) {name} ({nx} {nr} 1) "
              f"simpleGrading ({f(gx)} {f(gr)} 1)\n")
    s += ");\n\nedges\n(\n);\n\nboundary\n(\n"
    for name, (ptype, faces) in patches.items():
        s += f"    {name}\n    {{\n        type {ptype};\n        faces\n        (\n"
        for fc in faces:
            s += f"            ({' '.join(map(str, fc))})\n"
        s += "        );\n    }\n"
    s += ");\n\nmergePatchPairs\n(\n);\n"
    return s


# --------------------------------------------------------------------------
# Thermophysics, turbulence, numerics
# --------------------------------------------------------------------------
def thermo_dict(m, fl):
    gas = Gas.from_db(m["gas"]["name"], **m["gas"].get("overrides", {}))
    s = header("dictionary", fl["thermo_file"])
    s += f"""thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}

mixture
{{
    specie
    {{
        molWeight   {f(gas.W)};
    }}
    thermodynamics
    {{
        Cp          {f(gas.cp)};
        Hf          0;
    }}
    transport
    {{
        As          {f(gas.As)};
        Ts          {f(gas.S)};
    }}
}}
"""
    return s


def turb_dict(m, fl):
    model = m["step"]["turbulence"]
    s = header("dictionary", fl["turb_file"])
    if model == "laminar":
        s += "simulationType laminar;\n"
    else:
        s += f"simulationType RAS;\n\nRAS\n{{\n    {'model' if fl['org'] else 'RASModel'}  {model};\n    turbulence on;\n    printCoeffs on;\n}}\n"
        if not fl["org"]:
            pass
    return s


def control_dict(m, fl):
    st = m["step"]
    app = fl["solver"].split()[0]
    solver_line = "solver          shockFluid;\n" if fl["org"] else ""
    return header("dictionary", "controlDict") + f"""application     {app};
{solver_line}
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {f(st['end_time'])};
deltaT          1e-10;
writeControl    adjustableRunTime;
writeInterval   {f(st['write_interval'])};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           {f(st['max_co'])};
maxDeltaT       1e-5;
"""


def fv_schemes(m):
    return header("dictionary", "fvSchemes") + """fluxScheme      Kurganov;

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         Gauss linear;
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div(phi,epsilon) Gauss upwind;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
    reconstruct(rho) vanLeer;
    reconstruct(U)  vanLeerV;
    reconstruct(T)  vanLeer;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""


def fv_solution(m):
    return header("dictionary", "fvSolution") + """solvers
{
    "(rho|rhoU|rhoE)"
    {
        solver          diagonal;
    }

    "(U|e|h|k|omega|epsilon)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-09;
        relTol          0.01;
    }

    "(U|e|h|k|omega|epsilon)Final"
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors      0;
    nNonOrthogonalCorrectors 0;
}
"""


def decompose_dict(n):
    # 'simple' is available in every OpenFOAM build (scotch is not); slicing
    # along the nozzle axis gives balanced, compact subdomains here.
    return header("dictionary", "decomposeParDict") + f"""numberOfSubdomains {n};

method          simple;

coeffs
{{
    n           ({n} 1 1);
    delta       0.001;
    order       xyz;
}}

simpleCoeffs
{{
    n           ({n} 1 1);
    delta       0.001;
}}
"""


def set_fields_dict(m):
    noz = Nozzle(**m["nozzle"])
    bc = m["bc"]
    return header("dictionary", "setFieldsDict") + f"""defaultFieldValues
(
    volScalarFieldValue p {f(bc['p_amb'])}
    volScalarFieldValue T {f(bc['T_amb'])}
);

regions
(
    boxToCell
    {{
        box ({f(noz.x_in - 1)} -1 -1) (0 1 1);
        fieldValues
        (
            volScalarFieldValue p {f(bc['p0'])}
            volScalarFieldValue T {f(bc['T0'])}
        );
    }}
);
"""


# --------------------------------------------------------------------------
# Initial / boundary conditions
# --------------------------------------------------------------------------
def field_file(name, cls, dims, internal, bcs):
    s = header(cls, name, "0") + f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for patch, body in bcs.items():
        s += f"    {patch}\n    {{\n"
        for line in body.strip().splitlines():
            s += f"        {line.strip()}\n"
        s += "    }\n"
    s += "}\n"
    return s


def initial_fields(m):
    bc = m["bc"]
    gas = Gas.from_db(m["gas"]["name"], **m["gas"].get("overrides", {}))
    g = gas.gamma
    p0, T0, pa, Ta = bc["p0"], bc["T0"], bc["p_amb"], bc["T_amb"]
    noz = Nozzle(**m["nozzle"])
    wedge = {"front": "type wedge;", "back": "type wedge;", "axis": "type empty;"}
    files = {}
    files["p"] = field_file("p", "volScalarField", "[1 -1 -2 0 0 0 0]",
                            f"uniform {f(pa)}", {
        "inlet": f"type totalPressure;\np0 uniform {f(p0)};\nU U;\nphi phi;\nrho none;\npsi thermo:psi;\ngamma {f(g)};\nvalue uniform {f(p0)};",
        "nozzleWall": "type zeroGradient;",
        "substrate": "type zeroGradient;",
        "ambient": f"type totalPressure;\np0 uniform {f(pa)};\nU U;\nphi phi;\nrho none;\npsi thermo:psi;\ngamma {f(g)};\nvalue uniform {f(pa)};",
        **wedge})
    files["T"] = field_file("T", "volScalarField", "[0 0 0 1 0 0 0]",
                            f"uniform {f(Ta)}", {
        "inlet": f"type totalTemperature;\nT0 uniform {f(T0)};\nU U;\nphi phi;\npsi thermo:psi;\ngamma {f(g)};\nvalue uniform {f(T0)};",
        "nozzleWall": "type zeroGradient;",
        "substrate": "type zeroGradient;",
        "ambient": f"type inletOutlet;\ninletValue uniform {f(Ta)};\nvalue uniform {f(Ta)};",
        **wedge})
    files["U"] = field_file("U", "volVectorField", "[0 1 -1 0 0 0 0]",
                            "uniform (0 0 0)", {
        "inlet": "type pressureInletOutletVelocity;\nvalue uniform (0 0 0);",
        "nozzleWall": "type noSlip;",
        "substrate": "type noSlip;",
        "ambient": "type pressureInletOutletVelocity;\nvalue uniform (0 0 0);",
        **wedge})

    if m["step"]["turbulence"] != "laminar":
        # inlet turbulence: 5 % intensity, mixing length 7 % of inlet radius
        k_in = 1.5 * (0.05 * 20.0) ** 2
        Lm = 0.07 * noz.d_in / 2
        om_in = math.sqrt(k_in) / (0.09 ** 0.25 * Lm)
        files["k"] = field_file("k", "volScalarField", "[0 2 -2 0 0 0 0]",
                                f"uniform {f(k_in)}", {
            "inlet": f"type turbulentIntensityKineticEnergyInlet;\nintensity 0.05;\nvalue uniform {f(k_in)};",
            "nozzleWall": f"type kqRWallFunction;\nvalue uniform {f(k_in)};",
            "substrate": f"type kqRWallFunction;\nvalue uniform {f(k_in)};",
            "ambient": f"type inletOutlet;\ninletValue uniform {f(k_in)};\nvalue uniform {f(k_in)};",
            **wedge})
        files["omega"] = field_file("omega", "volScalarField", "[0 0 -1 0 0 0 0]",
                                    f"uniform {f(om_in)}", {
            "inlet": f"type turbulentMixingLengthFrequencyInlet;\nmixingLength {f(Lm)};\nk k;\nvalue uniform {f(om_in)};",
            "nozzleWall": f"type omegaWallFunction;\nvalue uniform {f(om_in)};",
            "substrate": f"type omegaWallFunction;\nvalue uniform {f(om_in)};",
            "ambient": f"type inletOutlet;\ninletValue uniform {f(om_in)};\nvalue uniform {f(om_in)};",
            **wedge})
        files["nut"] = field_file("nut", "volScalarField", "[0 2 -1 0 0 0 0]",
                                  "uniform 0", {
            "inlet": "type calculated;\nvalue uniform 0;",
            "nozzleWall": "type nutkWallFunction;\nvalue uniform 0;",
            "substrate": "type nutkWallFunction;\nvalue uniform 0;",
            "ambient": "type calculated;\nvalue uniform 0;",
            **wedge})
        files["alphat"] = field_file("alphat", "volScalarField", "[1 -1 -1 0 0 0 0]",
                                     "uniform 0", {
            "inlet": "type calculated;\nvalue uniform 0;",
            "nozzleWall": "type compressible::alphatWallFunction;\nPrt 0.85;\nvalue uniform 0;",
            "substrate": "type compressible::alphatWallFunction;\nPrt 0.85;\nvalue uniform 0;",
            "ambient": "type calculated;\nvalue uniform 0;",
            **wedge})
    return files


def allrun(m, fl):
    n = int(m["step"]["n_procs"])
    solver = fl["solver"]
    init = "setFields > log.setFields 2>&1\n" if m["step"]["init_nozzle"] else ""
    if n > 1:
        run = (f"echo \"STAGE decompose\"; decomposePar -force > log.decomposePar 2>&1\n"
               f"echo \"STAGE solve\"\n"
               f"mpirun -np {n} {solver} -parallel > log.solver 2>&1\n"
               f"reconstructPar > log.reconstructPar 2>&1\n")
    else:
        run = f"{solver} > log.solver 2>&1\n"
    return f"""#!/bin/bash
# CSFLOW generated run script
cd "${{0%/*}}" || exit 1
if [ -n "$CSFLOW_FOAM_BASHRC" ]; then
    source "$CSFLOW_FOAM_BASHRC" > /dev/null 2>&1
fi
set -e
echo "STAGE mesh";   blockMesh > log.blockMesh 2>&1
echo "STAGE check";  checkMesh > log.checkMesh 2>&1 || true
rm -rf 0 && cp -r 0.orig 0
echo "STAGE init";   {init.strip() or 'true'}
{'echo "STAGE solve"' if n == 1 else ''}
{run}echo "STAGE solved"
"""


def write_case(m, case: Path):
    fl = FLAVOURS[m["foam"]["flavour"]]
    case = Path(case)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(parents=True, exist_ok=True)
    write(case / "system" / "blockMeshDict", block_mesh_dict(m))
    write(case / "system" / "controlDict", control_dict(m, fl))
    write(case / "system" / "fvSchemes", fv_schemes(m))
    write(case / "system" / "fvSolution", fv_solution(m))
    write(case / "system" / "decomposeParDict", decompose_dict(max(1, int(m["step"]["n_procs"]))))
    write(case / "system" / "setFieldsDict", set_fields_dict(m))
    write(case / "constant" / fl["thermo_file"], thermo_dict(m, fl))
    write(case / "constant" / fl["turb_file"], turb_dict(m, fl))
    for name, text in initial_fields(m).items():
        write(case / "0.orig" / name, text)
    write(case / "Allrun", allrun(m, fl))
    os.chmod(case / "Allrun", 0o755)
    write(case / "model.json", json.dumps(m, indent=2))
    (case / "case.foam").touch()
    return case


# --------------------------------------------------------------------------
# Running and monitoring
# --------------------------------------------------------------------------
def set_status(case, **kw):
    p = Path(case) / "status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(kw, updated=time.time())
    p.write_text(json.dumps(st))


def get_status(case):
    p = Path(case) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_case(case):
    """Blocking: run Allrun, then post-process. Called by `python csflow_foam.py run`."""
    case = Path(case).resolve()
    m = json.loads((case / "model.json").read_text())
    env = dict(os.environ)
    if m["foam"].get("bashrc"):
        env["CSFLOW_FOAM_BASHRC"] = m["foam"]["bashrc"]
    set_status(case, state="running", stage="starting", pid=os.getpid(), started=time.time())
    proc = subprocess.Popen(["bash", str(case / "Allrun")], cwd=case, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.strip()
        print(line, flush=True)
        if line.startswith("STAGE"):
            set_status(case, stage=line.split()[1])
    rc = proc.wait()
    if rc != 0 or not latest_time(case):
        stage = get_status(case).get("stage", "?")
        set_status(case, state="failed", error=f"OpenFOAM stopped during '{stage}' (exit code {rc}). "
                   f"See log.{ {'mesh': 'blockMesh', 'init': 'setFields', 'decompose': 'decomposePar'}.get(stage, 'solver') } in the job folder.")
        return rc or 1
    set_status(case, stage="post")
    try:
        post_process(case)
    except Exception as e:  # noqa
        set_status(case, state="failed", error=f"Post-processing failed: {e}")
        raise
    set_status(case, state="done", stage="done", finished=time.time())
    return 0


def kill_case(case):
    st = get_status(case)
    pid = st.get("pid")
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    set_status(case, state="killed")


RE_TIME = re.compile(r"^Time = ([0-9.eE+-]+)", re.M)
RE_CO = re.compile(r"Mean and max Courant Numbers = ([0-9.eE+-]+) ([0-9.eE+-]+)")
RE_DT = re.compile(r"^deltaT = ([0-9.eE+-]+)", re.M)
RE_RES = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")
RE_EXEC = re.compile(r"ExecutionTime = ([0-9.eE+-]+) s")


def parse_log(case, max_points=2000):
    """Return time history (time, deltaT, Courant, residuals) from log.solver."""
    p = Path(case) / "log.solver"
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    chunks = RE_TIME.split(txt)
    hist = dict(time=[], dt=[], co=[], exec=[], res={})
    for i in range(1, len(chunks) - 1, 2):
        t = float(chunks[i])
        body = chunks[i + 1]
        dt = RE_DT.search(body)
        co = RE_CO.search(body)
        ex = RE_EXEC.search(body)
        hist["time"].append(t)
        hist["dt"].append(float(dt.group(1)) if dt else np.nan)
        hist["co"].append(float(co.group(2)) if co else np.nan)
        hist["exec"].append(float(ex.group(1)) if ex else np.nan)
        seen = {}
        for name, r in RE_RES.findall(body):
            if name.startswith("rho") or name == "Uz":
                continue          # diagonal (explicit) solves report no residual
            seen.setdefault(name, float(r))
        for name, r in seen.items():
            hist["res"].setdefault(name, []).append((t, r))
    n = len(hist["time"])
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        for k in ("time", "dt", "co", "exec"):
            hist[k] = [hist[k][i] for i in idx]
    hist["tail"] = "\n".join(txt.strip().splitlines()[-25:])
    return hist


# --------------------------------------------------------------------------
# Reading OpenFOAM ascii results (no OpenFOAM tools needed)
# --------------------------------------------------------------------------
def _strip(txt):
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//.*", "", txt)
    return txt


def _list_body(txt):
    """Return (n, body) of the first top-level counted list after the header."""
    txt = _strip(txt)
    i = txt.find("FoamFile")
    if i >= 0:
        txt = txt[txt.find("}", i) + 1:]
    mm = re.search(r"(\d+)\s*\(", txt)
    n = int(mm.group(1))
    start = mm.end()
    end = txt.rfind(")")
    return n, txt[start:end]


def read_points(case):
    n, body = _list_body((Path(case) / "constant/polyMesh/points").read_text())
    arr = np.array(body.replace("(", " ").replace(")", " ").split(), float)
    return arr.reshape(n, 3)


def read_faces(case):
    n, body = _list_body((Path(case) / "constant/polyMesh/faces").read_text())
    faces = [list(map(int, s.split())) for s in re.findall(r"\d+\(([^)]*)\)", body)]
    assert len(faces) == n
    return faces


def read_labels(case, name):
    n, body = _list_body((Path(case) / "constant/polyMesh" / name).read_text())
    return np.array(body.split(), int)


def cell_centres(case):
    pts = read_points(case)
    faces = read_faces(case)
    owner = read_labels(case, "owner")
    neigh = read_labels(case, "neighbour")
    ncell = owner.max() + 1
    fc = np.array([pts[f_].mean(axis=0) for f_ in faces])
    acc = np.zeros((ncell, 3))
    cnt = np.zeros(ncell)
    np.add.at(acc, owner, fc)
    np.add.at(cnt, owner, 1)
    nint = len(neigh)
    np.add.at(acc, neigh, fc[:nint])
    np.add.at(cnt, neigh, 1)
    return acc / cnt[:, None]


def read_field(path, ncell):
    txt = _strip(Path(path).read_text())
    mm = re.search(r"internalField\s+(uniform|nonuniform)", txt)
    rest = txt[mm.end():]
    if mm.group(1) == "uniform":
        val = rest[:rest.find(";")].strip()
        if val.startswith("("):
            return np.tile(np.array(val.strip("()").split(), float), (ncell, 1))
        return np.full(ncell, float(val))
    lm = re.search(r"List<(\w+)>\s*(\d+)\s*\(", rest)
    n = int(lm.group(2))
    body = rest[lm.end():]
    if lm.group(1) == "vector":
        end = body.find(")\n;")
        if end < 0:
            end = body.find(");")
        arr = np.array(body[:end].replace("(", " ").replace(")", " ").split()[:3 * n], float)
        return arr.reshape(n, 3)
    arr = np.array(body[:body.find(")")].split(), float)
    return arr[:n]


def time_dirs(case):
    out = []
    for d in Path(case).iterdir():
        try:
            t = float(d.name)
        except ValueError:
            continue
        if d.is_dir() and (d / "U").exists() and t > 0:
            out.append((t, d))
    return sorted(out)


def latest_time(case):
    td = time_dirs(case)
    return td[-1] if td else None


# --------------------------------------------------------------------------
# Resampling onto an (x, r) grid
# --------------------------------------------------------------------------
def resample(C, values, xg, rg, mask):
    from scipy.interpolate import griddata
    x = C[:, 0]
    r = np.hypot(C[:, 1], C[:, 2])
    X, Rr = np.meshgrid(xg, rg, indexing="ij")
    out = {}
    for k, v in values.items():
        lin = griddata((x, r), v, (X, Rr), method="linear")
        near = griddata((x, r), v, (X, Rr), method="nearest")
        lin = np.where(np.isnan(lin), near, lin)
        out[k] = lin
    out["mask"] = mask
    return out


def gas_fields(C, U, p, T, gas):
    r = np.hypot(C[:, 1], C[:, 2])
    ur = np.where(r > 1e-12, (U[:, 1] * C[:, 1] + U[:, 2] * C[:, 2]) / np.maximum(r, 1e-12), 0)
    rho = p / (gas.R * T)
    M = np.linalg.norm(U, axis=1) / np.sqrt(gas.gamma * gas.R * T)
    return dict(ux=U[:, 0], ur=ur, p=p, T=T, rho=rho, M=M)


class GridInterp:
    """Fast bilinear interpolation on a uniform (x, r) grid."""

    def __init__(self, xg, rg, fields):
        self.x0, self.dx, self.nx = xg[0], xg[1] - xg[0], len(xg)
        self.r0, self.dr, self.nr = rg[0], rg[1] - rg[0], len(rg)
        self.f = fields

    def __call__(self, x, r):
        fx = min(max((x - self.x0) / self.dx, 0), self.nx - 1.000001)
        fr = min(max((r - self.r0) / self.dr, 0), self.nr - 1.000001)
        i, j = int(fx), int(fr)
        a, b = fx - i, fr - j
        out = {}
        for k, F in self.f.items():
            out[k] = ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i + 1, j]
                      + (1 - a) * b * F[i, j + 1] + a * b * F[i + 1, j + 1])
        return out


def track_2d(dp, r0, gi: GridInterp, noz, gas, powder, drag_fn, x0, v0, Tp0,
             r_dom, e_wall=0.7, n_out=200):
    """RK4 integration of one particle (x, r, vx, vr, Tp) through the
    axisymmetric gas field with wall bounces in the nozzle."""
    y = np.array([x0, r0, v0, 0.0, Tp0])
    t = 0.0
    hist = [(t, *y)]
    mp_tau = powder.rho * dp ** 2 / 18.0

    def rhs(y):
        g = gi(y[0], abs(y[1]))
        urel = np.array([g["ux"] - y[2], g["ur"] - y[3]])
        acc, dT, *_ = particle_rhs(urel, g["T"], g["rho"], gas, dp, y[4], powder, drag_fn)
        return np.array([y[2], y[3], acc[0], acc[1], dT]), g

    impacted, reason = False, "max steps"
    for _ in range(200000):
        k1, g = rhs(y)
        speed = max(abs(y[2]), abs(g["ux"]), 1.0)
        tau = mp_tau / float(gas.mu(g["T"]))
        dt = min(0.5 * gi.dx / speed, 0.5 * tau, 2e-5)
        k2, _ = rhs(y + 0.5 * dt * k1)
        k3, _ = rhs(y + 0.5 * dt * k2)
        k4, _ = rhs(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if y[1] < 0:                      # crossed the axis: mirror
            y[1], y[3] = -y[1], -y[3]
        if y[0] <= noz.x_exit:            # bounce off nozzle wall
            rw = float(noz.radius(y[0])) - dp / 2
            if y[1] > rw:
                y[1] = 2 * rw - y[1]
                y[3] = -e_wall * abs(y[3])
        hist.append((t, *y))
        if y[0] >= noz.x_sub:
            impacted, reason = True, "impact"
            break
        if y[1] > r_dom or y[2] < 0.5 and y[0] > noz.x_exit:
            reason = "left domain" if y[1] > r_dom else "stalled"
            break
    H = np.array(hist)
    idx = np.linspace(0, len(H) - 1, min(n_out, len(H))).astype(int)
    H = H[idx]
    return dict(dp=dp, r0=r0, t=H[:, 0], x=H[:, 1], r=H[:, 2], vx=H[:, 3],
                vr=H[:, 4], Tp=H[:, 5], v=np.hypot(H[:, 3], H[:, 4]),
                impacted=impacted, reason=reason,
                v_imp=float(H[-1, 3]) if impacted else 0.0,
                T_imp=float(H[-1, 5]), r_imp=float(H[-1, 2]),
                t_flight=float(H[-1, 0]))


def post_process(case, nx=700, nr=140, frame_nx=350, frame_nr=70, log=print):
    case = Path(case)
    m = json.loads((case / "model.json").read_text())
    noz = Nozzle(**m["nozzle"])
    gas = Gas.from_db(m["gas"]["name"], **m["gas"].get("overrides", {}))
    powder = Powder.from_db(m["powder"]["name"], **m["powder"].get("overrides", {}))
    R = noz.d_exit / 2 * m["mesh"]["domain_r_factor"]

    log("reading mesh")
    C = cell_centres(case)
    ncell = len(C)
    xg = np.linspace(noz.x_in, noz.x_sub, nx)
    rg = np.linspace(0, R, nr)
    X, Rr = np.meshgrid(xg, rg, indexing="ij")
    mask = (X <= noz.x_exit) & (Rr > noz.radius(X))

    td = time_dirs(case)
    t_last, d_last = td[-1]
    log(f"reading fields at t = {t_last:g}")
    U = read_field(d_last / "U", ncell)
    p = read_field(d_last / "p", ncell)
    T = read_field(d_last / "T", ncell)
    gf = gas_fields(C, U, p, T, gas)
    grid = resample(C, gf, xg, rg, mask)

    # transient frames (Mach) for animation
    log("building animation frames")
    fxg = np.linspace(noz.x_in, noz.x_sub, frame_nx)
    frg = np.linspace(0, R, frame_nr)
    FX, FR = np.meshgrid(fxg, frg, indexing="ij")
    fmask = (FX <= noz.x_exit) & (FR > noz.radius(FX))
    frames = []
    for t, d in td:
        Ut = read_field(d / "U", ncell)
        Tt = read_field(d / "T", ncell)
        pt = read_field(d / "p", ncell)
        g_t = gas_fields(C, Ut, pt, Tt, gas)
        fr = resample(C, {"M": g_t["M"], "p": g_t["p"], "T": g_t["T"]}, fxg, frg, fmask)
        frames.append(dict(t=t, **{k: np.where(fmask, np.nan, fr[k]).astype(np.float32)
                                   for k in ("M", "p", "T")}))

    # particle tracking through the latest field
    log("tracking particles")
    gi = GridInterp(xg, rg, {k: grid[k] for k in ("ux", "ur", "T", "rho")})
    psd = m["powder"]["psd"]
    d, w = rosin_rammler_bins(psd["d50"], psd["n"], psd["dmin"], psd["dmax"], psd["nbins"])
    inj = m["injection"]
    x0 = noz.x_in + inj["x_from_inlet"]
    rw0 = float(noz.radius(x0))
    r_list = np.linspace(0, inj["r_frac_max"] * rw0, int(inj["n_radial"]))
    drag_fn = DRAG_MODELS[m["model"]["drag"]]
    crit_fn = CRIT_MODELS[m["model"]["crit"]]
    particles = []
    for i, dp in enumerate(d):
        for r0 in r_list:
            pr = track_2d(dp, r0, gi, noz, gas, powder, drag_fn, x0, inj["v0"],
                          inj["Tp0"], R)
            pr["w"] = w[i] / len(r_list)
            particles.append(pr)
        log(f"particles {i + 1}/{len(d)}")

    from csflow_physics import v_erosion
    rows, de_mass = [], 0.0
    for i, dp in enumerate(d):
        group = [q for q in particles if q["dp"] == dp]
        vi = np.array([q["v_imp"] for q in group])
        Ti = np.array([q["T_imp"] for q in group])
        for q in group:
            vcr = crit_fn(powder, q["T_imp"])
            vero = v_erosion(powder, q["T_imp"])
            if q["impacted"] and vcr <= q["v_imp"] <= vero:
                de_mass += q["w"]
                q["outcome"] = "Deposit"
            elif q["impacted"] and q["v_imp"] > vero:
                q["outcome"] = "Erosion"
            else:
                q["outcome"] = "Rebound" if q["impacted"] else "No impact"
        Tm = float(Ti.mean())
        vcr = crit_fn(powder, Tm)
        rows.append(dict(dp_um=dp * 1e6, mass_frac=float(w[i]), v_imp=float(vi.mean()),
                         v_imp_min=float(vi.min()), v_imp_max=float(vi.max()),
                         T_imp=Tm, v_crit=vcr, v_erosion=v_erosion(powder, Tm),
                         eta=float(vi.mean() / vcr), r_imp_mm=float(np.mean([q["r_imp"] for q in group]) * 1e3),
                         t_flight_ms=float(np.mean([q["t_flight"] for q in group]) * 1e3),
                         outcome=max(set(q["outcome"] for q in group),
                                     key=[q["outcome"] for q in group].count)))

    # centreline and wall data
    j0 = 0
    cl = {k: grid[k][:, j0] for k in ("ux", "p", "T", "rho", "M")}
    cl["x"] = xg
    cl["u"] = cl.pop("ux")
    # substrate pressure / temperature distribution (last x column)
    sub = dict(r=rg, p=grid["p"][-2, :], T=grid["T"][-2, :])
    info = dict(t_final=t_last, n_cells=ncell, mdot=_mass_flow(grid, xg, rg, noz),
                Me=float(np.interp(noz.x_exit, xg, cl["M"])),
                M_max=float(np.nanmax(np.where(mask, np.nan, grid["M"]))),
                ue=float(np.interp(noz.x_exit, xg, cl["u"])),
                Te=float(np.interp(noz.x_exit, xg, cl["T"])),
                pe=float(np.interp(noz.x_exit, xg, cl["p"])),
                deposition_efficiency=de_mass,
                mean_v_imp=float(sum(q["v_imp"] * q["w"] for q in particles)),
                mean_T_imp=float(sum(q["T_imp"] * q["w"] for q in particles)),
                regime="CFD (2D axisymmetric)")
    res = dict(mode="full", info=info, grid=dict(x=xg, r=rg, **{k: np.where(mask, np.nan, grid[k]) for k in gf}),
               frames=frames, frame_grid=dict(x=fxg, r=frg), field=cl, substrate=sub,
               particles=particles, table=rows,
               wall=dict(x=xg, r=np.where(xg <= noz.x_exit, noz.radius(xg), np.nan)))
    with open(case / "results.pkl", "wb") as fh:
        pickle.dump(res, fh)
    log("results written")
    return res


def _mass_flow(grid, xg, rg, noz, n_planes=9):
    """Gas mass flow: integral of rho*u*2*pi*r dr over nozzle cross-sections in
    the diverging part, averaged over several planes (steady flow conserves it)."""
    trap = getattr(np, "trapezoid", None) or np.trapz
    vals = []
    for xp in np.linspace(0.3, 0.8, n_planes) * noz.x_exit:
        i = int(np.argmin(np.abs(xg - xp)))
        rw = float(noz.radius(xg[i]))
        rr = np.concatenate([rg[rg < rw], [rw]])
        flux = grid["rho"][i] * grid["ux"][i]
        ff = np.interp(rr, rg, flux)
        vals.append(trap(2 * np.pi * rr * ff, rr))
    return float(np.median(vals))


def load_results(case):
    p = Path(case) / "results.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    cmd, case = sys.argv[1], sys.argv[2]
    if cmd == "run":
        sys.exit(run_case(case))
    elif cmd == "post":
        post_process(case)
    elif cmd == "write":
        from csflow_model import default_model
        write_case(default_model(), Path(case))
