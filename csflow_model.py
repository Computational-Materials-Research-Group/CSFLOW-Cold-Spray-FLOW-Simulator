"""CSFLOW model definition: one dictionary holds every input, like an
Abaqus .cae model. It is saved as model.json inside each job folder."""
import copy
import json


def default_model():
    return {
        "name": "Cu_N2_3MPa",
        "nozzle": dict(d_in=8.0e-3, d_throat=2.0e-3, d_exit=6.0e-3,
                       L_conv=20e-3, L_div=130e-3, standoff=30e-3),
        "gas": {"name": "Nitrogen (N2)", "overrides": {}},
        "powder": {
            "name": "Copper (Cu)", "overrides": {},
            "psd": dict(d50=25e-6, n=3.0, dmin=5e-6, dmax=45e-6, nbins=12),
        },
        "bc": dict(p0=3.0e6, T0=673.15, p_amb=101325.0, T_amb=293.15),
        "injection": dict(x_from_inlet=0.0, v0=5.0, Tp0=293.15,
                          n_radial=5, r_frac_max=0.6),
        "model": {"drag": "Henderson (compressible)",
                  "crit": "Schmidt et al. (2006)", "bow_shock_k": 0.25},
        "mesh": dict(n_conv=30, n_div=150, n_jet=60, n_rad=20, n_rad_out=25,
                     domain_r_factor=6.0, wall_grading=0.5,
                     substrate_grading=0.4),
        "step": dict(turbulence="kOmegaSST", end_time=1.0e-3,
                     write_interval=1.0e-4, max_co=0.4, n_procs=1,
                     init_nozzle=True),
        "foam": dict(flavour="ESI (openfoam.com): rhoCentralFoam",
                     bashrc=""),
    }


def merge(base, upd):
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load_model(text):
    return merge(default_model(), json.loads(text))


def dump_model(m):
    return json.dumps(m, indent=2)
