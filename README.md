# CSFLOW: Cold Spray FLOW Simulator, a Browser-Based CFD Environment for Cold Spray Nozzles and Particle Acceleration


![OpenFOAM](https://img.shields.io/badge/OpenFOAM-rhoCentralFoam-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-Streamlit%20App-red?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-2D%20Axisymmetric%20Wedge-purple?style=for-the-badge)
![Drag](https://img.shields.io/badge/Drag-Henderson%20Compressible-orange?style=for-the-badge)
![Flow](https://img.shields.io/badge/Flow-Supersonic%20de%20Laval-black?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-green?style=for-the-badge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24224141-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.24224141)

A **browser-based CFD environment for the gas and particle flow of the cold spray process**, built on OpenFOAM and Python. Process gas (N₂, He or air) expands through a converging–diverging (de Laval) nozzle to supersonic speed, forms a free jet and strikes the substrate through a bow shock. Powder particles are injected at the nozzle inlet and accelerated and heated by the gas. CSFLOW predicts the gas field, the particle trajectories, the **impact velocity and impact temperature of every particle size**, and the resulting **deposition efficiency**.

The app works like a conventional simulation package and follows the same workflow as **CSIM (Cold Spray SIMulator)**: the user moves through the modules of a model tree (**Part → Material → Mesh → Boundary conditions → Step → Job → Visualization → Compare**), submits and monitors jobs, and post-processes the results with contour plots, animations, XY data and reports. The impact conditions can be exported directly to CSIM for the finite element simulation of the particle impact (splat formation).

Two analysis types are available: a **quick quasi-1D model** that runs in under a second for parametric sweeps, and a **full 2D axisymmetric CFD model** that writes, runs and post-processes an OpenFOAM case.


<img width="1500" height="1100" alt="CSFLOW_CFD_result" src="https://github.com/user-attachments/assets/036b1577-5876-4e85-a9e4-b6e2b07c16f5" />



---

## Physics

### Gas phase, full CFD (OpenFOAM)

- 2D axisymmetric domain: a 5° wedge about the nozzle axis, covering the nozzle, the free jet and the substrate
- Compressible Navier–Stokes equations solved with the density-based Kurganov–Tadmor central scheme (ESI: `rhoCentralFoam`; Foundation v11+: `foamRun -solver shockFluid`)
- Perfect gas, constant specific heat, Sutherland viscosity (`hePsiThermo`)
- Turbulence: k-ω SST with wall functions (laminar is also available)
- Captures the throat, the supersonic expansion, the over- or under-expanded jet with its shock diamonds, the bow shock in front of the substrate and the wall boundary layer
- The converging section can be initialised at the stagnation state (`setFields`) for a faster start-up

### Gas phase, quick mode (quasi-1D)

- Isentropic flow through the nozzle area variation, with detection of the operating regime: not choked, normal shock inside the nozzle, over-expanded, ideally expanded or under-expanded
- Normal shock position found from the exit pressure balance when the back pressure is too high for shock-free flow
- Free jet: blend from the exit state to the fully expanded jet Mach number, a supersonic core of empirical length, then velocity decay
- Bow shock: a normal shock at a set distance in front of the substrate, followed by a linear stagnation-point deceleration to the wall

### Particle phase (both modes)

- Lagrangian tracking of spherical particles, one-way coupled (the particles do not affect the gas)
- **Henderson (1976) drag law**, valid from incompressible to hypersonic flow and from continuum to free-molecular flow; the standard Schiller–Naumann law is available for comparison
- Particle heating by **Ranz–Marshall** convection towards the adiabatic recovery temperature
- Full mode: 2D tracking (x, r) through the converged CFD field with RK4 integration, several radial start points per size class and inelastic bounces on the nozzle wall
- Particle sizes from a **Rosin–Rammler** mass distribution
- Outcome of each impact from the **critical velocity** (Schmidt et al. 2006 or Assadi et al. 2003) and the erosion velocity: deposit, rebound or erosion
- **Deposition efficiency** = mass fraction of particles that land inside the deposition window

---

## Process Conditions

| Quantity                          | Value                       |
| --------------------------------- | --------------------------- |
| Process gas                       | Nitrogen (N₂)               |
| Stagnation (gun) pressure `p0`    | 3.0 MPa                     |
| Stagnation (gun) temperature `T0` | 400 °C (673.15 K)           |
| Ambient pressure `p_amb`          | 101.325 kPa                 |
| Ambient temperature `T_amb`       | 20 °C                       |
| Powder                            | Copper (Cu)                 |
| Powder injection                  | Axial, at the nozzle inlet  |
| Injection velocity                | 5 m/s                       |
| Powder temperature                | 20 °C                       |
| Simulated time (full CFD)         | 1.0 ms                      |
| Write interval                    | 0.1 ms                      |
| Maximum Courant number            | 0.4                         |

These are the defaults of the app. Every value can be changed in the Material, Boundary conditions and Step modules.

---

## Model Parameters

### Nozzle and Substrate (Part)

| Parameter                   | Value     | Role                                            |
| --------------------------- | --------- | ----------------------------------------------- |
| Inlet diameter `d_in`       | 8.0 mm    | Diameter of the pre-chamber end                 |
| Throat diameter `d_throat`  | 2.0 mm    | Sets the gas mass flow                          |
| Exit diameter `d_exit`      | 6.0 mm    | Sets the exit Mach number with `d_throat`       |
| Converging length `L_conv`  | 20 mm     | Conical converging section                      |
| Diverging length `L_div`    | 130 mm    | Conical diverging section                       |
| Standoff distance           | 30 mm     | Nozzle exit to substrate                        |
| Expansion ratio `Ae/A*`     | 9.0       | Design exit Mach 3.81 with N₂                   |
| Divergent half-angle        | 0.88°     | From the diameters and `L_div`                  |

### Powder Size Distribution (Material)

| Parameter            | Value   | Role                                           |
| -------------------- | ------- | ---------------------------------------------- |
| Median `d50`         | 25 µm   | Rosin–Rammler median (mass based)              |
| Spread `n`           | 3.0     | Larger = narrower distribution                 |
| Size range           | 5–45 µm | Smallest and largest particle                  |
| Size classes         | 12      | Number of tracked diameters                    |
| Radial start points  | 5       | Full mode only, from the axis to 0.6 × the inlet radius |

### Mesh (blockMesh)

| Parameter                         | Value | Role                                             |
| --------------------------------- | ----- | ------------------------------------------------ |
| Converging section, axial cells   | 30    | Graded towards the throat                        |
| Diverging section, axial cells    | 150   | Graded towards the throat                        |
| Jet region, axial cells           | 60    | Graded towards the substrate (ratio 0.4)         |
| Nozzle / jet core, radial cells   | 20    | Graded towards the nozzle wall (ratio 0.5)       |
| Outer jet region, radial cells    | 25    | Graded away from the jet                         |
| Domain radius / exit radius       | 6.0   | Radial extent of the jet region                  |
| Wedge angle                       | 5°    | 2.5° each side of the x–y plane                  |

### Gas Properties

| Gas           | γ     | R (J/kg·K) | Sutherland μ₀ (Pa·s) | Sutherland S (K) | Pr   |
| ------------- | ----- | ---------- | -------------------- | ---------------- | ---- |
| Nitrogen (N₂) | 1.40  | 296.8      | 1.663e-5             | 106.7            | 0.71 |
| Helium (He)   | 1.667 | 2077.1     | 1.870e-5             | 79.4             | 0.67 |
| Air           | 1.40  | 287.05     | 1.716e-5             | 110.4            | 0.71 |

μ₀ is given at 273.15 K.

### Powder Materials

| Material          | Density (kg/m³) | cp (J/kg·K) | Melting point (K) | UTS (MPa) |
| ----------------- | --------------- | ----------- | ----------------- | --------- |
| Copper (Cu)       | 8960            | 385         | 1358              | 220       |
| Aluminium (Al)    | 2700            | 900         | 933               | 90        |
| Al 6061           | 2700            | 896         | 855               | 310       |
| Titanium (CP-Ti)  | 4510            | 520         | 1941              | 345       |
| Ti-6Al-4V         | 4430            | 526         | 1878              | 950       |
| Nickel (Ni)       | 8900            | 444         | 1728              | 317       |
| Stainless 316L    | 7990            | 500         | 1673              | 485       |
| Zinc (Zn)         | 7140            | 388         | 693               | 110       |

Every gas and powder property is editable in the app.

---

## Geometry

```
Axisymmetric section (x = nozzle axis, r = radius, units: mm):

         inlet         throat                               exit            substrate
           |             |                                    |                  |
  r = 4.0  |\            |                       ____________/|  outer jet       |#|
           | \           |          ____________/             |  region          |#|
           |  \__________|_________/                          |                  |#|
  r = 1.0  |   converging  |          diverging               |  free jet        |#|
 ----------+---------------+----------------------------------+------------------|#|--- axis
           |<-- 20 mm ---->|<------------- 130 mm ----------->|<---- 30 mm ----->|#|
         x = -20         x = 0                              x = 130          x = 160

Blocks of the wedge mesh:

  +--------+-----------------------------+------------------+
  |        |                             |   outer (4)      |  r = 18 mm (6 x exit radius)
  |        |                             +------------------+  r = 3 mm
  | conv(1)|          div (2)            |   jet core (3)   |
  +--------+-----------------------------+------------------+  axis (r = 0)
  inlet                                 exit            substrate

Patches: inlet, nozzleWall, substrate, ambient (exit plane above the nozzle and outer radius),
         front / back (wedge), axis (empty)
```

- **Domain**: x from -20 mm to 160 mm, r from 0 to 18 mm
- **Throat at x = 0**, conical walls on both sides
- **Ambient patch**: the open boundary of the jet region, where the jet entrains surrounding gas
- **Substrate**: a flat no-slip wall normal to the axis

### System Size

| Component                        | Default mesh | Coarse test mesh |
| -------------------------------- | ------------ | ---------------- |
| Cells in total                   | 6,300        | 1,960            |
| Converging block (axial × radial)| 30 × 20      | 20 × 10          |
| Diverging block                  | 150 × 20     | 80 × 10          |
| Jet core block                   | 60 × 20      | 40 × 10          |
| Outer jet block                  | 60 × 25      | 40 × 14          |
| Smallest cell                    | 35 µm        | 69 µm            |
| Time step (measured)             | 4.7e-9 s     | 9.1e-9 s         |

`checkMesh` of the original 12,800-cell mesh: max aspect ratio 19.2, max non-orthogonality 8.5°, max skewness 0.37, **Mesh OK**.

---

## Boundary Conditions

| Patch          | p                         | T                           | U                              | k / ω / ν_t             |
| -------------- | ------------------------- | --------------------------- | ------------------------------ | ----------------------- |
| `inlet`        | `totalPressure` p₀        | `totalTemperature` T₀       | `pressureInletOutletVelocity`  | 5 % intensity, mixing length |
| `nozzleWall`   | `zeroGradient`            | `zeroGradient` (adiabatic)  | `noSlip`                       | wall functions          |
| `substrate`    | `zeroGradient`            | `zeroGradient` (adiabatic)  | `noSlip`                       | wall functions          |
| `ambient`      | `totalPressure` p_amb     | `inletOutlet` T_amb         | `pressureInletOutletVelocity`  | `inletOutlet`           |
| `front`, `back`| `wedge`                   | `wedge`                     | `wedge`                        | `wedge`                 |
| `axis`         | `empty`                   | `empty`                     | `empty`                        | `empty`                 |

---

## Analysis Types

| | Quick (quasi-1D) | Full CFD (OpenFOAM) |
| --- | --- | --- |
| Gas | Isentropic nozzle flow with shock detection, empirical free jet, normal-shock layer at the substrate | 2D axisymmetric compressible RANS (k-ω SST), Kurganov–Tadmor central scheme |
| Particles | 1D along the axis | 2D (x, r), several radial start points, wall bounces |
| Run time | under 1 s | minutes to hours, scales with the number of cores |
| Use it for | parametric sweeps, nozzle sizing, first estimates | final designs, shock structure, publication figures |

---

## App Modules

| Module                | What it does                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| Part                  | Nozzle diameters and lengths, standoff distance; live section view, expansion ratio, design exit Mach |
| Material              | Process gas, powder material, Rosin–Rammler size distribution, powder injection                  |
| Mesh                  | blockMesh seeds and grading, mesh preview, cell count, smallest cell, estimated time step and run time; `blockMesh` + `checkMesh` button |
| Boundary conditions   | Gun pressure and temperature, ambient state; patch table; instant quasi-1D prediction of the nozzle regime |
| Step                  | Drag and critical velocity models, OpenFOAM version and bashrc, turbulence, end time, Courant number, cores |
| Job                   | Submit quick or full jobs; job manager; live residuals, Δt and Courant plots, solver log; kill, re-run post-processing, delete; write input only (.zip) |
| Visualization         | Mirrored contour plots with particle tracks, start-up and particle-flight animations, XY data, particle impact table, report and exports |
| Compare               | Overlays of several jobs and a table of key results                                                 |

---

## Simulation Phases (Full CFD Job)

| Phase            | Tool                     | Description                                                               |
| ---------------- | ------------------------ | ------------------------------------------------------------------------- |
| Mesh             | `blockMesh`              | Four-block 5° wedge mesh                                                  |
| Check            | `checkMesh`              | Mesh quality report                                                       |
| Initialise       | `setFields`              | Converging section set to p₀, T₀; rest at ambient                         |
| Decompose        | `decomposePar`           | Parallel runs only, `simple` decomposition along the axis                 |
| Solve            | `rhoCentralFoam`         | Transient run to the end time, adjustable time step                       |
| Reconstruct      | `reconstructPar`         | Parallel runs only                                                        |
| Post-process     | `csflow_foam.py`         | Reads mesh and fields (no OpenFOAM tools), resamples to an (x, r) grid, builds animation frames, tracks particles, computes deposition efficiency |

The coarse test case (1,960 cells, 0.6 ms) took **6 min 43 s on 2 MPI ranks**, including post-processing.

---

## Expected Results

The results below come from runs with the default process conditions.

### Gas Flow

| Quantity                          | Quick (quasi-1D)                 | Full CFD (coarse mesh, 0.6 ms) |
| --------------------------------- | -------------------------------- | ------------------------------ |
| Nozzle regime                     | Over-expanded                    | Over-expanded (shock diamonds) |
| Exit Mach number (axis)           | 3.81                             | 3.43                           |
| Exit gas velocity                 | 1020 m/s                         | 998 m/s                        |
| Exit gas temperature              | -100 °C                          | -70 °C                         |
| Exit static pressure              | 25.7 kPa                         | 46.1 kPa                       |
| Gas mass flow                     | 14.44 g/s                        | 14.12 g/s (discharge coefficient 0.98) |
| Fully expanded jet Mach number    | 2.86                             | -                              |
| Bow shock at the substrate        | yes                              | yes                            |

The exit Mach number from CFD is lower than the isentropic value because the wall boundary layer reduces the effective expansion ratio.

### Particle Impact (Full CFD)

| Particle d (µm) | Impact velocity (m/s) | Critical velocity (m/s) | Outcome |
| --------------- | --------------------- | ----------------------- | ------- |
| 6.7             | 730                   | 475                     | Deposit |
| 16.7            | 620                   | 441                     | Deposit |
| 26.7            | 563                   | 432                     | Deposit |
| 36.7            | 523                   | 433                     | Deposit |

| Quantity                       | Quick        | Full CFD     |
| ------------------------------ | ------------ | ------------ |
| Mean impact velocity           | 586 m/s      | 580 m/s      |
| Mean impact temperature        | 239 °C       | 239 °C       |
| Deposition efficiency          | 100 %        | 100 %        |

### Other Operating Points (Quick Mode)

| Case                         | Regime                         | Exit velocity | Mean impact velocity | DE    |
| ---------------------------- | ------------------------------ | ------------- | -------------------- | ----- |
| N₂, p₀ = 0.5 MPa             | Normal shock at x = 102.7 mm   | 165 m/s       | 331 m/s              | 2 %   |
| He, p₀ = 3 MPa               | Over-expanded, exit Mach 4.84  | 2489 m/s      | 941 m/s              | 36 % (fine particles erode) |
| Air, p₀ = 2 MPa, 300 °C, Al  | Over-expanded                  | 925 m/s       | 604 m/s              | 100 % |

---

## Governing Equations

### Isentropic Nozzle Flow

```
T / T0 = 1 / (1 + (γ - 1)/2 M²)
p / p0 = (T / T0)^(γ / (γ - 1))
A / A* = (1/M) [ (2/(γ + 1)) (1 + (γ - 1)/2 M²) ]^((γ + 1) / (2(γ - 1)))
u      = M sqrt(γ R T)
ṁ      = p0 A* sqrt(γ / (R T0)) ((γ + 1)/2)^(-(γ + 1) / (2(γ - 1)))       choked nozzle
```

### Normal Shock

```
M2²      = (1 + (γ - 1)/2 M1²) / (γ M1² - (γ - 1)/2)
p2 / p1  = 1 + 2γ/(γ + 1) (M1² - 1)
T2 / T1  = (p2/p1) (2 + (γ - 1) M1²) / ((γ + 1) M1²)
p02/p01  = [ (γ+1) M1² / ((γ-1) M1² + 2) ]^(γ/(γ-1)) [ (γ+1) / (2γ M1² - (γ-1)) ]^(1/(γ-1))
```

### Particle Motion

```
m_p dv/dt = ½ ρ_g C_D A_p |u - v| (u - v)            A_p = π d²/4,  m_p = ρ_p π d³/6
Re_p      = ρ_g |u - v| d / μ_g
M_rel     = |u - v| / sqrt(γ R T_g)
C_D       = C_D(Re_p, M_rel, γ, T_p/T_g)             Henderson (1976)
```

Henderson's law uses a subsonic form for M_rel ≤ 1, a supersonic form for M_rel ≥ 1.75 and linear interpolation between them.

### Particle Heating

```
m_p c_p dT_p/dt = h π d² (T_aw - T_p)
Nu    = h d / k_g = 2 + 0.6 Re_p^(1/2) Pr^(1/3)       Ranz-Marshall
T_aw  = T_g + Pr^(1/3) |u - v|² / (2 c_p,g)           adiabatic recovery temperature
```

### Critical and Erosion Velocity

```
v_crit = sqrt( F1 · 4 σ_UTS (1 - (T_i - T_R)/(T_m - T_R)) / ρ_p  +  F2 c_p (T_m - T_i) )
         F1 = 1.2, F2 = 0.3, T_R = 293 K                                  Schmidt et al. (2006)

v_crit = 667 - 14 ρ + 0.08 T_m + 0.1 σ_UTS - 0.4 T_i
         ρ in g/cm³, T in °C, σ_UTS in MPa                                Assadi et al. (2003)

v_ero  = 2 v_crit
deposit when v_crit ≤ v_impact ≤ v_ero
```

### Size Distribution and Deposition Efficiency

```
F(d)  = 1 - exp( -(d / d̄)^n ),   d̄ = d50 / (ln 2)^(1/n)                 Rosin-Rammler (mass)
DE    = Σ w_i   over the size classes (and start radii) that deposit
```

### Time Step of the Wedge Mesh

```
Δt ≈ Co · min( Δx / (|u| + c),  2 sin(2.5°) · 0.9 Δr_axis / c* )
```

`rhoCentralFoam` counts the wedge front and back faces in its Courant number, so the thin cell on the axis at the throat (Δr_axis wide, only 2 Δr_axis sin 2.5° thick) usually sets the time step. The Mesh module uses this estimate; it matched the measured time step to within 5 %.

---

## Output Files

Each job is stored in `runs/<job name>/`.

| File                    | Content                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| `model.json`            | Every input of the model (also the format of a saved model)                |
| `job.json`              | Analysis type and creation time                                            |
| `status.json`           | State (running, done, failed, killed), stage, start and finish times       |
| `results.pkl`           | Gas field, centreline data, animation frames, particle trajectories, impact table |
| `job.log`               | Output of the job runner                                                   |
| `log.blockMesh`, `log.checkMesh`, `log.setFields`, `log.decomposePar`, `log.solver`, `log.reconstructPar` | OpenFOAM logs (full CFD) |
| `system/`, `constant/`, `0.orig/`, `Allrun` | The OpenFOAM case (full CFD)                         |
| `case.foam`             | Open in ParaView                                                           |

### Exports from the Visualization Module

| Export                        | Content                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `<job>_centreline.csv`        | x, Mach, u, T, p, ρ along the axis                                     |
| `<job>_particles.csv`         | Impact velocity, temperature, critical and erosion velocity, outcome per size class |
| `<job>_to_CSIM.json`          | Diameter, radius, impact velocity and impact temperature per size class, for CSIM |
| `<job>_case.zip`              | The OpenFOAM case with results                                          |
| `csflow_comparison.csv`       | Key results of the compared jobs                                        |

---

## Repository Structure

```
csflow/
|
|-- app.py                  # Streamlit web interface (model tree, job manager, visualization)
|-- csflow_physics.py       # gas and powder databases, nozzle geometry, Henderson drag,
|                           # Ranz-Marshall heating, critical velocity, Rosin-Rammler
|-- csflow_quick.py         # quick mode: quasi-1D nozzle, free jet, bow shock, particle tracking
|-- csflow_foam.py          # full mode: OpenFOAM case writer, job runner, log parser,
|                           # results reader, 2D particle tracking
|-- csflow_model.py         # default model, save and load
|-- requirements.txt        # Python packages
|-- README.md               # this file
|
|-- images/
|   |-- csflow_cfd_result.png
|
|-- generated on run/
    |-- runs/<job name>/model.json
    |-- runs/<job name>/results.pkl
    |-- runs/<job name>/status.json
    |-- runs/<job name>/log.*            (full CFD)
    |-- runs/<job name>/system, constant, 0.orig, Allrun, case.foam   (full CFD)
```

---

## How to Run

### Requirements

- Python 3.10 or newer with streamlit (1.48 or newer), plotly, numpy, scipy and pandas
- For full CFD: OpenFOAM, the ESI version (openfoam.com) is the tested one: <https://www.openfoam.com>
- An MPI runtime (comes with OpenFOAM) for parallel runs
- Optional: ParaView for further post-processing of `case.foam`
- On Windows: WSL with Ubuntu

Check that OpenFOAM is available:

```
source /usr/lib/openfoam/openfoam2506/etc/bashrc
which rhoCentralFoam blockMesh
```

### Step 1: Install the Python packages

```
python3 -m pip install -r requirements.txt
```

or with conda:

```
conda create -n csflow -c conda-forge python=3.12 streamlit plotly numpy scipy pandas -y
conda activate csflow
```

### Step 2: Start the app

With the files on `D:` in WSL:

```
cd "/mnt/d/cold spray/csflow"
source /usr/lib/openfoam/openfoam2506/etc/bashrc      # optional, for full CFD
streamlit run app.py --server.headless true
```

Open <http://localhost:8501> in the Windows browser. If OpenFOAM was not sourced, paste the path of its `bashrc` into **Step → OpenFOAM bashrc**. The sidebar shows `OpenFOAM: found` when the app can run full CFD jobs.

### Step 3: Build the model

Go through **Part, Material, Mesh, Boundary conditions and Step**. The Boundary conditions module shows the quasi-1D prediction (nozzle regime, exit Mach number, exit velocity, mass flow) at once, and the Mesh module shows the estimated time step and run time. Save the model with **Save model (.json)**.

### Step 4: Run a job

In **Job**, give the job a name, choose **Quick** or **Full CFD (OpenFOAM)** and press **Submit**. Quick jobs finish in under a second. Full jobs run in the background: the monitor shows the progress, the initial residuals, the time step, the Courant number and the tail of the solver log. **Write input only** gives the OpenFOAM case as a .zip, to run elsewhere with:

```
./Allrun
python csflow_foam.py post <case folder>
```

### Step 5: Look at the results

Open **Visualization** and pick the job. Use **Compare** to overlay several jobs, for example different gun pressures, gases or nozzles. Export the impact conditions with **Send to CSIM (.json)**.

---

## Important: Time Step and Run Time of the Wedge Mesh

| Setting                                         | Effect                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------- |
| Many radial cells in the throat                 | Thinner axis cell at the throat, smaller time step: the run time grows with roughly the square of the radial refinement |
| Strong wall grading (small `wall_grading`)      | Resolves the boundary layer, but the axis cells stay thin as well               |
| Default mesh (6,300 cells, 1 ms)                | About 90 core-minutes                                                           |
| Coarse mesh (1,960 cells, 0.6 ms)               | About 7 minutes on 2 cores                                                      |

`rhoCentralFoam` includes the sound speed on the wedge faces in its Courant number, so the time step is set by the first cell on the axis at the throat, which is only 2 Δr sin 2.5° thick in the circumferential direction. Check the **Estimated run time** in the Mesh module before you refine, and use several cores (Step → CPU cores).

Run long enough for a steady jet: the gas needs about 0.2 ms to cross the nozzle, and 0.8 to 1.5 ms usually gives a steady jet against the substrate. In the start-up animation the centreline Mach number should stop changing over the last frames.

---

## What to Look for in Results

### Nozzle Regime

In an over-expanded nozzle the exit pressure is below ambient and the jet forms oblique shocks and **shock diamonds** after the exit. If the gun pressure is too low, a **normal shock** stands inside the diverging section and the gas leaves the nozzle subsonic, which cuts the particle velocity sharply.

### Gas Cooling

The gas expands and cools far below the gun temperature: about -70 °C at the exit for 400 °C in the chamber. Particles still arrive warmer than the exit gas because they heat up in the hot converging section and lag behind the gas.

### Bow Shock

In front of the substrate the jet passes a bow shock and slows down to stagnation. Small particles follow the gas closely and lose speed here; large particles cross the layer almost unaffected.

### Particle Size Effect

Small particles accelerate fastest and reach the highest impact velocity, large particles lag behind the gas. The impact velocity curve should lie between the critical velocity and the erosion velocity for good deposition.

### Deposition Efficiency

A deposition efficiency below 100 % means that some sizes fall below the critical velocity (rebound) or exceed the erosion velocity (erosion). Raising the gun pressure or temperature, or switching to helium, moves the impact velocity curve up.

---

## Common Errors and Fixes

| Error or symptom                                             | Cause                                                       | Fix                                                                              |
| ------------------------------------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Sidebar shows `OpenFOAM: not found (quick mode only)`        | OpenFOAM not sourced before starting the app                | Source its `bashrc` before `streamlit run`, or set the path in Step              |
| `cd: D:\...: No such file or directory`                      | Windows path typed in a Linux (WSL) terminal                | Use `/mnt/d/...`                                                                 |
| `gio: http://localhost:8501: Operation not supported`        | WSL cannot open a browser                                   | Harmless; open <http://localhost:8501> in Windows, or start with `--server.headless true` |
| `Attempted to use <scotch> without the scotchDecomp library` | OpenFOAM built without scotch                               | Not used by CSFLOW, which writes `method simple`; regenerate the case            |
| `mpirun has detected an attempt to run as root`              | Running as root (containers, servers)                       | Run as a normal user, or set `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` |
| Job failed during `mesh`                                     | Invalid geometry (throat not smaller than inlet and exit)   | Check the Part module; see `log.blockMesh`                                       |
| Job failed during `solve`                                    | Solver diverged or wrong OpenFOAM version selected          | See `log.solver`; lower the Courant number, check Step → OpenFOAM version        |
| Very slow full CFD run                                       | Thin axis cells at the throat (see above)                   | Fewer radial cells, more cores, or quick mode for sweeps                         |
| `StreamlitAPIException` about `width`                        | Streamlit older than 1.48                                   | `pip install -U streamlit`                                                       |
| `ModuleNotFoundError: No module named 'plotly'`              | Python without the app packages                             | `pip install -r requirements.txt`                                                |
| `Cannot evaluate this operating point`                       | Gun pressure not above ambient pressure                     | Raise `p0`                                                                       |
| Visualization lists no jobs                                  | No job has finished yet                                     | Wait for the job, or press **Re-run post-processing** in Job                     |
| Animation shows only a few frames                            | Write interval large compared with the end time             | Lower the write interval in Step                                                 |

---

## Extending the Model

| Extension                            | What to change                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| New powder or gas                    | Add an entry to `MATERIALS` or `GASES` in `csflow_physics.py`, or edit the properties in the app     |
| Curved (bell or MOC) nozzle contour  | Replace `Nozzle.radius` and add arc or spline edges to the `blockMeshDict` writer                    |
| Radial (downstream) powder injection | Inject at a position in the diverging section in Material; add a radial injection velocity          |
| Two-way coupling                     | Add an OpenFOAM cloud (`basicKinematicCloud` or thermo cloud) to the solver for high powder feed rates |
| Finer boundary layer                 | Lower `wall_grading` and raise the radial cells; check the estimated run time                       |
| Inclined spray angle                 | Needs a 3D mesh (snappyHexMesh) instead of the wedge                                                 |
| Size-dependent critical velocity     | Add a size term to `v_crit_schmidt`, or calibrate F1 against CSIM impact simulations                |
| Process chain with CSIM              | Load `<job>_to_CSIM.json` in CSIM and simulate the impact of each size class                         |

---

## Citation

If you use this code in your research, please cite:

```
@software{mishra_2026_csflow,
  author    = {Mishra, Akshansh},
  title     = {CSFLOW: Cold Spray FLOW Simulator, a Browser-Based CFD Environment
               for Cold Spray Nozzles and Particle Acceleration},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24224141},
  url       = {https://doi.org/10.5281/zenodo.24224141}
}
```

Plain text citation:

> Mishra, A. (2026). *CSFLOW: Cold Spray FLOW Simulator, a Browser-Based CFD Environment for Cold Spray Nozzles and Particle Acceleration* [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.24224141>

### References

- Henderson, C. B. (1976). Drag coefficients of spheres in continuum and rarefied flows. *AIAA Journal*, 14(6), 707–708.
- Schmidt, T., Gärtner, F., Assadi, H., & Kreye, H. (2006). Development of a generalized parameter window for cold spray deposition. *Acta Materialia*, 54(3), 729–742.
- Assadi, H., Gärtner, F., Stoltenhoff, T., & Kreye, H. (2003). Bonding mechanism in cold gas spraying. *Acta Materialia*, 51(15), 4379–4394.
- Ranz, W. E., & Marshall, W. R. (1952). Evaporation from drops. *Chemical Engineering Progress*, 48(3), 141–146.

---

## Author

**Akshansh Mishra**
GitHub: <https://github.com/akshansh11>

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

You are free to:

- **Share**: copy and redistribute the material in any medium or format
- **Adapt**: remix, transform, and build upon the material

Under the following terms:

- **Attribution**: You must give appropriate credit to Akshansh Mishra and provide a link to this repository
- **NonCommercial**: You may not use the material for commercial purposes

Copyright 2026 Akshansh Mishra. All rights reserved for commercial use.
