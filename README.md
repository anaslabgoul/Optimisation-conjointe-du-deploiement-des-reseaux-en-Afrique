# Joint Optimization of Mobile Network Deployment in Africa

> 🇫🇷 *Version française : [README.fr.md](README.fr.md)*

> Next-generation (5G/NG) deployment & energy strategy of a telecom operator
> in a competitive environment.
>
> **PSC — Collective Scientific Project — École Polytechnique × Orange**

This repository gathers all the mathematical models, heuristics and
artificial-intelligence approaches developed to determine the **best response**
of an operator (here **Orange**) to its competitors' deployment strategies, under
constraints of **budget**, **capacity**, **regulatory coverage** and — in the
extension — **energy** and **carbon footprint**.

The full report (in French) is available in [`rapport/Optimisation_conjointe.pdf`](rapport/Optimisation_conjointe.pdf).

---

## 1. Context and problem

Mobile traffic is exploding, driven by the proliferation of devices and
bandwidth-hungry usage. In Africa, modernizing the network (deploying 5G) is a
strategic lever to capture market share, but it faces limited annual budgets,
regulatory coverage requirements, and strong infrastructure constraints
(reliable energy).

We consider a competitive market:

| Element | Notation | Description |
|---|---|---|
| Operators | `I = {ORANGE, FREE MOBILE, BOUYGUES TELECOM, SFR}` | Orange = target operator `τ` |
| Legacy technologies | `G = {2G, 3G, 4G}` | 2G is kept (minimal service) |
| New generation | `NG` (5G) | Technology to deploy |
| Geographic areas | `A` | Each area `a` has a potential population `uₐ` |
| Sites | `Sτ` | Sites of the target operator |
| Horizon | `T = {0, 1, …, |T|}` | Planning periods |

**Main decision:** the binary variables `zₛᵗ = 1` if NG is installed on site `s`
at period `t`. Competitors' plans `Rₐ,ᵢᵗ` are assumed **known** (best response).
The **objective** is to maximize Orange's NG market share at the end of the horizon.

---

## 2. Methodological approach

The project explores a **hierarchy of solution methods**, from the exact model to
learning-based approaches, each addressing the trade-off
*solution quality ↔ computation time*.

```
Initial MINLP ──► Initial MILP ──► Reformulated MILP        (exact models)
                                        │
                                        ├──► Fix-and-Relax
                                        ├──► Genetic Algorithm (solver-based / direct)
                                        ├──► Memetic Algorithm
                                        ├──► MILP-GNN (warm-start via graph neural network)
                                        └──► Reinforcement Learning (PPO)

Extension: Joint deployment–legacy-retirement optimization (energy & CO₂)
```

### Summary of results

| Method | Quality (% of optimum) | Time gain | Note |
|---|---|---|---|
| **Reformulated MILP** | 100 % (exact) | −63 % vs initial MILP | −76 % constraints, −17 % variables |
| **Fix-and-Relax** | — | ineffective | Breaks the *presolve* deductions, > 30 min |
| **Direct GA** (structured greedy) | ≈ 81.6 % | −87.7 % (550 areas) | Polynomial complexity `O(GSA)` |
| **Memetic algorithm** | ≈ 97.8 % | −83 % vs solver | *First-improvement* local search |
| **MILP-GNN** (ratio 0.5) | ≈ 97.5 % | −71.6 % | ratio 0.8 → −91.7 % ; ratio 0.95 → infeasible |
| **RL — PPO** | 65.3 % market share* | inference in ~ms | *≈ 2× greedy (34.6 %), but violates QA in 40 % of steps |

> Reference instance: **Mayenne** (723 sites, 616 areas), provided by Orange.

---

## 3. Repository structure

The repository is organized to follow the report's progression. Datasets are
grouped only once in `data/` (Mayenne instance) and `Petite instance d'essai/`
(toy instance) — see the execution note in § 6.

```
.
├── rapport/                          # Final report (PDF, French)
│
├── data/                             # Canonical dataset: MAYENNE instance (723 sites, 616 areas)
├── Petite instance d'essai/          # Small toy instance (manual model validation)
│
├── 0_exploration_donnees/            # Exploration / visualization of the CSV files (reads ../data)
│
├── 1_modeles_milp/                   # § Mathematical models
│   ├── MILP initial.ipynb            #   Big-M linearization of the MINLP
│   └── MILP reformulé.ipynb          #   Reformulation (−76 % constraints)
│
├── 2_heuristiques/                   # § Heuristics & metaheuristics
│   ├── Fix&Relax - AG.ipynb          #   Fix-and-Relax + solver-based genetic algorithm
│   ├── Algorithmes_heuristiques_Work.ipynb  # Direct GA + memetic (solver-free evaluation)
│   └── Heuristic_algorithms.ipynb    #   First heuristics version
│
├── 3_milp_gnn/                       # § MILP-GNN
│   └── MILP GNN.ipynb                #   GraphSAGE to predict which variables to fix (warm-start)
│
├── 4_apprentissage_renforcement/     # § Reinforcement learning (PPO)
│   ├── ng_deployment_env.py          #   Gymnasium environment (deployment MDP)
│   ├── train_ppo.py                  #   PPO agent training
│   └── RL_evaluate_ng_deployment.py  #   Evaluation vs baselines (Greedy, Random)
│
├── 5_instances_aleatoires/           # Generation & tests on random instances (1st / 2nd model)
│
├── prototypes/                       # Exploratory work & earlier versions
│   ├── modeles_initiaux/             #   First / Second model (+ extended data)
│   ├── tests_pyomo/                  #   Pyomo prototypes (incl. ANFR data)
│   └── tests_petite_instance/        #   Rectification tests on the small instance
│
└── utils/                            # Data-loading utilities
    ├── importation_des_données.py    #   (run from the repository root)
    └── test-petite-instance-donnees.py
```

---

## 4. Models and methods

### 4.1 MILP models (`1_modeles_milp/`)
The problem is first posed as a **MINLP** (mixed-integer nonlinear program). Two
nonlinearities are handled:
- the customer **migration constraint** (binary × continuous product) is
  linearized with the **Big-M** technique using tightened bounds;
- the **indicator-decoding constraint** `δₐ,Cᵗ` (combinatorial explosion in
  `2^|I|`) is **eliminated** in the *reformulated MILP* by exploiting the fact that
  competitor coverage `Rₐᵗ` is a fixed parameter.

Modeling additions: time dependence of demand `Dᴺᴳ` and capacity `CAPAᴺᴳ`,
**monotonicity** of deployment (`zₛᵗ ≥ zₛᵗ⁻¹`), relaxation of the integrality of
population variables.

→ **−76 % constraints** (1,653,574 → 399,580) and **−63 % time** on 500-area
instances, at a nearly identical objective value (**Gurobi** solver).

### 4.2 Heuristics & metaheuristics (`2_heuristiques/`)
- **Fix-and-Relax** — sliding window with integer present / relaxed future /
  fixed past. Ineffective here: relaxing the future breaks the monotonicity
  constraint and thus the cutting power of the *presolve*.
- **Genetic algorithm** — two versions:
  1. *solver-based* (each individual = an assignment of `z`, fitness = the MILP
     subproblem): faithful but intractable;
  2. *direct*: encoding by **first-deployment dates**, **solver-free analytical
     evaluation** (`z ⇒ r ⇒ u`), trajectory precomputation, memoization, and a
     **structured greedy initialization** (swap / time-shift / jitter).
- **Memetic algorithm** — direct GA + **first-improvement local search** and
  **structured mutations** (`swap_periods`, `advance_best`, `delay_worst`). Best
  quality/time trade-off (≈ 97.8 % of the optimum).

### 4.3 MILP-GNN (`3_milp_gnn/`)
A **site-graph** representation (one node per site, an edge if two sites cover a
common area; features = degree + client potential). A 3-layer **GraphSAGE**
network predicts each site's deployment time. A **ratio** of the most confident
`zₛᵗ` variables is fixed to give the solver a **warm start**. The
precision ↔ time trade-off is driven by the ratio (0.5 → excellent ;
0.95 → risk of infeasibility).

### 4.4 Reinforcement learning (`4_apprentissage_renforcement/`)
Deployment is formulated as a **finite-horizon MDP** and solved with **PPO**
(*Proximal Policy Optimization*). The agent observes the full state (deployment,
coverage, subscriber distribution, competitor coverage, budget, regulatory gap)
and outputs a score per site; the `Zᵗ` top-scoring sites are deployed. Once
trained, it decides in a few milliseconds. Limitation: the reward penalizes but
does not guarantee the constraints (QA violated in 40 % of steps), and the MLP
architecture requires a fixed instance size.

### 4.5 Energy extension (report, § 7)
A **joint deployment–retirement optimization** model `(PE)`: a second decision
lever allows **retiring** the legacy layers (3G/4G) to transfer their energy
envelope to NG, under an **energy budget** `Eₜᵐᵃˣ` and a **carbon cap**
`CO₂,ₜᵐᵃˣ`, with redistribution of evicted customers. Formulated in the report
(implementation as future work).

---

## 5. Data (Mayenne instance)

The data comes from the **Mayenne** instance (723 sites, 616 areas) provided by
the Orange tutors. Mapping of files ↔ model parameters:

| Source file | Parameter(s) | Description |
|---|---|---|
| `AREAS.csv` | `u⁰ₐ,ᵢ,ₒ`, `uₐ` | Initial population per area, operator and offer |
| `EXISTING_SITES.csv` | `Sτ` | Target operator's sites (+ 3G/4G/5G status) |
| `AREAS_SITES_LINK.csv` | `Sₐ,τ`, `Aₛ` | Area ↔ site link (geographic coverage) |
| `COMPETITORS_STRATEGY.csv` | `Rₐ,ᵢᵗ` | Competitors' NG coverage per area & period |
| `DEMAND.csv` | `Dᴺᴳᵗ` | 5G traffic demand per period |
| `CAPACITY.csv` | `CAPAᴺᴳᵗ` | NG site capacity per period |
| `OPERATIONAL_LIMITS.csv` | `Z̄ᵗ` | Deployment budget per period |
| `STRATEGIC_GUIDELINES.csv` | `QAᵗ` | Regulatory minimum-coverage target |
| `UPGRADE_FUNCTION.csv` | `fₐ,C,o′,o` | Migration function between offers given the coverage context |

A **small toy instance** (`Petite instance d'essai/`) is used to manually validate
the models.

---

## 6. Getting started

### Main dependencies
The notebooks and scripts rely on the scientific Python ecosystem:

- `pandas`, `numpy` — data handling
- `pyomo` — MILP modeling, with a solver (**Gurobi**, or open-source
  **HiGHS** / **GLPK**)
- `torch`, `torch-geometric` — GNN (GraphSAGE)
- `gymnasium`, `stable-baselines3` — reinforcement learning (PPO)
- `matplotlib` — visualizations

### Running a model / a heuristic
The notebooks read their CSVs **by simple filename** (e.g. `AREAS.csv`), i.e. from
the **current working directory**. Since the Mayenne dataset is stored only once
(in `data/`), copy it next to the notebook before running it — for example:

```bash
cp data/*.csv 1_modeles_milp/     # or 2_heuristiques, 3_milp_gnn, 5_instances_aleatoires
cd 1_modeles_milp
jupyter notebook                  # open then run the desired notebook
```

For read-only browsing (the notebooks keep their saved outputs), no copy is
needed. The small-instance tests use the files in `Petite instance d'essai/` the
same way.

### Running reinforcement learning
The RL scripts take the dataset as an argument (`--data_dir`):

```bash
cd 4_apprentissage_renforcement
python train_ppo.py                              # training (adjust data_dir in the script)
python RL_evaluate_ng_deployment.py --data_dir ../data --episodes 10
```

### Loading utility
`utils/importation_des_données.py` loads the small instance; run it **from the
repository root** (paths are relative to `Petite instance d'essai/`).

---

## 7. Team

**Authors** — Labgoul Anas · Oujaa Haitam Yassine · Takfa Anass ·
Belfatmi Ayoub · Ait Mansour Abderrahmane

**Tutors (Orange / École Polytechnique)** — Matthieu Chardy · Amal Benhamiche ·
Youssouf Hadhbi · Aurélien Bechler

---

## 8. Main references

1. A. Cambier, M. Chardy, R. Figueiredo, A. Ouorou, M. Poss — *Optimizing the
   investments in mobile networks and subscriber migrations for a telecommunication
   operator*, Networks, 77(4):495–519, 2021.
2. M. Chardy, M. Ben Yahia, Y. Bao — *3G/4G load-balancing optimization for mobile
   network planning*, 2016.
3. A. Benhamiche, M. Chardy, B. Mebrek — *Modelling the mobile investment strategies
   under competition using mathematical programming*.
4. W. Hamilton, R. Ying, J. Leskovec — *Inductive Representation Learning on Large
   Graphs (GraphSAGE)*, 2017.
5. M. Gasse, D. Chételat, N. Ferroni, L. Charlin, A. Lodi — *Exact Combinatorial
   Optimization with Graph Convolutional Neural Networks*, 2019.
6. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov — *Proximal Policy
   Optimization Algorithms*, 2017.
7. P. Zappalà — *Méthodes de résolution des jeux en forme extensive avec application
   au marché des réseaux mobiles*, PhD thesis, Avignon Université, 2024.

> The complete bibliography is in the report.
