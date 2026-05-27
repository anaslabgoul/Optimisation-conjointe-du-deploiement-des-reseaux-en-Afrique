from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Constants — operator / offer indices

OPERATORS = [ "ORANGE","FREE MOBILE", "BOUYGUES TELECOM", "SFR"]
OFFERS    = ["o3G", "o4G", "o5G"]
TARGET_OP = "ORANGE"
NG_OFFER  = "o5G"
COMPETITORS = [op for op in OPERATORS if op != TARGET_OP]

OP_IDX  = {op: i for i, op in enumerate(OPERATORS)}
OFF_IDX = {o: i  for i, o  in enumerate(OFFERS)}

N_OPS   = len(OPERATORS)      # 4
N_OFFS  = len(OFFERS)         # 3
N_COMPS = len(COMPETITORS)    # 3

TARGET_IDX = OP_IDX[TARGET_OP]   # 0  (ORANGE)
NG_IDX     = OFF_IDX[NG_OFFER]   # 2  (o5G)

# Data Loader

class NGDeploymentData:

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._load_setup()
        self._load_sites()
        self._load_areas()
        self._load_area_site_link()
        self._load_operational_limits()
        self._load_strategic_guidelines()
        self._load_capacity()
        self._load_demand()
        self._load_competitors()
        self._load_migration()

    # ------------------------------------------------------------------
    def _csv(self, filename: str):
        path = self.data_dir / filename
        with open(path, newline="", encoding="utf-8-sig") as f:
            content = f.read()
        rows = list(csv.DictReader(content.splitlines(), delimiter=";"))
        return rows

    # ------------------------------------------------------------------
    def _load_setup(self):
        rows = self._csv("SETUP.csv")
        cfg  = {r["PARAMETERS"]: r["VALUES"] for r in rows}
        self.T = int(cfg["TIME_SLOTS"])       # total periods (includes t=0)

    # ------------------------------------------------------------------
    def _load_sites(self):
        """Load existing sites and their initial deployment state."""
        rows = self._csv("EXISTING_SITES.csv")
        self.sites = [r["EXISTING_SITES"] for r in rows]
        self.n_sites = len(self.sites)
        self.site_idx = {s: i for i, s in enumerate(self.sites)}

        # Initial 5G deployment: True if site already has 5G
        self.init_deployment = np.array(
            [int(r["5G"]) == 1 for r in rows],
            dtype=bool,
        )

    # ------------------------------------------------------------------
    def _load_areas(self):
        """Load areas and initial subscriber distribution."""
        rows = self._csv("AREAS.csv")
        self.areas   = [r["AREAS"] for r in rows]
        self.n_areas = len(self.areas)
        self.area_idx = {a: i for i, a in enumerate(self.areas)}

        # subscribers[a, op, offer] = initial count
        subs = np.zeros((self.n_areas, N_OPS, N_OFFS), dtype=np.float64)
        for ai, r in enumerate(rows):
            for offer in OFFERS:
                for op in OPERATORS:
                    col = f"{offer}-{op}"
                    subs[ai, OP_IDX[op], OFF_IDX[offer]] = float(r.get(col, 0.0))
        self.init_subscribers = subs

        # Total population per area = sum over all operators and offers
        self.area_population = subs.sum(axis=(1, 2))   # (n_areas,)

    # ------------------------------------------------------------------
    def _load_area_site_link(self):
        """Build binary connectivity matrix: site_covers_area[s, a]."""
        rows = self._csv("AREAS_SITES_LINK.csv")
        mat  = np.zeros((self.n_sites, self.n_areas), dtype=bool)
        for r in rows:
            site = r["SITES"]
            area = r["AREAS"]
            tech = r["TECHNOLOGIES"]
            if tech == "5G" and site in self.site_idx and area in self.area_idx:
                si = self.site_idx[site]
                ai = self.area_idx[area]
                mat[si, ai] = True
        self.site_covers_area = mat   # (n_sites, n_areas)

    # ------------------------------------------------------------------
    def _load_operational_limits(self):
        """Load budget Z^t per period."""
        rows = self._csv("OPERATIONAL_LIMITS.csv")
        budget = np.zeros(self.T, dtype=int)
        for r in rows:
            t = int(r["TIME_SLOTS"])
            if t < self.T:
                budget[t] = int(r["MAX_NUMBER_OF_DEPLOYMENTS"])
        self.budget_t = budget   # (T,) — budget_t[t] = max deployments at step t

    # ------------------------------------------------------------------
    def _load_strategic_guidelines(self):
        """Load regulatory coverage thresholds QA^t."""
        rows = self._csv("STRATEGIC_GUIDELINES.csv")
        qa = np.zeros(self.T, dtype=np.float64)
        for r in rows:
            t = int(r["TIME_SLOTS"])
            if t < self.T:
                qa[t] = float(r["QA"])
        self.qa_threshold_t = qa   # (T,) — QA at each period

    # ------------------------------------------------------------------
    def _load_capacity(self):
        """Load site capacity per technology per period."""
        rows = self._csv("CAPACITY.csv")
        cap  = np.zeros((self.T, N_OFFS), dtype=np.float64)
        for r in rows:
            t = int(r["TIME_SLOTS"])
            if t < self.T:
                for off in OFFERS:
                    tech = off.replace("o", "")   # o5G → 5G
                    cap[t, OFF_IDX[off]] = float(r.get(tech, 0.0))
        # Forward-fill missing periods
        for t in range(1, self.T):
            for j in range(N_OFFS):
                if cap[t, j] == 0:
                    cap[t, j] = cap[t - 1, j]
        self.capacity_t = cap   # (T, N_OFFS)

    # ------------------------------------------------------------------
    def _load_demand(self):
        """Load traffic demand per subscriber per offer per period."""
        rows = self._csv("DEMAND.csv")
        dem  = np.zeros((self.T, N_OFFS), dtype=np.float64)
        for r in rows:
            t = int(r["TIME_SLOTS"])
            if t < self.T:
                for off in OFFERS:
                    tech = off.replace("o", "")
                    dem[t, OFF_IDX[off]] = float(r.get(tech, 0.0))
        # Forward-fill
        for t in range(1, self.T):
            for j in range(N_OFFS):
                if dem[t, j] == 0:
                    dem[t, j] = dem[t - 1, j]
        self.demand_t = dem   # (T, N_OFFS)

    # ------------------------------------------------------------------
    def _load_competitors(self):
        """Load competitor 5G coverage per area per period."""
        rows = self._csv("COMPETITORS_STRATEGY.csv")
        comp_cov = np.zeros((self.T, self.n_areas, N_COMPS), dtype=bool)
        for r in rows:
            t    = int(r["TIME_SLOTS"])
            area = r["AREAS"]
            if t < self.T and area in self.area_idx:
                ai = self.area_idx[area]
                for ci, comp in enumerate(COMPETITORS):
                    val = r.get(comp, "False")
                    comp_cov[t, ai, ci] = val.strip().lower() == "true"
        self.competitor_cov = comp_cov   # (T, n_areas, N_COMPS)

    # ------------------------------------------------------------------
    def _load_migration(self):
        """
        Build migration lookup table from UPGRADE_FUNCTION.csv.

        Key  : (cov_tuple: tuple[int,int,int,int], offer: str, from_op: str)
               where cov_tuple = (FM_has5G, BT_has5G, SFR_has5G, OR_has5G)
        Value: dict[to_op: str → percentage: float]
        """
        rows = self._csv("UPGRADE_FUNCTION.csv")
        table: dict[tuple, dict[str, float]] = {}
        for r in rows:
            cov   = (int(r["FREE MOBILE"]), int(r["BOUYGUES TELECOM"]),
                     int(r["SFR"]),         int(r["ORANGE"]))
            offer   = r["OFFERS"]
            from_op = r["FROM_OPERATOR"]
            to_op   = r["TO_OPERATOR"]
            pct     = float(r["PERCENTAGES"])
            key = (cov, offer, from_op)
            if key not in table:
                table[key] = {}
            table[key][to_op] = pct
        self.migration_table = table


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class NGDeploymentEnv(gym.Env):
    
    
    metadata = {"render_modes": ["human", "ansi"]}

    # Reward weights (can be overridden at init)
    DEFAULT_ALPHA   = 1.0    # ΔMS weight
    DEFAULT_BETA    = 0.3    # ΔCov weight
    DEFAULT_LAMBDA_REG = 20.0
    DEFAULT_LAMBDA_CAP = 10.0
    DEFAULT_ETA     = 2.0    # terminal bonus weight
    DEFAULT_GAMMA   = 0.99

    def __init__(
        self,
        data_dir: str | Path,
        alpha: float   = DEFAULT_ALPHA,
        beta: float    = DEFAULT_BETA,
        lambda_reg: float = DEFAULT_LAMBDA_REG,
        lambda_cap: float = DEFAULT_LAMBDA_CAP,
        eta: float     = DEFAULT_ETA,
        gamma: float   = DEFAULT_GAMMA,
        noise_std: float = 0.0,          # σ of Gaussian migration noise (0 = deterministic)
        render_mode: str | None = None,
    ):
        super().__init__()
        self.data = NGDeploymentData(data_dir)
        self.alpha      = alpha
        self.beta       = beta
        self.lambda_reg = lambda_reg
        self.lambda_cap = lambda_cap
        self.eta        = eta
        self.gamma      = gamma
        self.noise_std  = noise_std
        self.render_mode = render_mode

        d = self.data   # shorthand

        # Derived sizes
        self.n_sites = d.n_sites
        self.n_areas = d.n_areas
        self.T       = d.T                     # total time slots
        self.Z_max   = max(d.budget_t)     # for normalisation

        # Observation vector length
        self._obs_len = (
            self.n_sites                       # deployment
            + self.n_areas                     # ORANGE coverage
            + self.n_areas * N_OPS * N_OFFS    # subscribers (normalised)
            + self.n_areas * N_COMPS           # competitor coverage
            + 3                                # τ, b, δ
        )

        # ---- Spaces -------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._obs_len,),
            dtype=np.float32,
        )
        # Scores ∈ ℝ; we clip to a reasonable range for stability
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_sites,),
            dtype=np.float32,
        )

        # Internal state (initialised by reset())
        self._deployment   : np.ndarray   # (n_sites,)  bool
        self._subscribers  : np.ndarray   # (n_areas, N_OPS, N_OFFS) float
        self._t            : int

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        d = self.data

        self._t           = 0
        self._deployment  = d.init_deployment.copy()
        self._subscribers = d.init_subscribers.copy()

        obs  = self._build_observation()
        info = self._build_info()
        return obs.astype(np.float32), info

    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one MDP step.

        Parameters
        ----------
        action : (n_sites,) float array — site scores φ_s

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert not self._is_terminal(), "Episode already terminated; call reset()."

        t_prev = self._t
        t_next = t_prev + 1

        # ------------------------------------------------------------------
        # 1. Select sites to deploy (top-Z_t with hard mask)
        # ------------------------------------------------------------------
        deployed_sites = self._select_action(action)   # list[int]

        # ------------------------------------------------------------------
        # 2. Update deployment state (monotone)
        # ------------------------------------------------------------------
        prev_deployment = self._deployment.copy()
        for si in deployed_sites:
            self._deployment[si] = True

        # ------------------------------------------------------------------
        # 3. Compute ORANGE coverage before and after
        # ------------------------------------------------------------------
        coverage_before = self._compute_orange_coverage(prev_deployment)
        coverage_after  = self._compute_orange_coverage(self._deployment)

        # ------------------------------------------------------------------
        # 4. Migrate subscribers
        # ------------------------------------------------------------------
        subs_before = self._subscribers.copy()
        self._subscribers = self._apply_migration(coverage_after, t_next)
        subs_after  = self._subscribers

        # ------------------------------------------------------------------
        # 5. Advance time
        # ------------------------------------------------------------------
        self._t = t_next
        terminated = self._is_terminal()

        # ------------------------------------------------------------------
        # 6. Compute reward
        # ------------------------------------------------------------------
        reward = self._compute_reward(
            subs_before, subs_after,
            coverage_before, coverage_after,
            t_next, terminated,
        )

        obs  = self._build_observation()
        info = self._build_info(deployed_sites=deployed_sites)

        return obs.astype(np.float32), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_action(self, scores: np.ndarray) -> list[int]:
        """
        Apply hard mask to scores (already-deployed → −∞) then pick top-Z_t.
        Returns list of site indices to deploy this step.
        """
        t = self._t + 1   # decisions happen for period t+1
        Z_t = int(self.data.budget_t[t]) if t < self.T else 0
        if Z_t == 0:
            return []

        masked = scores.copy().astype(np.float64)
        masked[self._deployment] = -np.inf   # mask already-deployed

        # Number of available (not yet deployed) sites
        available = int((~self._deployment).sum())
        k = min(Z_t, available)
        if k == 0:
            return []

        # top-k indices
        top_k = np.argpartition(masked, -k)[-k:]
        # Filter out any accidental −∞ (shouldn't happen but safety)
        top_k = [i for i in top_k if self._deployment[i] == False]
        return top_k

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def _compute_orange_coverage(self, deployment: np.ndarray) -> np.ndarray:
        """
        Compute ORANGE 5G coverage per area.
        coverage[a] = 1 iff at least one deployed ORANGE 5G site covers area a.
        Returns (n_areas,) bool array.
        """
        # site_covers_area: (n_sites, n_areas)
        # coverage[a] = OR over deployed sites that cover a
        deployed_mat = self.data.site_covers_area[deployment]   # (k, n_areas)
        if deployed_mat.shape[0] == 0:
            return np.zeros(self.n_areas, dtype=bool)
        return deployed_mat.any(axis=0)

    def _population_coverage(self, coverage: np.ndarray) -> float:
        """
        Fraction of population covered by ORANGE 5G.
        coverage : (n_areas,) bool
        """
        total_pop = self.data.area_population.sum()
        if total_pop == 0:
            return 0.0
        covered_pop = (self.data.area_population * coverage).sum()
        return float(covered_pop / total_pop)

    # ------------------------------------------------------------------
    # Subscriber migration
    # ------------------------------------------------------------------

    def _build_coverage_vector(
        self,
        orange_coverage: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """
        Build per-area coverage vector for all operators.
        Shape: (n_areas, N_OPS) — 1 if operator has 5G in that area.
        """
        cov = np.zeros((self.n_areas, N_OPS), dtype=int)
        # ORANGE (index 3)
        cov[:, TARGET_IDX] = orange_coverage.astype(int)
        # Competitors — from COMPETITORS_STRATEGY
        comp_cov = self.data.competitor_cov[min(t, self.T - 1)]   # (n_areas, N_COMPS)
        for ci, comp in enumerate(COMPETITORS):
            cov[:, OP_IDX[comp]] = comp_cov[:, ci].astype(int)
        return cov   # (n_areas, N_OPS)

    def _apply_migration(self, orange_coverage: np.ndarray, t: int) -> np.ndarray:
        subs  = self._subscribers.copy()
        table = self.data.migration_table
        cov_vec = self._build_coverage_vector(orange_coverage, t)

        new_subs = subs.copy()

        for ai in range(self.n_areas):
            cov_tuple = tuple(cov_vec[ai].tolist())

            for offer in OFFERS:
                off_idx = OFF_IDX[offer]
                for from_op in OPERATORS:
                    from_idx = OP_IDX[from_op]

                    # Lire depuis l'état ORIGINAL (pas new_subs)
                    n = subs[ai, from_idx, off_idx]
                    if n <= 0:
                        continue

                    key = (cov_tuple, offer, from_op)
                    migrations = table.get(key, {})

                    # --- Calculer tous les flux AVANT d'écrire quoi que ce soit ---
                    flows = {}   # to_op -> nombre d'abonnés qui bougent
                    total_leaving = 0.0

                    for to_op, pct in migrations.items():
                        # Ignorer le cas "reste sur place en 5G"
                        if to_op == from_op and offer == NG_OFFER:
                            continue
                        migrating = n * pct
                        if self.noise_std > 0:
                            noise     = self.np_random.normal(0, self.noise_std * migrating)
                            migrating = max(0.0, migrating + noise)
                        flows[to_op] = flows.get(to_op, 0.0) + migrating  # accumulate, don't overwrite
                        total_leaving += migrating

                    # Garantir qu'on ne retire pas plus que ce qu'on a
                    if total_leaving > n:
                        scale = n / total_leaving
                        flows = {k: v * scale for k, v in flows.items()}
                        total_leaving = n

                    # --- Appliquer les flux ---
                    new_subs[ai, from_idx, off_idx] -= total_leaving

                    for to_op, move in flows.items():
                        to_idx = OP_IDX[to_op]
                        new_subs[ai, to_idx, NG_IDX] += move

        return np.clip(new_subs, 0.0, None)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        subs_before: np.ndarray,
        subs_after:  np.ndarray,
        cov_before:  np.ndarray,
        cov_after:   np.ndarray,
        t: int,
        terminated: bool,
    ) -> float:
        """
        R = α·ΔMS + β·ΔCov − λ_reg·Pen_reg² − λ_cap·Pen_cap
        + η · MS_total  (only at terminal step)
        """
        total_pop = self.data.area_population.sum()

        # ---- R1: incremental market share gain ---------------------------
        ms_before = subs_before[:, TARGET_IDX, NG_IDX].sum()
        ms_after  = subs_after[:, TARGET_IDX, NG_IDX].sum()
        delta_ms  = ms_after - ms_before
        R1        = self.alpha * delta_ms

        # ---- R2: incremental population coverage -------------------------
        pop_cov_before = self._population_coverage(cov_before)
        pop_cov_after  = self._population_coverage(cov_after)
        delta_cov      = pop_cov_after - pop_cov_before
        R2             = self.beta * delta_cov

        # ---- R3: regulatory penalty --------------------------------------
        qa_threshold = (
            self.data.qa_threshold_t[t]
            if t < self.T else self.data.qa_threshold_t[-1]
        )
        violation = max(0.0, qa_threshold - pop_cov_after)
        R3        = -self.lambda_reg * (violation ** 2)

        # ---- R4: capacity penalty ----------------------------------------
        cap_t = (
            self.data.capacity_t[t]
            if t < self.T else self.data.capacity_t[-1]
        )
        ng_cap      = cap_t[NG_IDX]
        dem_t       = (
            self.data.demand_t[t]
            if t < self.T else self.data.demand_t[-1]
        )
        ng_demand_per_sub = dem_t[NG_IDX]

        # For each site, compute traffic load from NG subscribers it serves
        # For simplicity: compute aggregate over all sites
        # Total ORANGE NG subscribers after migration
        total_ng_subs = subs_after[:, TARGET_IDX, NG_IDX].sum()
        # Number of deployed ORANGE 5G sites
        n_deployed = int(self._deployment.sum())
        if n_deployed > 0:
            # Average traffic per site
            avg_traffic = total_ng_subs * ng_demand_per_sub / n_deployed
            cap_excess  = max(0.0, avg_traffic - ng_cap)
        else:
            cap_excess = 0.0
        R4 = -self.lambda_cap * cap_excess

        # ---- Terminal bonus ----------------------------------------------
        R_bonus = 0.0
        if terminated:
            R_bonus = self.eta * ms_after

        reward = R1 + R2 + R3 + R4 + R_bonus
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        Flatten state into a 1D float32 observation vector.

        Layout:
          [deployment z | coverage r | subscribers μ (norm) |
           competitor_cov R | time τ | budget b | reg gap δ]
        """
        d     = self.data
        t     = self._t
        t_obs = min(t, self.T - 1)

        # --- Deployment (n_sites,) ----------------------------------------
        z_vec = self._deployment.astype(np.float32)

        # --- ORANGE coverage (n_areas,) ------------------------------------
        cov   = self._compute_orange_coverage(self._deployment)
        r_vec = cov.astype(np.float32)

        # --- Subscribers normalised (n_areas * N_OPS * N_OFFS,) -----------
        max_pop  = self.data.area_population.max()
        if max_pop == 0:
            max_pop = 1.0
        mu_norm  = (self._subscribers / max_pop).astype(np.float32)
        mu_vec   = mu_norm.ravel()

        # --- Competitor coverage (n_areas * N_COMPS,) ----------------------
        comp_cov = d.competitor_cov[t_obs].astype(np.float32)   # (n_areas, N_COMPS)
        comp_vec = comp_cov.ravel()

        # --- Scalar context -----------------------------------------------
        tau = np.float32(t / self.T)

        Z_t       = float(d.budget_t[t_obs])
        b         = np.float32(Z_t / self.Z_max)

        qa        = d.qa_threshold_t[t_obs]
        pop_cov   = self._population_coverage(cov)
        delta_reg = np.float32(qa - pop_cov)   # positive = in violation

        obs = np.concatenate([
            z_vec, r_vec, mu_vec, comp_vec,
            [tau, b, delta_reg],
        ])
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_terminal(self) -> bool:
        return self._t >= self.T - 1   # last step is T-1 (0-indexed)

    def _build_info(self, deployed_sites: list[int] | None = None) -> dict:
        cov   = self._compute_orange_coverage(self._deployment)
        pop_cov    = self._population_coverage(cov)
        total_ng   = float(self._subscribers[:, TARGET_IDX, NG_IDX].sum())
        total_subs = float(self._subscribers.sum())
        market_share = total_ng / total_subs if total_subs > 0 else 0.0
        t_obs = min(self._t, self.T - 1)
        return {
            "t":                  self._t,
            "n_deployed":         int(self._deployment.sum()),
            "population_coverage":pop_cov,
            "orange_ng_subs":     total_ng,
            "market_share_ng":    market_share,
            "qa_threshold":       self.data.qa_threshold_t[t_obs],
            "regulatory_ok":      pop_cov >= self.data.qa_threshold_t[t_obs],
            "deployed_sites":     deployed_sites or [],
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        info = self._build_info()
        lines = [
            f"╔══ NGDeploymentEnv — t={self._t}/{self.T} ══╗",
            f"  Sites deployed     : {info['n_deployed']}/{self.n_sites}",
            f"  Population cov.    : {info['population_coverage']:.1%}  (QA: {info['qa_threshold']:.1%})",
            f"  ORANGE 5G subs     : {info['orange_ng_subs']:.1f}",
            f"  NG market share    : {info['market_share_ng']:.1%}",
            f"  Regulatory OK      : {'✓' if info['regulatory_ok'] else '✗'}",
            f"╚{'═'*38}╝",
        ]
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    # ------------------------------------------------------------------
    # Utility — greedy oracle (for benchmarking)
    # ------------------------------------------------------------------

    def greedy_action(self) -> np.ndarray:
        """
        Heuristic: score each site by the population it would newly cover.
        Useful as a baseline and for imitation pre-training.
        """
        cov_before = self._compute_orange_coverage(self._deployment)
        scores     = np.zeros(self.n_sites, dtype=np.float32)
        for si in range(self.n_sites):
            if self._deployment[si]:
                scores[si] = -np.inf
                continue
            # New areas covered by adding site si
            test_dep = self._deployment.copy()
            test_dep[si] = True
            cov_after = self._compute_orange_coverage(test_dep)
            new_pop   = self.data.area_population[cov_after & ~cov_before].sum()
            scores[si] = float(new_pop)
        return scores