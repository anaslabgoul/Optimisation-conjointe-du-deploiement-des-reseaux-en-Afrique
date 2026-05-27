"""
evaluate_ng_deployment.py
=========================
Evaluates a trained PPO model against three baselines on the NGDeploymentEnv
and prints a detailed report to the terminal.

Strategies compared
-------------------
  PPO          — your trained model
  Greedy       — built-in population-coverage heuristic
  Random       — uniform random site scores
  Do-nothing   — never deploys any site

Usage
-----
  python evaluate_ng_deployment.py \
      --data_dir  /path/to/INPUTS \
      --model     ng_deployment_ppo \
      --norm_pkl  ng_deployment_norm.pkl \
      --episodes  10
"""

import argparse
import numpy as np
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ng_deployment_env import NGDeploymentEnv


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(data_dir: str, noise_std: float = 0.0):
    return NGDeploymentEnv(data_dir=data_dir, noise_std=noise_std)


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def make_ppo_policy(model, vec_norm: VecNormalize):
    def policy(obs, env):
        obs_batch = obs[np.newaxis, :]
        obs_norm  = vec_norm.normalize_obs(obs_batch)
        action, _ = model.predict(obs_norm, deterministic=True)
        return action[0]
    return policy

def greedy_policy(obs, env):     return env.greedy_action()
def random_policy(obs, env):     return env.action_space.sample()
def do_nothing_policy(obs, env): return np.full(env.n_sites, -1e9, dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, policy_fn) -> dict:
    obs, _ = env.reset()
    history = defaultdict(list)
    cumulative_reward = 0.0

    while True:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        history["t"].append(info["t"])
        history["n_deployed"].append(info["n_deployed"])
        history["population_coverage"].append(info["population_coverage"])
        history["market_share_ng"].append(info["market_share_ng"])
        history["orange_ng_subs"].append(info["orange_ng_subs"])
        history["qa_threshold"].append(info["qa_threshold"])
        history["regulatory_ok"].append(info["regulatory_ok"])
        history["reward"].append(reward)
        history["cumulative_reward"].append(cumulative_reward)

        if terminated or truncated:
            break

    return dict(history)


def evaluate_strategy(name, policy_fn, data_dir, n_episodes, noise_std=0.0):
    env = make_env(data_dir, noise_std=noise_std)
    episodes = []
    for ep in range(n_episodes):
        episodes.append(run_episode(env, policy_fn))
    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def agg(episodes, key, reducer=lambda x: x[-1]):
    """Extract a scalar from each episode then return mean, std."""
    vals = [reducer(ep[key]) for ep in episodes]
    return float(np.mean(vals)), float(np.std(vals))


def pct_steps_ok(episodes):
    all_steps = [v for ep in episodes for v in ep["regulatory_ok"]]
    return 100.0 * sum(all_steps) / len(all_steps)


def step_means(episodes, key):
    T = len(episodes[0][key])
    return [float(np.mean([ep[key][t] for ep in episodes])) for t in range(T)]


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

STRATEGY_ORDER = ["PPO", "Greedy", "Random", "Do-nothing"]
W = 72   # total box width

def bar(value, total=1.0, width=20, fill="█", empty="░"):
    filled = int(round(value / total * width))
    return fill * filled + empty * (width - filled)


def hline(char="─", width=W):
    return char * width


def print_header(title: str):
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print("║" + title.center(W - 2) + "║")
    print("╚" + "═" * (W - 2) + "╝")


def section(title: str):
    pad = max(0, W - 6 - len(title))
    print()
    print(f"  ┌─ {title} {'─' * pad}┐")


def section_end():
    print(f"  └{'─' * (W - 4)}┘")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    section("SUMMARY  (mean ± std over episodes, final step)")

    COL = 16
    header = f"  │  {'Metric':<28}" + "".join(f"{s:>{COL}}" for s in STRATEGY_ORDER) + "  │"
    print(header)
    print(f"  │  {'─'*28}" + "─" * (COL * len(STRATEGY_ORDER)) + "──│")

    def row(label, vals_dict):
        r = f"  │  {label:<28}"
        for s in STRATEGY_ORDER:
            r += f"{vals_dict[s]:>{COL}}"
        print(r + "  │")

    # Total reward
    row("Total reward",
        {s: f"{agg(results[s], 'cumulative_reward')[0]:+.1f} "
            f"±{agg(results[s], 'cumulative_reward')[1]:.1f}"
         for s in STRATEGY_ORDER})

    # Final NG market share
    row("Final NG market share",
        {s: f"{agg(results[s], 'market_share_ng')[0]:.2%}"
         for s in STRATEGY_ORDER})

    # Final population coverage
    row("Final pop. coverage",
        {s: f"{agg(results[s], 'population_coverage')[0]:.2%}"
         for s in STRATEGY_ORDER})

    # Final sites deployed
    row("Final sites deployed",
        {s: f"{agg(results[s], 'n_deployed')[0]:.1f}"
         for s in STRATEGY_ORDER})

    # Final ORANGE NG subscribers
    row("Final ORANGE NG subs",
        {s: f"{agg(results[s], 'orange_ng_subs')[0]:.0f}"
         for s in STRATEGY_ORDER})

    # % steps regulatory OK
    row("% steps reg. compliant",
        {s: f"{pct_steps_ok(results[s]):.1f}%"
         for s in STRATEGY_ORDER})

    section_end()


def print_visual_bars(results: dict):
    section("VISUAL COMPARISON  (final values, mean over episodes)")

    for label, key, is_frac in [
        ("NG Market Share",    "market_share_ng",    True),
        ("Pop. Coverage",      "population_coverage", True),
        ("Reg. Compliance",    "regulatory_ok",       False),
    ]:
        print(f"\n    {label}")
        for s in STRATEGY_ORDER:
            eps = results[s]
            if key == "regulatory_ok":
                val = pct_steps_ok(eps) / 100.0
            else:
                val, _ = agg(eps, key)
            b = bar(val)
            print(f"    {s:<14} {b}  {val:.1%}")

    section_end()


def print_step_table(results: dict):
    section("STEP-BY-STEP  population coverage vs QA threshold")

    T = len(next(iter(results.values()))[0]["t"])
    cov_means = {s: step_means(results[s], "population_coverage") for s in STRATEGY_ORDER}
    qa_means  = step_means(next(iter(results.values())), "qa_threshold")

    COL = 13
    print(f"  │  {'t':>3}  {'QA':>5}  " +
          "".join(f"{'['+s+']':>{COL}}" for s in STRATEGY_ORDER) + "  │")
    print(f"  │  {'─'*3}  {'─'*5}  " + "─" * (COL * len(STRATEGY_ORDER)) + "──│")

    for t in range(T):
        qa  = qa_means[t]
        row = f"  │  {t:>3}  {qa:>4.0%}  "
        for s in STRATEGY_ORDER:
            cov = cov_means[s][t]
            ok  = "✓" if cov >= qa else "✗"
            row += f"  {cov:>5.1%} {ok}    "
        print(row + "│")

    section_end()


def print_market_share_table(results: dict):
    section("STEP-BY-STEP  ORANGE NG market share")

    T = len(next(iter(results.values()))[0]["t"])
    ms_means = {s: step_means(results[s], "market_share_ng") for s in STRATEGY_ORDER}

    COL = 13
    print(f"  │  {'t':>3}  " +
          "".join(f"{'['+s+']':>{COL}}" for s in STRATEGY_ORDER) + "  │")
    print(f"  │  {'─'*3}  " + "─" * (COL * len(STRATEGY_ORDER)) + "──│")

    for t in range(T):
        row = f"  │  {t:>3}  "
        for s in STRATEGY_ORDER:
            ms = ms_means[s][t]
            row += f"  {ms:>8.2%}     "
        print(row + "│")

    section_end()


def print_ppo_episode_detail(results: dict):
    section("PPO — per-episode breakdown")

    print(f"  │  {'Ep':>3}  {'Tot. Reward':>12}  {'Final MS':>10}  "
          f"{'Final Cov':>10}  {'Sites':>6}  {'Reg.OK%':>8}  │")
    print(f"  │  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*8}  │")

    for i, ep in enumerate(results["PPO"]):
        r   = ep["cumulative_reward"][-1]
        ms  = ep["market_share_ng"][-1]
        cov = ep["population_coverage"][-1]
        nd  = ep["n_deployed"][-1]
        ok  = 100.0 * sum(ep["regulatory_ok"]) / len(ep["regulatory_ok"])
        subs = ep["orange_ng_subs"][-1]
        print(f"  │  {i:>3}  {r:>+12.2f}  {ms:>9.2%}  {cov:>9.2%}  "
              f"{nd:>6}  {ok:>7.1f}% {subs:>11.0f} │")
        
        

    section_end()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="/Users/anasstakfa/Desktop/PSC/instance_PSC_MAYENNE/INPUTS")
    parser.add_argument("--model",     default="ng_deployment_ppo")
    parser.add_argument("--norm_pkl",  default="ng_deployment_norm.pkl")
    parser.add_argument("--episodes",  type=int,   default=10)
    parser.add_argument("--noise_std", type=float, default=0.0)
    args = parser.parse_args()

    # --- Load PPO ----------------------------------------------------------
    print("\nLoading PPO model …")
    dummy_env = DummyVecEnv([lambda: make_env(args.data_dir)])
    vec_norm  = VecNormalize.load(args.norm_pkl, dummy_env)
    vec_norm.training    = False
    vec_norm.norm_reward = False
    model = PPO.load(args.model, env=dummy_env)
    print("  ✓ Model loaded.")

    # --- Strategies --------------------------------------------------------
    strategies = {
        "PPO":        make_ppo_policy(model, vec_norm),
        "Greedy":     greedy_policy,
        "Random":     random_policy,
        "Do-nothing": do_nothing_policy,
    }

    # --- Run ---------------------------------------------------------------
    results = {}
    for name, fn in strategies.items():
        print(f"  Running [{name}] × {args.episodes} episodes …", end=" ", flush=True)
        results[name] = evaluate_strategy(
            name, fn, args.data_dir, args.episodes, args.noise_std
        )
        print("done.")

    # --- Print report ------------------------------------------------------
    print_header("  NG 5G DEPLOYMENT — STRATEGY EVALUATION REPORT  ")
    print_summary_table(results)
    print_visual_bars(results)
    print_step_table(results)
    print_market_share_table(results)
    print_ppo_episode_detail(results)
    

    print("\n  Report complete.\n")


if __name__ == "__main__":
    main()
