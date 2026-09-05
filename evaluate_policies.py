"""
Evaluation + visualization pipeline comparing the Baseline AV against
Policy A (eHMI + wait-time trigger), Policy B (Intelligent Intersection /
I2V), and Policy C (interaction-aware prediction).

Run directly to regenerate the comparison table and figures:

    python3 evaluate_policies.py

Figures are written to ./figures/:
    headline_metrics.png   -- 2x2 bar comparison of the four headline metrics
    dynamics_<scenario>.png-- yield/exploitation/throughput over rounds
    wait_streak.png        -- distribution of consecutive-yield streak lengths
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import av_policies as ap

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

POLICIES = ["baseline", "A", "B", "C"]
SCENARIO_KEYS = [key for key, _, _ in ap.SCENARIOS]


def run_all(n_rounds=ap.N_ROUNDS, seed=42):
    """results[scenario][policy] -> simulate_population(...) dict"""
    results = {}
    for scenario in SCENARIO_KEYS:
        mus = ap.sample_mus(scenario, seed=seed)
        results[scenario] = {}
        for policy in POLICIES:
            results[scenario][policy] = ap.simulate_population(
                policy, mus, n_rounds=n_rounds, seed=seed
            )
    return results


def build_summary_table(results, tail=10):
    rows = []
    for scenario in SCENARIO_KEYS:
        for policy in POLICIES:
            s = ap.summarize(results[scenario][policy], tail=tail)
            s["scenario"] = scenario
            s["avg_delay"] = s["av_yield_rate"] * ap.w
            rows.append(s)
    return pd.DataFrame(rows)


def _style_ax(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")


def plot_headline_bars(df, save_path, close=True):
    metrics = [
        ("avg_delay", "Average AV Delay\n(yield rate x w, cost units)", False),
        ("throughput_rate", "Throughput Rate\n(successful resolutions, %)", True),
        ("exploitation_rate", "Exploitation Rate\n(AV yielded unnecessarily, %)", True),
        ("collision_rate", "Collision Rate\n(%)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("white")
    x = np.arange(len(POLICIES))

    for ax, (col, title, as_pct) in zip(axes.flat, metrics):
        means = df.groupby("policy")[col].mean().reindex(POLICIES)
        stds = df.groupby("policy")[col].std().reindex(POLICIES)
        vals = means * 100 if as_pct else means
        errs = stds * 100 if as_pct else stds

        colors = [ap.POLICY_COLORS[p] for p in POLICIES]
        bars = ax.bar(x, vals, yerr=errs, color=colors, width=0.6,
                      capsize=4, error_kw={"linewidth": 1, "ecolor": "#52514e"})
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "A", "B", "C"], fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        _style_ax(ax)

        for rect, val in zip(bars, vals):
            label = f"{val:.1f}%" if as_pct else f"{val:.2f}"
            ax.annotate(label, (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#0b0b0b")

    handles = [plt.Rectangle((0, 0), 1, 1, color=ap.POLICY_COLORS[p]) for p in POLICIES]
    fig.legend(handles, [ap.POLICY_LABELS[p] for p in POLICIES],
               loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Baseline vs. Mitigation Policies -- averaged over 4 belief scenarios "
        f"(N={ap.N_HUMANS} humans x {ap.N_ROUNDS} rounds, error bars = std across scenarios)",
        fontsize=11.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_dynamics(results, scenario, save_path, close=True):
    rounds = np.arange(1, ap.N_ROUNDS + 1)
    series = [
        ("av_yield_rate", "AV Yield Rate (%)"),
        ("exploitation_rate", "Exploitation Rate (%)"),
        ("success_rate", "Throughput / Success Rate (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor("white")

    for ax, (key, title) in zip(axes, series):
        for policy in POLICIES:
            y = results[scenario][policy][key] * 100
            ax.plot(rounds, y, color=ap.POLICY_COLORS[policy], linewidth=2,
                    label=ap.POLICY_LABELS[policy])
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylim(-5, 105)
        _style_ax(ax)

    axes[0].set_ylabel("%")
    axes[-1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False)

    scenario_label = dict((k, l) for k, l, _ in ap.SCENARIOS)[scenario]
    fig.suptitle(
        f"Interaction Dynamics Over Rounds -- '{scenario_label}' human population",
        fontsize=12, fontweight="bold", y=1.05,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_wait_streak(results, save_path, close=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    data = []
    for policy in POLICIES:
        pooled = np.concatenate(
            [results[s][policy]["wait_streak_lengths"] for s in SCENARIO_KEYS]
        )
        data.append(pooled)

    bp = ax.boxplot(
        data, vert=True, showfliers=True, patch_artist=True,
        widths=0.5, labels=["Baseline", "A", "B", "C"],
    )
    for patch, policy in zip(bp["boxes"], POLICIES):
        patch.set_facecolor(ap.POLICY_COLORS[policy])
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("#0b0b0b")

    ax.set_yscale("log")
    ax.set_ylabel("Consecutive-yield streak length (rounds, log scale)")
    ax.set_title(
        "Wait-Streak / Gridlock Duration Distribution\n"
        "(Baseline never self-corrects; Policy A caps it at the wait threshold)",
        fontsize=11, fontweight="bold",
    )
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = run_all()
    df = build_summary_table(results)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    display_df = df.set_index(["scenario", "policy"])[
        ["av_yield_rate", "avg_delay", "throughput_rate", "exploitation_rate",
         "collision_rate", "gridlock_rate", "avg_wait_streak", "max_wait_streak"]
    ]
    print("=" * 100)
    print("PER-SCENARIO METRICS (steady state, last 10 of 30 rounds)")
    print("=" * 100)
    print(display_df.to_string())

    print("\n" + "=" * 100)
    print("AVERAGED ACROSS SCENARIOS")
    print("=" * 100)
    overall = df.groupby("policy")[
        ["av_yield_rate", "avg_delay", "throughput_rate", "exploitation_rate",
         "collision_rate", "gridlock_rate"]
    ].mean().reindex(POLICIES)
    print(overall.to_string())

    plot_headline_bars(df, os.path.join(FIG_DIR, "headline_metrics.png"))
    for scenario in SCENARIO_KEYS:
        plot_dynamics(results, scenario, os.path.join(FIG_DIR, f"dynamics_{scenario}.png"))
    plot_wait_streak(results, os.path.join(FIG_DIR, "wait_streak.png"))

    print(f"\nFigures written to {FIG_DIR}/")
    return df, results


if __name__ == "__main__":
    main()
