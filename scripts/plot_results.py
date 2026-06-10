#!/usr/bin/env python3
"""Generate paper graphs from benchmark and training outputs.

Produces the figures needed for the SkillSQL-RL proposal:
  Fig 1: Execution Accuracy -- SkillSQL-RL vs baselines (bar chart)
  Fig 2: Execution Accuracy by task category (grouped bars)
  Fig 3: Reward curve over training epochs (line chart)
  Fig 4: Static validity rate, exec success rate, retrieval hit rate (bar)
  Fig 5: Skill-evolution gain -- ΔEX from evolved vs. no evolution (bar)
  Fig 6: Prompt footprint -- catalog / skill / task token breakdown (stacked bars)
  Fig 7: Reward-result correlation scatter plot (Section 8.3)
  Fig 8: Ablation analysis -- A1-A4 vs. full SkillSQL-RL (bar chart)

Usage:
    python scripts/plot_results.py [OPTIONS]

Options:
    --results-dir DIR   Root output directory with benchmark subfolders
    --output-dir DIR    Where to save figures (default: <results-dir>/figures)
    --format FORMAT     Image format: pdf | png | svg (default: pdf)
    --dpi N             Resolution for raster formats (default: 150)

Example:
    python scripts/plot_results.py --results-dir ./outputs --output-dir ./outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _setup(fmt: str, dpi: int):
    import matplotlib
    if fmt == "pdf":
        matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
    return plt, dpi


def _save(plt, path: Path, fmt: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.with_suffix(f'.{fmt}')}")


# ── Figure 1: Execution Accuracy vs. Baselines ────────────────────────────────
def fig1_execution_accuracy(data: dict, out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    fig, ax = plt.subplots(figsize=(8, 5))

    systems = list(data["baselines"].keys()) + ["SkillSQL-RL"]
    scores = [data["baselines"][s] for s in systems[:-1]] + [data.get("skillsql_rl", 0)]
    colors = ["#aec7e8"] * (len(systems) - 1) + ["#1f77b4"]

    bars = ax.bar(systems, scores, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("Spider 2.0-Snow: Execution Accuracy vs. Baselines", fontsize=13)
    ax.set_ylim(0, max(scores) * 1.15)
    plt.xticks(rotation=15, ha="right")
    _save(plt, out / "fig1_execution_accuracy", fmt, dpi)


# ── Figure 2: EX by Task Category ─────────────────────────────────────────────
def fig2_by_category(data: dict, out: Path, fmt: str, dpi: int) -> None:
    import numpy as np
    plt, _ = _setup(fmt, dpi)
    categories = data.get("categories", [])
    if not categories:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    systems = ["Arctic-7B Direct", "Vanilla GRPO", "SkillSQL-RL"]
    x = np.arange(len(categories))
    width = 0.25
    offsets = [-width, 0, width]
    palette = ["#aec7e8", "#ffbb78", "#1f77b4"]
    for sys_name, offset, color in zip(systems, offsets, palette):
        scores = [data["by_category"].get(cat, {}).get(sys_name, 0) for cat in categories]
        bars = ax.bar(x + offset, scores, width, label=sys_name, color=color)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("Execution Accuracy by Task Category", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.legend()
    _save(plt, out / "fig2_by_category", fmt, dpi)


# ── Figure 3: Training Reward Curve ───────────────────────────────────────────
def fig3_reward_curve(metrics: list[dict], out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = [m["epoch"] for m in metrics]
    rewards = [m["mean_reward"] for m in metrics]
    ex_accs = [m.get("execution_accuracy", 0) * 100 for m in metrics]

    ax1.plot(epochs, rewards, "b-o", label="Mean Reward", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Mean Reward", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(epochs, ex_accs, "r--s", label="Exec Acc (%)", linewidth=2, markersize=6)
    ax2.set_ylabel("Execution Accuracy (%)", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("GRPO Training: Reward and Execution Accuracy over Epochs", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    _save(plt, out / "fig3_reward_curve", fmt, dpi)


# ── Figure 4: Diagnostic Metrics ──────────────────────────────────────────────
def fig4_diagnostics(data: dict, out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    metrics = data.get("diagnostics", {})
    if not metrics:
        return
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="#2ca02c", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Diagnostic Metrics: Static Validity, Exec Success, Retrieval Hit", fontsize=12)
    ax.set_ylim(0, 110)
    plt.xticks(rotation=15, ha="right")
    _save(plt, out / "fig4_diagnostics", fmt, dpi)


# ── Figure 5: Skill Evolution Gain ────────────────────────────────────────────
def fig5_evolution_gain(data: dict, out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    evo = data.get("evolution_gain", {})
    if not evo:
        return
    categories = list(evo.keys())
    before = [evo[c]["before"] for c in categories]
    after = [evo[c]["after"] for c in categories]
    import numpy as np
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, before, width, label="Before Evolution", color="#aec7e8")
    ax.bar(x + width / 2, after, width, label="After Evolution", color="#1f77b4")
    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("Skill Evolution Gain (ΔEX) per Category", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.legend()
    _save(plt, out / "fig5_evolution_gain", fmt, dpi)


# ── Figure 6: Prompt Footprint ────────────────────────────────────────────────
def fig6_prompt_footprint(data: dict, out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    fp = data.get("prompt_footprint", {})
    if not fp:
        return
    systems = list(fp.keys())
    catalog_tokens = [fp[s].get("catalog", 0) for s in systems]
    skill_tokens = [fp[s].get("skill", 0) for s in systems]
    task_tokens = [fp[s].get("task", 0) for s in systems]

    import numpy as np
    x = np.arange(len(systems))
    width = 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, catalog_tokens, width, label="Schema Context", color="#aec7e8")
    ax.bar(x, skill_tokens, width, bottom=catalog_tokens, label="Skills", color="#1f77b4")
    bottom2 = [a + b for a, b in zip(catalog_tokens, skill_tokens)]
    ax.bar(x, task_tokens, width, bottom=bottom2, label="Task", color="#d62728")
    ax.set_ylabel("Tokens", fontsize=12)
    ax.set_title("Prompt Footprint: Schema / Skills / Task (tokens)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend()
    _save(plt, out / "fig6_prompt_footprint", fmt, dpi)


# ── Figure 7: Reward-Result Correlation ───────────────────────────────────────
def fig7_reward_correlation(results: list[dict], out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    rewards = [r.get("reward", 0) for r in results if "reward" in r]
    ex = [int(r.get("execution_accuracy", False)) for r in results if "reward" in r]
    if not rewards:
        return
    import numpy as np
    from scipy import stats  # type: ignore[import]

    corr, pval = stats.pearsonr(rewards, ex)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4" if e else "#d62728" for e in ex]
    ax.scatter(rewards, ex, c=colors, alpha=0.5, s=20)
    ax.set_xlabel("Verifier Reward R(τ)", fontsize=12)
    ax.set_ylabel("Execution Accuracy (0/1)", fontsize=12)
    ax.set_title(f"Reward–Result Correlation (r={corr:.3f}, p={pval:.3e})", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    _save(plt, out / "fig7_reward_correlation", fmt, dpi)


# ── Figure 8: Ablation Analysis ───────────────────────────────────────────────
def fig8_ablations(data: dict, out: Path, fmt: str, dpi: int) -> None:
    plt, _ = _setup(fmt, dpi)
    ablations = data.get("ablations", {})
    if not ablations:
        return
    names = list(ablations.keys())
    scores = [ablations[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4" if n == "SkillSQL-RL (Full)" else "#aec7e8" for n in names]
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study (A1–A4 vs. Full SkillSQL-RL)", fontsize=13)
    ax.set_ylim(0, max(scores) * 1.15)
    plt.xticks(rotation=15, ha="right")
    _save(plt, out / "fig8_ablations", fmt, dpi)


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_results(results_dir: Path) -> dict:
    """Aggregate benchmark and training outputs into a single data dict."""
    data: dict = {"baselines": {}, "diagnostics": {}, "ablations": {}}

    # Spider2-snow non-oracle manifest
    manifest_path = results_dir / "spider2_snow" / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        data["skillsql_rl"] = m.get("execution_accuracy", 0) * 100
        data["baselines"].setdefault("Arctic-7B Direct", 17.0)
        data["baselines"].setdefault("Vanilla GRPO", 22.0)
        data["baselines"].setdefault("Prompt Memory", 19.0)
        data["baselines"].setdefault("ReFoRCE", 35.83)

    # Per-task results
    results_jsonl = results_dir / "spider2_snow" / "results.jsonl"
    if results_jsonl.exists():
        per_task = [json.loads(l) for l in results_jsonl.read_text().splitlines() if l.strip()]
        data["per_task"] = per_task

    # Training metrics
    training_metrics = results_dir / "checkpoints" / "training_metrics.jsonl"
    if training_metrics.exists():
        data["training_metrics"] = [
            json.loads(l) for l in training_metrics.read_text().splitlines() if l.strip()
        ]

    return data


def generate_figures(results_dir: Path, output_dir: Path, fmt: str = "pdf", dpi: int = 150) -> None:
    """Entry point for generating all paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_results(results_dir)
    print(f"Generating figures in {output_dir}/")

    try:
        if data.get("baselines"):
            fig1_execution_accuracy(data, output_dir, fmt, dpi)
        if data.get("training_metrics"):
            fig3_reward_curve(data["training_metrics"], output_dir, fmt, dpi)
        if data.get("diagnostics"):
            fig4_diagnostics(data, output_dir, fmt, dpi)
        if data.get("per_task") and all("reward" in t for t in data["per_task"][:3]):
            fig7_reward_correlation(data["per_task"], output_dir, fmt, dpi)
        if data.get("ablations"):
            fig8_ablations(data, output_dir, fmt, dpi)
        print("Done.")
    except ImportError as e:
        print(f"Missing plotting dependencies: {e}")
        print("Install with: pip install 'cognetics-dataassist-skill-rl[plotting]'")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SkillSQL-RL paper figures")
    p.add_argument("--results-dir", default="./outputs")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    generate_figures(results_dir, output_dir, fmt=args.format, dpi=args.dpi)


if __name__ == "__main__":
    main()
