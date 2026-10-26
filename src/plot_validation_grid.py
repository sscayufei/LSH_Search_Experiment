"""Render a readable log-scale view of the measured validation grid, without rerunning it."""
import csv
import json
import os
from pathlib import Path

from src.prepare_data import checksum, save_json


def main():
    root = Path("results")
    os.environ["MPLCONFIGDIR"] = str((root / ".matplotlib").resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw, figures = root / "raw", root / "figures"
    output = figures / "recall_candidate_log.png"
    if output.exists():
        raise FileExistsError("Log-scale plot already exists")
    with (raw / "validation_grid.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plan = json.loads((raw / "phase4_plan.json").read_text(encoding="utf-8"))["policy"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, constrained_layout=True)
    for k in sorted({int(r["k"]) for r in rows}):
        group = [r for r in rows if int(r["k"]) == k]
        x, y = [float(r["mean_candidate_ratio"]) * 100 for r in group], [float(r["recall"]) for r in group]
        ax.plot(x, y, "o-", label=f"k={k}")
        for px, py, row in zip(x, y, group, strict=True):
            ax.annotate(f"L={row['L']}", (px, py), xytext=(5, 5), textcoords="offset points", fontsize=9)
    selected = next(r for r in rows if r["selected"] == "True")
    ax.scatter([float(selected["mean_candidate_ratio"]) * 100], [float(selected["recall"])], marker="*", s=220, color="black", zorder=5, label="selected")
    front = sorted([r for r in rows if r["pareto"] == "True"], key=lambda r: float(r["mean_candidate_ratio"]))
    ax.plot([float(r["mean_candidate_ratio"]) * 100 for r in front], [float(r["recall"]) for r in front], "k--", alpha=0.35, label="Pareto frontier")
    ax.axvline(plan["selection"]["max_mean_candidate_ratio"] * 100, color="gray", linestyle=":", label="5% mean candidate budget")
    ax.axhline(plan["selection"]["measured_recall_goal"], color="gray", linestyle="-.", alpha=0.7, label="0.90 Recall goal")
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], labels=["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set(xlim=(0.012, 29), ylim=(0.18, 1.065), xlabel="Mean candidate ratio (%, log scale)", ylabel="Conditional semantic Recall", title=f"Validation tradeoff: tau={selected['tau']}, seed={selected['seed']}")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(output)
    plt.close(fig)
    inputs = [raw / "validation_grid.csv", raw / "phase4_plan.json", raw / "phase4_manifest.json"]
    save_json(raw / "phase4_plot_manifest.json", {"purpose": "Additional log-scale view only; measured values and frozen selection unchanged",
              "inputs": {str(p): checksum(p) for p in inputs}, "code_sha256": checksum(__file__),
              "matplotlib": matplotlib.__version__, "output": {str(output): checksum(output)}})
    print(f"Saved {output}; numerical measurements and parameter lock unchanged")


if __name__ == "__main__":
    main()
