"""Scientific figures from actual test summary; no parameter selection."""
import os
from pathlib import Path

import numpy as np


def plot_results(summary, figures, retrieval):
    figures = Path(figures)
    os.environ["MPLCONFIGDIR"] = str((figures.parent / ".matplotlib").resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("Brute-force", "Exact Hash Only", "Hash + LSH")
    labels = ("Brute-force", "Exact Hash\nOnly", "Hash + LSH")
    colors = ("#59758a", "#a7a7a7", "#d3893c")
    semantic = [next(r for r in summary if r["method"] == m and r["group"] == "semantic") for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=150, constrained_layout=True)
    fields = (("recall", "Conditional semantic Recall", 1),
              ("mean_candidate_ratio", "Mean candidate ratio (%)", 100),
              ("mean_time_ms", "Mean latency (ms, log scale)", 1),
              ("index_mib", "Additional index estimate (MiB)", 1))
    for ax, (field, title, scale) in zip(axes.flat, fields, strict=True):
        values = [r[field] * scale if r[field] is not None else np.nan for r in semantic]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.bar_label(bars, labels=[f"{v:.4g}" if np.isfinite(v) else "N/A" for v in values], padding=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if field == "mean_time_ms":
            ax.set_yscale("log")
            ax.set_ylim(min(values) / 2, max(values) * 3)
        else:
            ax.set_ylim(0, 1.15 if field == "recall" else max(values) * 1.2)
    count = semantic[0]
    fig.suptitle(f"Test semantic queries: {count['queries']} total, {count['nonempty_ground_truth_queries']} nonempty Ground Truth\n"
                 f"tau={retrieval['tau']}, k={retrieval['k']}, L={retrieval['L']}; precomputed embeddings")
    fig.savefig(figures / "method_comparison.png")
    plt.close(fig)

    groups = ("semantic", "exact", "all")
    group_labels = ("Semantic", "Exact", "Mixed")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=150, constrained_layout=True)
    positions = np.arange(3)
    for i, (method, color) in enumerate(zip(methods, colors, strict=True)):
        rows = [next(r for r in summary if r["method"] == method and r["group"] == group) for group in groups]
        for ax, field in zip(axes, ("recall", "mean_time_ms"), strict=True):
            bars = ax.bar(positions + (i - 1) * 0.25, [r[field] if r[field] is not None else np.nan for r in rows], width=0.24, label=method, color=color)
            ax.bar_label(bars, labels=[f"{r[field]:.3g}" if r[field] is not None else "N/A" for r in rows], padding=3, fontsize=8)
            ax.set_xticks(positions, group_labels)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    axes[0].set(title="Conditional Recall by query group", ylim=(0, 1.16))
    axes[1].set(title="Mean latency by group (ms, log scale)", yscale="log")
    values = [r["mean_time_ms"] for r in summary]
    axes[1].set_ylim(min(values) / 2, max(values) * 4)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.savefig(figures / "query_group_comparison.png")
    plt.close(fig)
