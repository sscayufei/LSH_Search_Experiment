"""Phase 6 plots: measured curves and assumptions are explicitly separated."""
import os

import numpy as np


def plot_scale(measured, predictions, figures):
    os.environ["MPLCONFIGDIR"] = str((figures.parent / ".matplotlib").resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["N"] for r in measured]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150, constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(sizes, [r["mean_candidates"] for r in measured], "o-", label="measured")
    subset = [r for r in predictions if r["quantity"] == "mean_candidates" and r["value_type"] == "extrapolated_from_measured_10k"]
    ax.plot([r["N"] for r in subset], [r["value"] for r in subset], "--", label="10k fraction extrapolation")
    ax.set(title="Candidate count: measurements and extrapolation", ylabel="Mean candidate vectors", xscale="log", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(sizes, [r["mean_false_candidates"] for r in measured], "o-", label="measured below-tau candidates")
    for kind, similarity, label, style in (("conditional_theory", 0.0, "theory: all backgrounds s=0", ":"),
                                           ("extrapolated_from_measured_10k", "", "10k fraction extrapolation", "--")):
        subset = [r for r in predictions if r["value_type"] == kind and r["quantity"] == "mean_false_candidates" and r["reference_similarity"] == similarity]
        ax.plot([r["N"] for r in subset], [r["value"] for r in subset], style, label=label)
    ax.set(title="False candidates (removed by cosine filter)", ylabel="Mean false candidate vectors", xscale="log", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(sizes, [r["mean_time_ms"] for r in measured], "o-", label="measured mean")
    ax.plot(sizes, [r["p95_time_ms"] for r in measured], "s-", label="measured P95")
    ax.set(title="Query latency: measured sizes only", ylabel="Latency (ms)", xscale="log")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(sizes, [r["index_bytes"] / 2**20 for r in measured], "o-", label="measured hybrid index estimate")
    ax.plot(sizes, [r["lsh_index_bytes"] / 2**20 for r in measured], "s-", label="measured LSH component estimate")
    ax.set(title="Additional index objects: measured sizes only", ylabel="Object-size estimate (MiB)", xscale="log")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Corpus size N")
        ax.grid(alpha=0.25)
    fig.suptitle("Fixed k=12, L=32, tau=0.75, seed=42; 100 fixed semantic queries\nMillion-scale dashed/dotted values are assumptions, not measurements")
    fig.savefig(figures / "scale_analysis.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, constrained_layout=True)
    axes[0].plot(sizes, [r["mean_candidate_ratio"] * 100 for r in measured], "o-", label="mean")
    axes[0].plot(sizes, [r["p95_candidate_ratio"] * 100 for r in measured], "s-", label="P95")
    axes[0].set(title="Measured candidate ratio", ylabel="Candidate ratio (%)")
    axes[0].legend()
    axes[1].plot(sizes, [r["max_bucket_size"] for r in measured], "o-", label="maximum occupied bucket")
    axes[1].plot(sizes, [r["p95_occupied_bucket_size"] for r in measured], "s-", label="P95 occupied bucket")
    axes[1].set(title="Measured bucket crowding", ylabel="Vectors per bucket")
    axes[1].legend(fontsize=8)
    axes[2].plot(sizes, [r["recall"] for r in measured], "o-")
    for i, row in enumerate(measured):
        alignment = "left" if i == 0 else "right" if i == len(measured) - 1 else "center"
        offset = 6 if i == 0 else -6 if i == len(measured) - 1 else 0
        axes[2].annotate(f"{row['nonempty_ground_truth_queries']}/100 G>0", (row["N"], row["recall"]), xytext=(offset, 8), textcoords="offset points", ha=alignment, fontsize=8)
    axes[2].set(title="Measured conditional Recall", ylabel="Recall", ylim=(0, 1.06))
    for ax in axes:
        ax.set(xlabel="Corpus size N", xscale="log")
        ax.grid(alpha=0.25)
    fig.savefig(figures / "scale_diagnostics.png")
    plt.close(fig)
