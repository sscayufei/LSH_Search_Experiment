"""Rerender Phase 6 plots while auditing that all numerical artifacts stay unchanged."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from src.plot_scale import plot_scale
from src.prepare_data import checksum, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(args.results_dir)
    manifest_path = root / "raw/scale_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    numerical = {name: info["sha256"] for name, info in manifest["outputs"].items() if not name.endswith(".png")}
    if any(checksum(name) != digest for name, digest in numerical.items()):
        raise ValueError("Numerical artifacts changed before plot rendering")
    measured = [json.loads((root / f"raw/scale_n{N}/summary.json").read_text(encoding="utf-8")) for N in manifest["measured_sizes"]]
    with (root / "raw/scale_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    for row in predictions:
        row["N"], row["value"] = int(row["N"]), float(row["value"])
        row["reference_similarity"] = float(row["reference_similarity"]) if row["reference_similarity"] else ""
    previous = {name: info for name, info in manifest["outputs"].items() if name.endswith(".png")}
    old_code = manifest["code_sha256"][str(Path("src/plot_scale.py"))]
    plot_scale(measured, predictions, root / "figures")
    if any(checksum(name) != digest for name, digest in numerical.items()):
        raise AssertionError("Rendering altered numerical results")
    for name in previous:
        manifest["outputs"][name] = {"bytes": Path(name).stat().st_size, "sha256": checksum(name)}
    manifest["render_only_update"] = {"utc": datetime.now(timezone.utc).isoformat(), "reason": "place endpoint labels inside the diagnostic axes",
                                          "previous_plot_code_sha256": old_code, "previous_plot_outputs": previous, "numerical_artifacts_unchanged": True}
    manifest["code_sha256"][str(Path("src/plot_scale.py"))] = checksum("src/plot_scale.py")
    manifest["code_sha256"][str(Path(__file__))] = checksum(__file__)
    save_json(manifest_path, manifest)
    print("Scale figures rerendered; every numerical artifact checksum is unchanged")


if __name__ == "__main__":
    main()
