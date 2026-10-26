"""Phase 6 real nested corpus: preserve original 10k, encode unique extensions offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import yaml

from src.benchmark import load_frozen_config
from src.ground_truth import load_jsonl
from src.prepare_data import checksum, save_json, verify_saved

LOG = logging.getLogger(__name__)


def build_nested_records(base, source_rows, excluded_texts, target, seed):
    if target < len(base) or len({r["text"] for r in base}) != len(base):
        raise ValueError("Invalid target or duplicate base texts")
    if [r["doc_id"] for r in base] != list(range(len(base))):
        raise ValueError("Base IDs must match original row order")
    seen = {r["text"] for r in base} | set(excluded_texts)
    extension = []
    for pair_id, row in enumerate(source_rows):
        for column in ("anchor", "positive"):
            text = row.get(column)
            if not isinstance(text, str) or not text.strip() or text in seen:
                continue
            seen.add(text)
            extension.append({"text": text, "source_pair_id": pair_id, "source_column": column})
    available = len(base) + len(extension)
    if target > available:
        raise ValueError(f"Need {target} unique texts; only {available} available without query leakage")
    random.Random(seed).shuffle(extension)
    records = [dict(r) for r in base]
    for row in extension[:target - len(base)]:
        records.append({"doc_id": len(records), **row})
    return records, available


def validate_vectors(vectors, rows, dimension):
    if vectors.dtype != np.float32 or vectors.shape != (rows, dimension):
        raise ValueError("Scale embeddings have invalid dtype/shape")
    maximum_error = 0.0
    for start in range(0, rows, 4096):
        block = vectors[start:start + 4096]
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite scale embeddings")
        error = float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1)))
        if error > 1e-4:
            raise ValueError("Scale embeddings are not normalized")
        maximum_error = max(maximum_error, error)
    return maximum_error


def verify_scale_data(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["outputs"].items():
        if checksum(output / name) != info["sha256"]:
            raise ValueError(f"Scale corpus checksum mismatch: {name}")
    records = load_jsonl(output / "corpus.jsonl")
    vectors = np.load(output / "corpus_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if [r["doc_id"] for r in records] != list(range(len(records))) or len({r["text"] for r in records}) != len(records):
        raise ValueError("Scale corpus IDs/texts invalid")
    error = validate_vectors(vectors, len(records), manifest["dimension"])
    return manifest, records, vectors, error


def run(config_path, plan_path, phase4_manifest, output_override=None):
    config, config_digest = load_frozen_config(config_path, phase4_manifest)
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    source = Path(config["output"])
    output = Path(output_override or plan["data_output"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("manifest.json", "corpus.jsonl", "corpus_embeddings.npy")):
        raise FileExistsError("Scale corpus already exists; verify it or choose another --output")
    verify_saved(source)
    base = load_jsonl(source / "corpus.jsonl")
    sizes = plan["measured_sizes"]
    if sizes != sorted(set(sizes)) or sizes[0] != len(base):
        raise ValueError("Measured sizes must be increasing, starting with original corpus size")
    excluded = {q["text"] for split in ("validation", "test") for q in load_jsonl(source / f"{split}_queries.jsonl")}
    os.environ["HF_HOME"] = str((Path(config["cache"]) / "huggingface").resolve())
    os.environ["HF_HUB_CACHE"] = str((Path(config["cache"]) / "models").resolve())
    from datasets import Dataset

    files = list((Path(config["cache"]) / "datasets").rglob(f"{config['dataset']['revision']}/*.arrow"))
    if len(files) != 1:
        raise ValueError("Pinned dataset Arrow cache not uniquely available; no substitute data generated")
    columns = Dataset.from_file(str(files[0])).to_dict()
    rows = ({"anchor": a, "positive": p} for a, p in zip(columns["anchor"], columns["positive"], strict=True))
    records, available = build_nested_records(base, rows, excluded, sizes[-1], config["seed"])
    LOG.info("Real nested corpus: base=%d target=%d available=%d; new query texts excluded", len(base), len(records), available)
    (output / "corpus.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    dimension = config["embedding"]["dimension"]
    vectors = np.lib.format.open_memmap(output / "corpus_embeddings.npy", mode="w+", dtype=np.float32, shape=(len(records), dimension))
    original = np.load(source / "corpus_embeddings.npy", mmap_mode="r", allow_pickle=False)
    vectors[:len(base)] = original
    vectors.flush()
    del columns, rows
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(plan["encoding"]["threads"])
    torch.manual_seed(config["seed"])
    tick = perf_counter()
    model = SentenceTransformer(config["embedding"]["model"], revision=config["embedding"]["revision"],
                                cache_folder=str(Path(config["cache"]) / "models"), device=plan["encoding"]["device"], local_files_only=True)
    model_seconds = perf_counter() - tick
    tick = perf_counter()
    chunk_times = []
    for start in range(len(base), len(records), plan["encoding"]["chunk_size"]):
        stop = min(start + plan["encoding"]["chunk_size"], len(records))
        chunk_start = perf_counter()
        block = np.asarray(model.encode([r["text"] for r in records[start:stop]], batch_size=plan["encoding"]["batch_size"],
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        error = validate_vectors(block, stop - start, dimension)
        vectors[start:stop] = block
        vectors.flush()
        chunk_times.append({"start": start, "stop": stop, "seconds": perf_counter() - chunk_start, "max_norm_error": error})
        LOG.info("Encoded real texts %d/%d; elapsed %.1fs", stop, len(records), perf_counter() - tick)
    encoding_seconds = perf_counter() - tick
    maximum_error = validate_vectors(vectors, len(records), dimension)
    if not np.array_equal(vectors[:len(base)], original):
        raise AssertionError("Original 10k embedding prefix changed")
    save_json(output / "encoding_chunks.json", chunk_times)
    manifest = {"phase": 6, "created_utc": datetime.now(timezone.utc).isoformat(), "base_size": len(base), "target_size": len(records),
                "measured_sizes": sizes, "available_unique_texts": available, "dimension": dimension,
                "seed": config["seed"], "frozen_config_sha256": config_digest, "model": config["embedding"],
                "encoding": plan["encoding"], "model_load_seconds": model_seconds, "new_encoding_seconds": encoding_seconds,
                "max_norm_error": maximum_error, "base_vectors_reused_exactly": True,
                "sampling": "original anchor-only 10k fixed prefix; seeded permutation of first-occurrence unique anchor/positive extension pool; all query strings excluded from new records",
                "distribution_caveat": "The base is anchor-only; extension draws from both columns. Nested growth is reproducible but is not an isolated causal test of N under identical text distributions.",
                "inputs": {str(p): checksum(p) for p in (Path(config_path), Path(plan_path), files[0], source / "manifest.json", source / "corpus.jsonl", source / "corpus_embeddings.npy", source / "validation_queries.jsonl", source / "test_queries.jsonl")},
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "code_sha256": checksum(__file__),
                "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": checksum(output / name)} for name in ("corpus.jsonl", "corpus_embeddings.npy", "encoding_chunks.json")}}
    save_json(output / "manifest.json", manifest)
    LOG.info("Scale corpus saved: %d genuine embeddings; new encoding %.3fs; maximum norm error %g", len(records), encoding_seconds, maximum_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selected_experiment.yaml")
    parser.add_argument("--plan", default="configs/scale.yaml")
    parser.add_argument("--phase4-manifest", default="results/raw/phase4_manifest.json")
    parser.add_argument("--output")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    output = Path(args.output or plan["data_output"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "prepare_scale.log", encoding="utf-8")])
    try:
        if args.verify_only:
            manifest, records, vectors, error = verify_scale_data(output)
            LOG.info("Scale data verification passed: %d rows, max norm error %g", len(records), error)
        else:
            run(args.config, args.plan, args.phase4_manifest, args.output)
    except Exception:
        LOG.exception("Scale data preparation failed; no random embeddings substituted")
        raise


if __name__ == "__main__":
    main()
