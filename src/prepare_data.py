"""Reproducible text splits and real SentenceTransformer embeddings only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

LOG = logging.getLogger(__name__)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_splits(rows, config):
    """Deduplicate exact original strings; retain all selected-anchor pair sources."""
    settings = config["dataset"]
    rng = random.Random(config["seed"])
    anchors = {}
    pairs = []
    invalid = 0
    valid_anchors = 0
    for row_id, row in enumerate(rows):
        anchor, positive = row.get("anchor"), row.get("positive")
        if not isinstance(anchor, str) or not anchor.strip() or not isinstance(positive, str) or not positive.strip():
            invalid += 1
            continue
        valid_anchors += 1
        anchors.setdefault(anchor, None)
        pairs.append((row_id, anchor, positive))
    unique = list(anchors)
    rng.shuffle(unique)
    n = settings["corpus_size"]
    if n <= 0 or len(unique) < n:
        raise ValueError(f"Need {n} unique anchors; found {len(unique)}")
    corpus = [{"doc_id": i, "text": text} for i, text in enumerate(unique[:n])]
    text_to_id = {item["text"]: item["doc_id"] for item in corpus}
    eligible = {}
    for row_id, anchor, positive in pairs:
        if anchor in text_to_id and positive not in text_to_id:
            source = {"pair_id": row_id, "corpus_id": text_to_id[anchor]}
            eligible.setdefault(positive, []).append(source)
    semantic = list(eligible)
    rng.shuffle(semantic)
    counts = [settings[k] for k in ("validation_semantic_queries", "test_semantic_queries", "validation_exact_queries", "test_exact_queries")]
    if any(not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError("Query counts must be nonnegative integers")
    vs, ts, ve, te = counts
    if len(semantic) < vs + ts or ve + te > n:
        raise ValueError(f"Insufficient queries: semantic={len(semantic)}, exact={n}; requested={counts}")
    exact_ids = rng.sample(range(n), ve + te)

    def queries(split, texts, ids):
        result = [{"kind": "semantic", "text": t, "sources": eligible[t]} for t in texts]
        result.extend({"kind": "exact", "text": corpus[i]["text"], "corpus_id": i} for i in ids)
        for i, item in enumerate(result):
            item["query_id"] = f"{split}-{i:04d}"
        return result

    splits = {"corpus": corpus,
              "validation_queries": queries("validation", semantic[:vs], exact_ids[:ve]),
              "test_queries": queries("test", semantic[vs:vs + ts], exact_ids[ve:])}
    stats = {"source_rows": len(pairs) + invalid, "invalid_rows": invalid,
             "valid_anchor_occurrences": valid_anchors, "unique_anchors": len(unique),
             "anchor_duplicate_ratio": 1 - len(unique) / valid_anchors if valid_anchors else 0,
             "eligible_unique_semantic_queries": len(semantic)}
    return splits, stats


def validate(splits, vectors, config):
    settings = config["dataset"]
    corpus = splits["corpus"]
    texts = {r["text"] for r in corpus}
    if len(corpus) != settings["corpus_size"] or len(texts) != len(corpus):
        raise ValueError("Corpus size/uniqueness mismatch")
    if [r["doc_id"] for r in corpus] != list(range(len(corpus))):
        raise ValueError("Unstable corpus IDs")
    va, te = (splits[k] for k in ("validation_queries", "test_queries"))
    if {r["text"] for r in va} & {r["text"] for r in te}:
        raise ValueError("Validation/test text overlap")
    norm_errors = {}
    for name, records in splits.items():
        arr = vectors[name]
        if arr.shape != (len(records), config["embedding"]["dimension"]) or arr.dtype != np.float32:
            raise ValueError(f"Invalid shape/dtype: {name} {arr.shape} {arr.dtype}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite embeddings: {name}")
        error = float(np.max(np.abs(np.linalg.norm(arr, axis=1) - 1))) if len(arr) else 0.0
        if error > 1e-4:
            raise ValueError(f"Non-unit embeddings: {name}, error={error}")
        norm_errors[name] = error
        if name == "corpus":
            continue
        split = name.removesuffix("_queries")
        if [r["query_id"] for r in records] != [f"{split}-{i:04d}" for i in range(len(records))]:
            raise ValueError("Unstable query IDs")
        if len({r["text"] for r in records}) != len(records) or len({r["query_id"] for r in records}) != len(records):
            raise ValueError("Duplicate queries/IDs")
        for kind in ("semantic", "exact"):
            if sum(r["kind"] == kind for r in records) != settings[f"{split}_{kind}_queries"]:
                raise ValueError("Query kind count mismatch")
        for item in records:
            if item["kind"] == "semantic":
                if item["text"] in texts or not item["sources"]:
                    raise ValueError("Invalid semantic query")
                if any(not 0 <= s["corpus_id"] < len(corpus) or s["pair_id"] < 0 for s in item["sources"]):
                    raise ValueError("Invalid source corpus ID")
            elif item["kind"] == "exact":
                if not 0 <= item["corpus_id"] < len(corpus) or corpus[item["corpus_id"]]["text"] != item["text"]:
                    raise ValueError("Invalid exact query")
            else:
                raise ValueError("Unknown query kind")
    return norm_errors


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_saved(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if checksum(output / name) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    splits, vectors = {}, {}
    for name in ("corpus", "validation_queries", "test_queries"):
        splits[name] = [json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        vectors[name] = np.load(output / f"{name}_embeddings.npy", allow_pickle=False)
    errors = validate(splits, vectors, manifest["config"])
    return {"status": "passed", "rows": {k: len(v) for k, v in splits.items()}, "max_norm_error": errors}


def run(config):
    # Set before importing Hub/datasets: those libraries resolve cache defaults on import.
    os.environ["HF_HOME"] = str((Path(config["cache"]) / "huggingface").resolve())
    os.environ["HF_HUB_CACHE"] = str((Path(config["cache"]) / "huggingface" / "hub").resolve())
    os.environ["HF_XET_CACHE"] = str((Path(config["cache"]) / "huggingface" / "xet").resolve())
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import torch

    if config["embedding"]["normalize"] is not True or config["embedding"]["dtype"] != "float32":
        raise ValueError("Phase 1 requires normalized float32 embeddings")
    output, cache = Path(config["output"]), Path(config["cache"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Output already complete; use --verify-only or choose a new output directory")
    start = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    api = HfApi()
    ds = config["dataset"]
    emb = config["embedding"]
    LOG.info("Resolving immutable dataset and model revisions")
    dataset_revision = api.dataset_info(ds["name"], revision=ds["revision"]).sha
    model_revision = api.model_info(emb["model"], revision=emb["revision"]).sha
    LOG.info("Loading dataset revision %s", dataset_revision)
    rows = load_dataset(ds["name"], ds["config"], split=ds["split"], revision=dataset_revision, cache_dir=str(cache / "datasets"))
    splits, stats = build_splits(rows, config)
    stats["data_preparation_seconds"] = time.perf_counter() - start
    LOG.info("Splits: %s", {k: len(v) for k, v in splits.items()})
    model_start = time.perf_counter()
    model = SentenceTransformer(emb["model"], revision=model_revision, device=emb["device"], cache_folder=str(cache / "models"))
    stats["model_load_seconds"] = time.perf_counter() - model_start
    smoke_texts = [splits["corpus"][0]["text"]] * 2
    smoke = np.asarray(model.encode(smoke_texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    if smoke.shape != (2, emb["dimension"]) or not np.isfinite(smoke).all() or not np.allclose(np.linalg.norm(smoke, axis=1), 1, atol=1e-4) or not np.allclose(smoke[0], smoke[1], atol=1e-6):
        raise ValueError("Real model smoke test failed")
    stats["real_model_smoke_test"] = "passed: repeated real text, finite unit vectors, expected dimension"
    vectors = {}
    stats["encoding_seconds"] = {}
    for name, records in splits.items():
        LOG.info("Encoding %s (%s texts)", name, len(records))
        encoding_start = time.perf_counter()
        vectors[name] = np.asarray(model.encode([r["text"] for r in records], batch_size=emb["batch_size"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
        stats["encoding_seconds"][name] = time.perf_counter() - encoding_start
    stats["max_norm_error"] = validate(splits, vectors, config)
    files = []
    for name, records in splits.items():
        text_path = output / f"{name}.jsonl"
        text_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        vector_path = output / f"{name}_embeddings.npy"
        np.save(vector_path, vectors[name], allow_pickle=False)
        files.extend([text_path, vector_path])
    stats["rows"] = {k: len(v) for k, v in splits.items()}
    stats["query_kind_counts"] = {name: {kind: sum(r["kind"] == kind for r in records) for kind in ("semantic", "exact")} for name, records in splits.items() if name != "corpus"}
    stats["artifact_file_bytes"] = {p.name: p.stat().st_size for p in files}
    stats["embedding_dimension"] = emb["dimension"]
    stats["vector_payload_bytes"] = {k: a.nbytes for k, a in vectors.items()}
    stats["total_seconds_before_reload"] = time.perf_counter() - start
    save_json(output / "statistics.json", stats)
    files.append(output / "statistics.json")
    manifest = {"config": config, "dataset_revision": dataset_revision, "model_revision": model_revision,
                "dataset_fingerprint": rows._fingerprint, "source_pair_id_definition": "zero-based row in pinned dataset split",
                "python": sys.version, "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": checksum(p)} for p in files}}
    save_json(output / "manifest.json", manifest)
    result = verify_saved(output)
    save_json(output / "verification.json", result)
    LOG.info("Saved artifact verification: %s", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "prepare.log", encoding="utf-8")])
    try:
        if args.verify_only:
            result = verify_saved(output)
            save_json(output / "verification.json", result)
            LOG.info("Independent reload verification: %s", result)
        else:
            run(config)
    except Exception:
        LOG.exception("Phase 1 failed; no substitute embeddings were generated")
        raise


if __name__ == "__main__":
    main()
