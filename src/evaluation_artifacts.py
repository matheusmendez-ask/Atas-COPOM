"""Portable evaluation artifacts built from the actual index, never from .env dumps."""

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from src.config import settings
from src.evaluation import EvaluationReport
from src.generation.answerer import SYSTEM_PROMPT
from src.vectorstore.qdrant_manager import FUSION_METHOD, PREFETCH_MULTIPLIER


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def runtime_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    source_hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src").rglob("*.py"))
    }
    # Explicit allowlist: credentials, endpoints and .env contents never enter artifacts.
    config = {
        key: getattr(settings, key)
        for key in (
            "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL_NAME",
            "EMBEDDING_DIMENSION",
            "SPARSE_MODEL_NAME",
            "BM25_LANGUAGE",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "TOKENIZER_MODEL",
            "LLM_MODEL",
            "LLM_TEMPERATURE",
        )
    }
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest(source_hashes),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])
        ),
        "config": config,
        "fusion": FUSION_METHOD.value,
        "prefetch_multiplier": PREFETCH_MULTIPLIER,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }


def save_report(
    report: EvaluationReport,
    golden: dict,
    snapshot: list[dict],
    output: Path,
    limit: int,
    metadata: dict | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "metadata": metadata or runtime_metadata(),
        "golden_sha256": digest(golden),
        "golden": golden,
        "corpus_sha256": digest(snapshot),
        "corpus": snapshot,
        "limit": limit,
        "generated": report.generated,
        "metrics": {
            "answerable_count": len(report.answerable),
            "hit_at_1": report.hit_at(1),
            "hit_at_3": report.hit_at(3),
            "hit_at_k": report.hit_at(limit),
            "mrr": report.mrr,
            "random_hit1_full_corpus": report.random_hit1_baseline,
            "provenance": report.provenance_accuracy,
            "facts": report.facts_accuracy,
            "citation_range_validity": report.citation_validity,
            "citation_contract_validity": report._rate(report.results, "citation_contract_ok"),
            "citation_presence_answerable": report._rate(report.answerable, "citations_present"),
            "numeric_support_heuristic": report._rate(report.results, "numeric_support_ok"),
            "refusal": report.refusal_rate,
            "generation_failures": len(report.generation_failures),
        },
        "results": [asdict(r) for r in report.results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a previous experiment.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output


def default_output() -> Path:
    return Path("evaluation/runs") / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
