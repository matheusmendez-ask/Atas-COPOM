"""Rebuild a local corpus and compare meeting inference on identical vectors.

Run: python -m src.experiments --chunk-sizes 150 --output-dir evaluation/runs
Uses development questions only. No generation or API key is required.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from src.evaluation import load_golden_set, run_evaluation
from src.evaluation_artifacts import runtime_metadata, save_report
from src.generation.answerer import CopomAnswerer
from src.ingestion.collector import BronzeCollector
from src.processing.chunker import AtaChunker
from src.vectorstore.qdrant_manager import QdrantManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[150])
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/runs"))
    args = parser.parse_args()
    records = BronzeCollector.select_current_versions(BronzeCollector().load_all_records())
    if not records or any(size < 30 for size in args.chunk_sizes):
        parser.error("Bronze precisa conter atas; chunk-sizes deve ser >= 30")
    golden = load_golden_set()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for size in args.chunk_sizes:
        overlap = size // 6
        chunker = AtaChunker(chunk_size=size, chunk_overlap=overlap)
        docs = [chunker.process_record(record) for record in records]
        manager = QdrantManager(url=":memory:", collection_name="experiment")
        try:
            summary = manager.sync_documents(docs)
            if summary.failed:
                raise RuntimeError("Indexação incompleta; nenhum placar será publicado")
            snapshot = manager.payload_snapshot()
            metadata = runtime_metadata()
            metadata.update(execution="qdrant_memory", chunk_size=size, chunk_overlap=overlap)
            answerer = CopomAnswerer(qdrant=manager)
            for mode in ("without_meeting_filter", "with_meeting_filter"):
                if mode == "without_meeting_filter":
                    with patch("src.vectorstore.qdrant_manager.infer_meeting", return_value=None):
                        report = run_evaluation(
                            answerer, golden, corpus_payloads=[r["payload"] for r in snapshot]
                        )
                else:
                    report = run_evaluation(
                        answerer, golden, corpus_payloads=[r["payload"] for r in snapshot]
                    )
                path = args.output_dir / f"{run_id}-{size}-{mode}.json"
                save_report(
                    report, golden, snapshot, path, 5, {**metadata, "meeting_inference": mode}
                )
                print(f"{path}: hit@1={report.hit_at(1):.1%}, MRR={report.mrr:.3f}", flush=True)
        finally:
            manager.client.close()


if __name__ == "__main__":
    main()
