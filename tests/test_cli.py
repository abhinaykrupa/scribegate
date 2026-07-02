"""Tests for scribegate.cli (end-to-end pipeline + decision log) and
scribegate.benchmark (markdown report generation from result fixtures)."""

import json
from pathlib import Path

import pytest

from scribegate import cli
from scribegate import benchmark

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"
GOLDEN_DIR = REPO_ROOT / "data" / "golden_notes"

TWO_TRANSCRIPTS = ["glaucoma_01", "cataract_01"]


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------

def test_cli_run_writes_result_json_per_transcript(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(TWO_TRANSCRIPTS, results_dir=results_dir)

    for tid in TWO_TRANSCRIPTS:
        out_path = results_dir / f"{tid}.json"
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["transcript_id"] == tid
        assert "generated_note" in data
        assert "judge_result" in data
        assert "violations" in data
        assert isinstance(data["violations"], list)
        assert data["route"] in ("auto_accept", "review", "regenerate")
        assert "decision_reasons" in data
        assert "timestamps" in data
        assert "generated_at" in data["timestamps"]


def test_cli_run_result_violations_are_plain_dicts(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(TWO_TRANSCRIPTS, results_dir=results_dir)

    data = json.loads((results_dir / "glaucoma_01.json").read_text())
    for v in data["violations"]:
        assert isinstance(v, dict)
        assert "code" in v and "severity" in v and "message" in v and "line_text" in v


def test_cli_run_single_transcript_via_main(tmp_path, capsys):
    results_dir = tmp_path / "results"
    exit_code = cli.main(["run", "--transcript", "glaucoma_02", "--results-dir", str(results_dir)])
    assert exit_code == 0
    assert (results_dir / "glaucoma_02.json").exists()
    captured = capsys.readouterr()
    assert "glaucoma_02" in captured.out
    assert "route=" in captured.out


def test_cli_run_all_processes_every_transcript(tmp_path):
    results_dir = tmp_path / "results"
    all_ids = cli.discover_transcript_ids(TRANSCRIPT_DIR)
    assert len(all_ids) == 20

    exit_code = cli.main(["run", "--all", "--results-dir", str(results_dir)])
    assert exit_code == 0

    written = sorted(p.stem for p in results_dir.glob("*.json") if p.stem != "benchmark")
    assert written == all_ids


def test_cli_requires_transcript_or_all(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "--results-dir", str(tmp_path)])


# ---------------------------------------------------------------------------
# Decision log append behavior
# ---------------------------------------------------------------------------

def test_decision_log_appends_one_line_per_run(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(["glaucoma_01"], results_dir=results_dir)
    log_path = results_dir / cli.DECISION_LOG_NAME
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["transcript_id"] == "glaucoma_01"
    assert "ts" in entry
    assert "aggregate" in entry
    assert "route" in entry
    assert "violation_count" in entry

    # Run again (e.g. re-running the same transcript) -> log grows, never overwritten.
    cli.run(["glaucoma_01"], results_dir=results_dir)
    lines_after = log_path.read_text().strip().splitlines()
    assert len(lines_after) == 2
    # Original first line is untouched.
    assert lines_after[0] == lines[0]


def test_decision_log_accumulates_across_multiple_transcripts(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(TWO_TRANSCRIPTS, results_dir=results_dir)
    log_path = results_dir / cli.DECISION_LOG_NAME
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    ids_logged = {json.loads(line)["transcript_id"] for line in lines}
    assert ids_logged == set(TWO_TRANSCRIPTS)


# ---------------------------------------------------------------------------
# benchmark.py: markdown generation from fixtures
# ---------------------------------------------------------------------------

def test_benchmark_main_writes_markdown_from_results(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(TWO_TRANSCRIPTS, results_dir=results_dir)

    out_path = results_dir / "benchmark.md"
    exit_code = benchmark.main(["--results-dir", str(results_dir)])
    assert exit_code == 0
    assert out_path.exists()

    text = out_path.read_text()
    assert "# ScribeGate Benchmark Report" in text
    assert "Reading this table" in text
    assert "glaucoma_01" in text
    assert "cataract_01" in text
    assert "Table 1: Per visit type" in text
    assert "Table 2: Per transcript" in text


def test_benchmark_load_results_sorted_deterministically(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(["glaucoma_02", "cataract_01"], results_dir=results_dir)
    results = benchmark.load_results(results_dir)
    ids = [r["transcript_id"] for r in results]
    assert ids == sorted(ids)


def test_benchmark_visit_type_table_has_expected_columns(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(TWO_TRANSCRIPTS, results_dir=results_dir)
    results = benchmark.load_results(results_dir)
    table = benchmark._build_visit_type_table(results)
    header = table.splitlines()[0]
    for col in ("Completeness", "Hallucination", "Coding Plausibility", "Terminology", "Mean Aggregate"):
        assert col in header


def test_benchmark_skips_decision_log_and_non_result_files(tmp_path):
    results_dir = tmp_path / "results"
    cli.run(["glaucoma_01"], results_dir=results_dir)
    # decision_log.jsonl exists alongside glaucoma_01.json but must not be
    # picked up as a result (it's .jsonl, not .json, and lacks the expected shape anyway).
    results = benchmark.load_results(results_dir)
    assert len(results) == 1
    assert results[0]["transcript_id"] == "glaucoma_01"
