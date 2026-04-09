"""
LexGuard — Smoke Tests
=======================
Lightweight end-to-end tests that exercise the core pipeline functions
using the real contract PDFs in ./data/.  All tests run OFFLINE — no
Snowflake connection or API keys required.

Usage:
    pytest tests/test_smoke.py -v
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# ──────────────────────────────────────────────
# Fixture: real contract data directory
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the real contracts in ./data/."""
    d = PROJECT_ROOT / "data"
    assert d.exists(), f"Data directory not found at {d}"
    pdfs = list(d.glob("*.pdf")) + list(d.glob("*.PDF"))
    assert len(pdfs) > 0, f"No PDF files found in {d}"
    return d


# ──────────────────────────────────────────────
# Test 1: PDF Discovery
# ──────────────────────────────────────────────

def test_pdf_discovery(data_dir):
    """discover_pdfs() should find the real contract PDFs."""
    from ingest import discover_pdfs

    paths = discover_pdfs(str(data_dir))
    assert len(paths) >= 1, "Should discover at least 1 PDF in data/"


# ──────────────────────────────────────────────
# Test 2: Chunk Extraction
# ──────────────────────────────────────────────

def test_chunk_extraction(data_dir):
    """extract_chunks() should produce non-empty chunks with correct keys."""
    from ingest import discover_pdfs, extract_chunks

    paths = discover_pdfs(str(data_dir))
    chunks = extract_chunks(paths)

    assert len(chunks) >= 1, "Real contracts should produce at least 1 chunk"
    required_keys = {"CHUNK_ID", "DOC_NAME", "CHUNK_TEXT", "METADATA", "UPLOAD_TIMESTAMP"}
    for chunk in chunks:
        assert required_keys.issubset(chunk.keys()), f"Missing keys: {required_keys - chunk.keys()}"
        assert len(chunk["CHUNK_TEXT"]) > 0, "Chunk text should not be empty"


# ──────────────────────────────────────────────
# Test 3: DataFrame Construction
# ──────────────────────────────────────────────

def test_build_dataframe(data_dir):
    """build_dataframe() should return a DataFrame with the expected schema."""
    from ingest import discover_pdfs, extract_chunks, build_dataframe

    paths = discover_pdfs(str(data_dir))
    chunks = extract_chunks(paths)
    df = build_dataframe(chunks)

    expected_cols = ["CHUNK_ID", "DOC_NAME", "CHUNK_TEXT", "METADATA", "UPLOAD_TIMESTAMP"]
    assert list(df.columns) == expected_cols
    assert len(df) == len(chunks)


# ──────────────────────────────────────────────
# Test 4: Risk Calculator
# ──────────────────────────────────────────────

def test_risk_calculator_high():
    """Clauses with 'indemnify' should be flagged High Risk."""
    from tools import calculate_risk_level

    result = calculate_risk_level("The tenant shall indemnify the landlord.")
    assert "High Risk" in result


def test_risk_calculator_medium():
    """Clauses with 'penalty' should be flagged Medium Risk."""
    from tools import calculate_risk_level

    result = calculate_risk_level("A penalty of 5% applies for late payment.")
    assert "Medium Risk" in result


def test_risk_calculator_low():
    """Clauses without risk keywords should be Low Risk."""
    from tools import calculate_risk_level

    result = calculate_risk_level("Rent is due on the first of each month.")
    assert "Low Risk" in result


# ──────────────────────────────────────────────
# Test 5: Text Cleaning
# ──────────────────────────────────────────────

def test_clean_text():
    """clean_text() should collapse whitespace and strip edges."""
    from ingest import clean_text

    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("no\nnewlines\there") == "no newlines here"


# ──────────────────────────────────────────────
# Test 6: Deterministic UUIDs
# ──────────────────────────────────────────────

def test_deterministic_uuids(data_dir):
    """Two runs over the same data with the same seed must produce identical chunk IDs."""
    import importlib
    import config as cfg_mod
    from ingest import discover_pdfs, extract_chunks

    paths = discover_pdfs(str(data_dir))

    # Run 1 — reset the UUID RNG
    cfg_mod._uuid_rng.seed(cfg_mod.GLOBAL_SEED)
    chunks_a = extract_chunks(paths)

    # Run 2 — reset again
    cfg_mod._uuid_rng.seed(cfg_mod.GLOBAL_SEED)
    chunks_b = extract_chunks(paths)

    ids_a = [c["CHUNK_ID"] for c in chunks_a]
    ids_b = [c["CHUNK_ID"] for c in chunks_b]
    assert ids_a == ids_b, "Chunk IDs should be identical across deterministic runs"


# ──────────────────────────────────────────────
# Test 7: LocalStore — File Separation
# ──────────────────────────────────────────────

def test_local_store_separation(data_dir, tmp_path):
    """LocalStore.ingest() should create three separate namespaced JSON files."""
    from ingest import discover_pdfs, extract_chunks
    from local_store import LocalStore

    paths = discover_pdfs(str(data_dir))
    chunks = extract_chunks(paths)

    store = LocalStore(working_dir=str(tmp_path / "store"))
    store.ingest(chunks)

    expected_files = [
        "kv_store_documents.json",
        "kv_store_chunks.json",
        "kv_store_clause_index.json",
    ]
    for fname in expected_files:
        fpath = tmp_path / "store" / fname
        assert fpath.exists(), f"Expected file '{fname}' not created"
        # Each file should be valid JSON with content
        import json
        data = json.loads(fpath.read_text())
        assert isinstance(data, dict), f"'{fname}' should contain a JSON object"
        assert len(data) > 0, f"'{fname}' should not be empty"


# ──────────────────────────────────────────────
# Test 8: LocalStore — Clause Search
# ──────────────────────────────────────────────

def test_local_store_search(data_dir, tmp_path):
    """search_clauses() should find matching chunks from the index."""
    from ingest import discover_pdfs, extract_chunks
    from local_store import LocalStore

    paths = discover_pdfs(str(data_dir))
    chunks = extract_chunks(paths)

    store = LocalStore(working_dir=str(tmp_path / "store"))
    store.ingest(chunks)

    # Search for a common legal keyword that should exist in real contracts
    results = store.search_clauses("termination")
    assert len(results) >= 1, "Should find at least 1 chunk containing 'termination'"
    assert any("terminat" in r["text"].lower() for r in results)

    # A keyword NOT in any contract should return nothing
    empty = store.search_clauses("cryptocurrency")
    assert len(empty) == 0, "Should find nothing for an absent keyword"


# ──────────────────────────────────────────────
# Test 9: LocalStore — Deterministic Output
# ──────────────────────────────────────────────

def test_local_store_determinism(data_dir, tmp_path):
    """Two ingestions of the same data must produce identical data (ignoring timestamps)."""
    import json
    import importlib
    import config as cfg_mod
    from ingest import discover_pdfs, extract_chunks
    from local_store import LocalStore

    paths = discover_pdfs(str(data_dir))

    # Run 1
    cfg_mod._uuid_rng.seed(cfg_mod.GLOBAL_SEED)
    chunks_a = extract_chunks(paths)
    store_a = LocalStore(working_dir=str(tmp_path / "store_a"))
    store_a.ingest(chunks_a)

    # Run 2
    cfg_mod._uuid_rng.seed(cfg_mod.GLOBAL_SEED)
    chunks_b = extract_chunks(paths)
    store_b = LocalStore(working_dir=str(tmp_path / "store_b"))
    store_b.ingest(chunks_b)

    def strip_timestamps(data: dict) -> dict:
        """Recursively remove time-dependent fields for comparison."""
        cleaned = {}
        for k, v in data.items():
            if k in ("timestamp", "first_seen"):
                continue
            if isinstance(v, dict):
                cleaned[k] = strip_timestamps(v)
            else:
                cleaned[k] = v
        return cleaned

    # Compare all three files after stripping wall-clock timestamps
    for fname in ["kv_store_documents.json", "kv_store_chunks.json", "kv_store_clause_index.json"]:
        data_a = strip_timestamps(json.loads((tmp_path / "store_a" / fname).read_text()))
        data_b = strip_timestamps(json.loads((tmp_path / "store_b" / fname).read_text()))
        assert data_a == data_b, (
            f"'{fname}' differs between runs — storage is not deterministic"
        )
