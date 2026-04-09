"""
LexGuard — Phase 5: Data Ingestion Pipeline
=============================================
Reads PDF contracts from ./data/, chunks them page-by-page using PyMuPDF,
packages metadata as JSON, and bulk-uploads to Snowflake's CONTRACT_CHUNKS
table via write_pandas.

Usage:
    python ingest.py
"""

import os
import re
import json
import uuid
import glob
import datetime

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import config  # Seeds all randomness on import
from local_store import LocalStore

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

load_dotenv()

# Path to PDF contracts — update if your PDFs live elsewhere
PDF_DIR = os.path.join(".", "data")

# Snowflake credentials (loaded from .env)
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_USER    = os.getenv("SNOW_USER")
SNOW_PASS    = os.getenv("SNOW_PASS")
SNOW_WH      = os.getenv("SNOW_WH")
SNOW_DB      = os.getenv("SNOW_DB")
SNOW_SCHEMA  = os.getenv("SNOW_SCHEMA")

# Target table
TARGET_TABLE = "CONTRACT_CHUNKS"

# Minimum character threshold — pages with fewer chars trigger OCR fallback
MIN_TEXT_CHARS = 50


# ──────────────────────────────────────────────
# Helper utilities (reused from Phase 3 notebook)
# ──────────────────────────────────────────────

def clean_text(s: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def discover_pdfs(pdf_dir: str) -> list[str]:
    """Return a sorted list of PDF file paths in the given directory."""
    paths = sorted(
        glob.glob(os.path.join(pdf_dir, "*.pdf"))
        + glob.glob(os.path.join(pdf_dir, "*.PDF"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No PDF files found in '{os.path.abspath(pdf_dir)}'. "
            "Please ensure your contracts are placed in that folder."
        )
    return paths


# ──────────────────────────────────────────────
# Core chunking logic
# ──────────────────────────────────────────────

def extract_chunks(pdf_paths: list[str]) -> list[dict]:
    """
    Read each PDF page-by-page with PyMuPDF, extract text, and return a list
    of chunk dictionaries ready for DataFrame conversion.

    Each chunk dict contains:
        CHUNK_ID          – UUID string
        DOC_NAME          – PDF filename
        CHUNK_TEXT         – Cleaned page text
        METADATA          – JSON string with page_number, modality, char_count
        UPLOAD_TIMESTAMP  – ISO-8601 timestamp
    """
    chunks: list[dict] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for pdf_path in tqdm(pdf_paths, desc="📄 Processing PDFs", unit="file"):
        doc_name = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)

        for page_idx in tqdm(
            range(len(doc)),
            desc=f"   ↳ {doc_name[:45]}…",
            unit="page",
            leave=False,
        ):
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text")
            text = clean_text(raw_text)
            modality = "text"

            # OCR fallback for scanned / image-heavy pages
            if len(text) < MIN_TEXT_CHARS:
                try:
                    from pdf2image import convert_from_path
                    import pytesseract

                    images = convert_from_path(
                        pdf_path,
                        first_page=page_idx + 1,
                        last_page=page_idx + 1,
                        dpi=300,
                    )
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        text = clean_text(ocr_text)
                        modality = "ocr"
                except Exception as e:
                    tqdm.write(
                        f"  ⚠  OCR fallback failed for {doc_name} "
                        f"page {page_idx + 1}: {e}"
                    )

            # Skip empty pages
            if not text:
                continue

            metadata = {
                "page_number": page_idx + 1,
                "modality": modality,
                "char_count": len(text),
            }

            chunks.append(
                {
                    "CHUNK_ID": config.get_seeded_uuid(),
                    "DOC_NAME": doc_name,
                    "CHUNK_TEXT": text,
                    "METADATA": json.dumps(metadata),
                    "UPLOAD_TIMESTAMP": now,
                }
            )

        doc.close()

    return chunks


# ──────────────────────────────────────────────
# DataFrame construction
# ──────────────────────────────────────────────

def build_dataframe(chunks: list[dict]) -> pd.DataFrame:
    """Convert the list of chunk dicts into a Pandas DataFrame matching the
    CONTRACT_CHUNKS table schema."""
    df = pd.DataFrame(
        chunks,
        columns=[
            "CHUNK_ID",
            "DOC_NAME",
            "CHUNK_TEXT",
            "METADATA",
            "UPLOAD_TIMESTAMP",
        ],
    )
    return df


# ──────────────────────────────────────────────
# Snowflake upload
# ──────────────────────────────────────────────

def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create and return an authenticated Snowflake connection."""
    missing = [
        name
        for name, val in {
            "SNOW_ACCOUNT": SNOW_ACCOUNT,
            "SNOW_USER": SNOW_USER,
            "SNOW_PASS": SNOW_PASS,
            "SNOW_WH": SNOW_WH,
            "SNOW_DB": SNOW_DB,
            "SNOW_SCHEMA": SNOW_SCHEMA,
        }.items()
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing Snowflake env vars: {', '.join(missing)}. "
            "Please set them in your .env file."
        )

    conn = snowflake.connector.connect(
        account=SNOW_ACCOUNT,
        user=SNOW_USER,
        password=SNOW_PASS,
        insecure_mode=True,  # Bypass OCSP certificate validation issues
    )

    # Auto-create database, schema, and table if they don't exist
    cur = conn.cursor()
    try:
        # Use custom role if provided, otherwise stick with user's default
        snow_role = os.getenv("SNOW_ROLE")
        if snow_role:
            cur.execute(f"USE ROLE {snow_role}")

        cur.execute(f"USE WAREHOUSE {SNOW_WH}")
        # cur.execute(f"CREATE DATABASE IF NOT EXISTS {SNOW_DB}")
        cur.execute(f"USE DATABASE {SNOW_DB}")
        # cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOW_SCHEMA}")
        cur.execute(f"USE SCHEMA {SNOW_SCHEMA}")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
                CHUNK_ID     STRING,
                DOC_NAME     STRING,
                CHUNK_TEXT   STRING,
                METADATA     VARIANT,
                UPLOAD_TIMESTAMP STRING
            )
        """)
        print("✅ Snowflake context ready (database / schema / table verified).")
    except snowflake.connector.errors.ProgrammingError as e:
        conn.close()
        raise RuntimeError(
            f"Failed to set up Snowflake objects: {e}\n\n"
            "Please verify in your Snowflake UI that:\n"
            f"  1. Warehouse '{SNOW_WH}' exists and your role has USAGE on it\n"
            f"  2. Your user has ACCOUNTADMIN role access"
        ) from e
    finally:
        cur.close()

    return conn


def upload_to_snowflake(df: pd.DataFrame) -> None:
    """Upload a DataFrame to the CONTRACT_CHUNKS table using write_pandas."""
    print("\n☁️  Connecting to Snowflake …")
    conn = get_snowflake_connection()

    try:
        print(f"⬆️  Uploading {len(df):,} rows to {SNOW_DB}.{SNOW_SCHEMA}.{TARGET_TABLE} …")

        # write_pandas handles staging + COPY INTO under the hood
        success, num_chunks, num_rows, _ = write_pandas(
            conn,
            df,
            TARGET_TABLE,
            database=SNOW_DB,
            schema=SNOW_SCHEMA,
        )

        if success:
            print(f"✅ Upload complete — {num_rows:,} rows written in {num_chunks} chunk(s).")
        else:
            print("❌ Upload reported failure. Please check Snowflake for details.")
    finally:
        conn.close()
        print("🔒 Snowflake connection closed.")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  LexGuard — Phase 5: Data Ingestion Pipeline")
    print("=" * 60)

    # 1. Discover PDFs
    pdf_paths = discover_pdfs(PDF_DIR)
    print(f"\n📂 Found {len(pdf_paths)} PDF(s) in '{os.path.abspath(PDF_DIR)}':\n")
    for p in pdf_paths:
        print(f"   • {os.path.basename(p)}")

    # 2. Extract & chunk
    print()
    chunks = extract_chunks(pdf_paths)
    print(f"\n🧩 Total chunks extracted: {len(chunks)}")

    if not chunks:
        print("⚠  No chunks produced — nothing to upload. Exiting.")
        return

    # 3. Persist to separated local store (inspired by HyperGraphRAG)
    store = LocalStore(working_dir=config.HYPERPARAMS["working_dir"])
    store.ingest(chunks)
    print(f"\n💾 Local store saved to '{config.HYPERPARAMS['working_dir']}'")

    # 4. Build DataFrame
    df = build_dataframe(chunks)
    print(f"\n📊 DataFrame shape: {df.shape}")
    print(df.head(3).to_string(index=False, max_colwidth=60))

    # 4. Upload to Snowflake
    upload_to_snowflake(df)

    print("\n🎉 Ingestion pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
