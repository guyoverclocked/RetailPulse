"""Reproducible data ingestion.

Reads raw Rossmann files (real download or synthetic sample), records a SHA-256
manifest, joins store metadata, writes partitioned Parquet plus DuckDB views,
and validates the result. Idempotent: rerunning produces identical curated
tables.

Raw files are NEVER committed — see the data README and license boundaries.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from retailpulse.config import get_config, resolve_path
from retailpulse.data.validate import validate_curated, validate_store, validate_train


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of one ingestion run."""

    run_id: str
    manifest_path: Path
    curated_path: Path
    n_rows: int
    n_stores: int
    date_min: str
    date_max: str


def sha256(path: Path) -> str:
    """Hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_raw_dir(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load raw train/store CSVs from a directory (real or synthetic)."""
    train_path = raw_dir / "train.csv"
    store_path = raw_dir / "store.csv"
    if not train_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            f"raw files missing in {raw_dir}: expected train.csv and store.csv. "
            "Run 'make sample-data' for synthetic data or see data/README.md for "
            "the Kaggle download instructions."
        )
    train = pl.read_csv(train_path, try_parse_dates=True)
    store = pl.read_csv(store_path)
    return train, store


def _find_raw_dir() -> Path:
    """Locate the raw input: real raw dir if present, else the synthetic sample."""
    cfg = get_config()
    raw_dir = resolve_path(cfg.data.paths["raw_dir"])
    sample_dir = resolve_path("data/sample")
    if (raw_dir / "train.csv").exists():
        return raw_dir
    if (sample_dir / "train.csv").exists():
        return sample_dir
    raise FileNotFoundError(
        f"no raw data in {raw_dir} and no synthetic sample in {sample_dir}. "
        "Run 'make sample-data' first."
    )


def _write_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    source: str,
    train_path: Path,
    store_path: Path,
    n_rows: int,
    n_stores: int,
    date_min: str,
    date_max: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "files": {
            "train.csv": {
                "path": str(train_path),
                "sha256": sha256(train_path),
                "rows": n_rows,
            },
            "store.csv": {
                "path": str(store_path),
                "sha256": sha256(store_path),
                "rows": n_stores,
            },
        },
        "curated": {
            "rows": n_rows,
            "stores": n_stores,
            "date_min": date_min,
            "date_max": date_max,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def run_ingestion() -> IngestionResult:
    """Full ingestion: raw -> validated -> curated Parquet + DuckDB views."""
    cfg = get_config()
    raw_dir = _find_raw_dir()

    train, store = _read_raw_dir(raw_dir)

    train = validate_train(train)
    store = validate_store(store)

    # Join store metadata onto each store-day (left join: every day must match
    # a store; missing stores are a validation failure).
    curated = train.join(store, on="Store", how="left")
    if curated["StoreType"].null_count() > 0:
        raise ValueError("store metadata missing for some store IDs in train data")
    curated = validate_curated(curated)

    # Partitioned Parquet by year, sorted for deterministic output. Polars
    # excludes the partition column from the data files automatically.
    curated = curated.sort(["Date", "Store"]).with_columns(pl.col("Date").dt.year().alias("year"))
    curated_path = resolve_path(cfg.data.paths["curated_dir"])
    curated_path.mkdir(parents=True, exist_ok=True)
    curated.write_parquet(
        curated_path / "curated.parquet",
        partition_by=["year"],
    )

    # DuckDB view for SQL analytics.
    duckdb_file = resolve_path(cfg.data.paths["duckdb_file"])
    duckdb_file.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_file))
    glob_path = str(curated_path / "curated.parquet" / "**" / "*.parquet").replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW curated AS SELECT * FROM read_parquet('{glob_path}')")
    con.close()

    run_id = uuid.uuid4().hex[:12]
    dates = curated["Date"]
    manifest_path = resolve_path(cfg.data.paths["manifest_file"])
    _write_manifest(
        manifest_path,
        run_id=run_id,
        source=str(raw_dir),
        train_path=raw_dir / "train.csv",
        store_path=raw_dir / "store.csv",
        n_rows=curated.height,
        n_stores=curated["Store"].n_unique(),
        date_min=str(dates.min()),
        date_max=str(dates.max()),
    )

    return IngestionResult(
        run_id=run_id,
        manifest_path=manifest_path,
        curated_path=curated_path,
        n_rows=curated.height,
        n_stores=curated["Store"].n_unique(),
        date_min=str(dates.min()),
        date_max=str(dates.max()),
    )


def load_curated() -> pl.DataFrame:
    """Load the curated Parquet table into memory (partition column dropped)."""
    cfg = get_config()
    curated_path = resolve_path(cfg.data.paths["curated_dir"])
    if not curated_path.exists() or not list(curated_path.glob("curated.parquet/**/*.parquet")):
        raise FileNotFoundError("curated data missing; run 'retailpulse ingest' first")
    df = pl.read_parquet(curated_path / "curated.parquet" / "**" / "*.parquet")
    if "year" in df.columns:
        df = df.drop("year")
    return df.sort(["Date", "Store"])


def run_validation() -> None:
    """Standalone validation CLI entry: re-validate curated data and fail hard."""
    from retailpulse.data.validate import all_columns_declared

    df = load_curated()
    validate_curated(df)
    missing = all_columns_declared()
    if missing:
        raise ValueError(f"columns not declared in availability table: {missing}")
