"""
Database Connection and Initialization Utilities.
Handles PostgreSQL connection pooling via SQLAlchemy and psycopg2,
and schema initialization.
"""

import os
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as Psycopg2Connection
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Find repository root and load .env
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv_path = REPO_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv(override=True)

# Connection parameters
POSTGRES_USER = os.getenv("POSTGRES_USER", "accident_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "accident_secure_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "accident_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

# Cached SQLAlchemy engine
_engine: Engine | None = None


def get_db_url(driver: str = "psycopg2") -> str:
    """Return the formatted PostgreSQL database connection URL."""
    return f"postgresql+{driver}://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def get_db_engine() -> Engine:
    """Return a singleton SQLAlchemy engine with connection pooling."""
    global _engine
    if _engine is None:
        db_url = get_db_url()
        _engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Test connections for liveness before using
            pool_recycle=3600,   # Recycle connections every hour
        )
        logger.info(f"Initialized SQLAlchemy engine for {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    return _engine


@contextmanager
def get_db_connection() -> Generator[Psycopg2Connection, None, None]:
    """Context manager providing a direct psycopg2 connection with auto-commit/rollback."""
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> dict:
    """
    Test PostgreSQL connectivity and return system metadata (version, schemas, postgis).
    """
    engine = get_db_engine()
    with engine.connect() as conn:
        version_row = conn.execute(text("SELECT version();")).fetchone()
        version = version_row[0] if version_row else "Unknown"

        postgis_check = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis');")
        ).scalar()

        schemas = [
            r[0]
            for r in conn.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('raw', 'staging', 'clean', 'mart', 'ops');")
            ).fetchall()
        ]

    status = {
        "connected": True,
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "database": POSTGRES_DB,
        "user": POSTGRES_USER,
        "version": version,
        "postgis_enabled": bool(postgis_check),
        "existing_schemas": schemas,
    }
    logger.info(f"PostgreSQL connection verified: host={POSTGRES_HOST}, PostGIS={status['postgis_enabled']}, schemas={schemas}")
    return status


def init_schema(schema_file: Path | None = None) -> bool:
    """
    Execute DDL script to create raw, staging, clean, mart, ops schemas and tables.
    """
    if schema_file is None:
        schema_file = Path(__file__).resolve().parent / "schema.sql"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_file}")

    logger.info(f"Reading schema script from {schema_file}")
    with open(schema_file, "r", encoding="utf-8") as f:
        ddl_script = f.read()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl_script)
            logger.info("Executed master schema DDL successfully.")

    # Validate created tables
    engine = get_db_engine()
    with engine.connect() as conn:
        tables = conn.execute(
            text(
                """
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('raw', 'staging', 'clean', 'mart', 'ops')
                ORDER BY table_schema, table_name;
                """
            )
        ).fetchall()

    logger.info(f"Verified {len(tables)} tables across 5 schemas.")
    for schema, table in tables:
        logger.info(f"  - [{schema}] {table}")

    return True


if __name__ == "__main__":
    try:
        status = test_connection()
        print("\n--- PostgreSQL Connection Status ---")
        for k, v in status.items():
            print(f"  {k}: {v}")
        print("\nInitializing Schemas...")
        init_schema()
        print("\nAll 5 Schemas Initialized Successfully!")
    except Exception as e:
        print(f"\n[ERROR] Connection or Initialization Failed: {e}")
