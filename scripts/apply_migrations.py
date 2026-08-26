"""
Apply the SQL files in migrations/ to the configured database.

This repository deliberately does NOT use Alembic (see app/models.py's
existing notes on JSON columns): app.main's lifespan calls
Base.metadata.create_all(), which creates missing TABLES but never
ALTERs an existing one. That is fine for a fresh database and wrong for
one created before a column was added -- this script closes exactly that
gap, and nothing more.

Every migration is written to be idempotent (ADD COLUMN IF NOT EXISTS /
CREATE INDEX IF NOT EXISTS), so re-running this is a no-op and is safe
against a database that is already up to date.

Usage (from the repository root, with the virtualenv active):

    python scripts/apply_migrations.py            # apply
    python scripts/apply_migrations.py --check    # report drift, change nothing

Reads DATABASE_URL from the environment/.env. The connection string is
never printed.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"

sys.path.insert(0, str(REPO_ROOT))


def _engine():
    import os

    load_dotenv(REPO_ROOT / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL is not set (environment or .env).")
        raise SystemExit(1)
    # Deliberately never printed -- it contains credentials.
    return create_engine(url)


def report_drift(engine) -> dict:
    """Columns app/models.py declares that the database does not have."""
    from app import models

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    drift = {}

    for model in (models.Resume, models.JobDescription, models.CandidateProfile):
        table = model.__tablename__
        if table not in tables:
            drift[table] = ["<table missing entirely>"]
            continue
        db_columns = {c["name"] for c in inspector.get_columns(table)}
        model_columns = {c.name for c in model.__table__.columns}
        missing = sorted(model_columns - db_columns)
        if missing:
            drift[table] = missing

    return drift


def main() -> int:
    check_only = "--check" in sys.argv
    engine = _engine()

    try:
        before = report_drift(engine)
        if before:
            print("Schema drift detected (columns declared in app/models.py but absent):")
            for table, columns in sorted(before.items()):
                print(f"  {table}: {', '.join(columns)}")
        else:
            print("No schema drift: the database already matches app/models.py.")

        if check_only:
            return 1 if before else 0

        migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not migrations:
            print(f"No .sql files found in {MIGRATIONS_DIR}.")
            return 1

        for path in migrations:
            print(f"\napplying {path.name} ...")
            sql = path.read_text(encoding="utf-8")
            with engine.begin() as connection:
                connection.execute(text(sql))
            print(f"  {path.name}: OK")

        after = report_drift(engine)
        print()
        if after:
            print("STILL DRIFTED after migration:")
            for table, columns in sorted(after.items()):
                print(f"  {table}: {', '.join(columns)}")
            return 1

        print("Database now matches app/models.py.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
