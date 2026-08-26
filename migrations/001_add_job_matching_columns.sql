-- Migration 001: job-matching columns
--
-- Brings a database created BEFORE the job-matching work in sync with
-- app/models.py. Needed because app.main's lifespan calls
-- Base.metadata.create_all(), which only creates MISSING TABLES -- it
-- never ALTERs an existing one, so a database created earlier keeps the
-- old column set and INSERTs fail with:
--
--   psycopg2.errors.UndefinedColumn:
--   column "projects" of relation "candidate_profiles" does not exist
--
-- SAFETY
--   * Every statement uses IF NOT EXISTS -- running this repeatedly is a
--     no-op, so it is safe to re-run and safe to run on a fresh database
--     that already has the columns.
--   * Every added column is NULLABLE with no default, so existing rows
--     are untouched: no rewrite, no data loss, no backfill needed. A row
--     written before this migration simply reads back NULL, which is
--     exactly what the application already expects (a job created via
--     POST /jobs has no source/job_url; a profile parsed before the
--     projects work has no projects).
--   * No column is dropped, renamed, or retyped.
--
-- Column types mirror app/models.py exactly:
--   candidate_profiles.projects            -> Column(JSON)   -> JSON
--   job_descriptions.source                -> Column(String) -> VARCHAR
--   job_descriptions.external_job_id       -> Column(String) -> VARCHAR
--   job_descriptions.job_url               -> Column(String) -> VARCHAR
--   job_descriptions.company               -> Column(String) -> VARCHAR
--   job_descriptions.location              -> Column(String) -> VARCHAR
--   job_descriptions.posted_at             -> Column(String) -> VARCHAR
--
-- Requires PostgreSQL 9.6+ (for ADD COLUMN IF NOT EXISTS).

BEGIN;

-- Candidate projects (project-aware relevance evidence).
ALTER TABLE candidate_profiles ADD COLUMN IF NOT EXISTS projects JSON;

-- Online job-discovery provenance + deduplication identity.
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS source VARCHAR;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS external_job_id VARCHAR;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS job_url VARCHAR;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS company VARCHAR;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS location VARCHAR;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS posted_at VARCHAR;

-- Indexes matching index=True in app/models.py. These back the
-- (source, external_job_id) deduplication lookup in
-- app.services.find_job_by_source_id.
CREATE INDEX IF NOT EXISTS ix_job_descriptions_source
    ON job_descriptions (source);
CREATE INDEX IF NOT EXISTS ix_job_descriptions_external_job_id
    ON job_descriptions (external_job_id);

COMMIT;
