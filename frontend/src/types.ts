// Mirrors the relevant subset of app/schemas.py for POST /job-matches.

export type MatchStatus = "pass" | "fail" | "unknown" | "partial";

export interface ScoreComponent {
  name:
    | "required_skills"
    | "preferred_skills"
    | "experience"
    | "education"
    | "seniority"
    | "project_evidence";
  status: MatchStatus;
  weight: number;
  raw_value: number;
  contribution: number;
}

export interface SkillEvidence {
  matched_required: number;
  total_required: number;
  matched_preferred: number;
  total_preferred: number;
}

export interface MatchEvidence {
  skills: SkillEvidence;
  eligibility: MatchStatus;
  unresolved_notes: string[];
}

export interface MatchResult {
  evidence: MatchEvidence;
  weights_version: string;
  overall_score: number;
  components: ScoreComponent[];
}

export interface RankedJobMatch {
  job_id: number;
  job_title: string;
  result: MatchResult;
  source: string | null;
  job_url: string | null;
  company: string | null;
  location: string | null;
}

export interface JobDiscoveryReport {
  status: "not_requested" | "not_configured" | "ok" | "failed";
  source: string | null;
  query: string | null;
  location: string | null;
  fetched: number;
  newly_ingested: number;
  reused_existing: number;
  failed_to_ingest: number;
  detail: string | null;
}

export interface JobSearchResponse {
  candidate_profile_id: number;
  matches: RankedJobMatch[];
  discovery: JobDiscoveryReport;
}
