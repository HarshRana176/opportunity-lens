# Phase 3 Evaluation Report — Project-Aware Relevance Experiment

**Status: FROZEN.** This document records results already produced and independently
verified earlier in the same working session. No evaluation was rerun, no threshold or
weight was changed, and no label/rubric/matcher/scoring file was modified to produce this
report. It is a transcription of prior measured output, not a new analysis.

**Scope.** This report covers only the Phase 2 project-aware relevance experiment (the
80-candidate synthetic corpus at `evaluation/labeling_projects/`). It is separate from,
and does not modify or supersede, the earlier Phase 1 baseline evaluation of the
employment-only 360-candidate v2 corpus.

---

## 1. Objective

Compare three things against a frozen, independently human-labeled 80-candidate corpus:

1. **Arm 1** — the existing, frozen, employment-only deterministic matcher
   (`app.matching.build_match_evidence` + `app.scoring.score_match`, unmodified,
   `DEFAULT_WEIGHTS` v1) vs. human `employment_relevance` labels.
2. **Arm 2** — human `project_relevance` labels, characterized as a signal/headroom
   measure only. The matcher produces no project-relevance prediction, so this is **not**
   a model evaluation.
3. **Arm 3** — `combine_relevance()` (the experimental combination rule in
   `evaluation/schema_projects.py`, §5 of `PROJECT_RUBRIC.md`) applied to the two frozen
   human sub-scores. This is an **oracle upper bound**, not a model prediction.

The purpose was to determine whether the current employment-only matcher already captures
what project evidence would add, or whether there is a measurable, human-rated gap —
without concluding anything about whether an actual project-aware model would close that
gap, since no such model was built or run.

---

## 2. Methodology (measured / procedural)

- **Corpus**: 80 candidates across 8 job descriptions (`jdp-001`..`jdp-008`), each hand-
  authored with employment and project narrative evidence, independently double-labeled
  on two 0–3 relevance dimensions (`employment_relevance`, `project_relevance`) per
  `PROJECT_RUBRIC.md` v1.1.
- **Ground truth provenance**: two independent human annotation passes (Pass A:
  `annotation_package_scored.csv`; Pass B: `independent_style_annotation_scores.csv`),
  reconciled via a documented adjudication process into
  `adjudicated_annotation_scores.csv` and frozen into
  `blinded_worksheet_projects.json`.
- **Scoring path**: `CandidateProfile`/`JobProfile` objects were built *exclusively* from
  each worksheet entry's `candidate`/`job` sub-keys (never from the sibling `labels` key),
  then passed unmodified through `app.matching.build_match_evidence` and
  `app.scoring.score_match` with `DEFAULT_WEIGHTS` (v1: `required_skills=2.0,
  preferred_skills=1.0, experience=1.5, education=1.0, seniority=1.0`).
- **Bucketing**: `overall_score` (a float in [0,1]) was mapped to a 0–3 bucket using the
  same fixed quartile split established in Phase 1 and never re-tuned:
  `[0,.25)→0, [.25,.5)→1, [.5,.75)→2, [.75,1]→3`.
- **Binary-relevant threshold**: score/label ≥ 2, matching the existing convention in
  `evaluation/harness.py` and Phase 1.
- **Splits**: results are reported for **agreed-only (n=59)** — candidates where both
  `employment_status` and `project_status` in the adjudication record equal `"agreed"`,
  i.e. Pass A and Pass B matched independently with no adjudicator input — **adjudicated
  (n=21)**, and **all (n=80)**. 59 + 21 = 80 exactly; verified as a true partition (see §8).
- A read-only `open()` guard was active for the entire run and the entire subsequent
  verification pass, raising immediately on any path containing `_blind_key` or
  `datasets_projects`. It never fired in either run.

---

## 3. Arm 1 — frozen employment-only matcher vs. `employment_relevance` (MEASURED)

| split | n | exact match | MAE | binary(≥2) acc | prec | rec | f1 |
|---|---|---|---|---|---|---|---|
| **agreed (headline)** | 59 | 0.1356 | 1.4576 | 0.1356 | 0.1356 | 1.0000 | 0.2388 |
| adjudicated | 21 | 0.3810 | 0.8571 | 0.3810 | 0.3810 | 1.0000 | 0.5517 |
| all | 80 | 0.2000 | 1.3000 | 0.2000 | 0.2000 | 1.0000 | 0.3333 |

Confusion matrices (pred bucket → human label count):

- **agreed (n=59)**: pred=2 → {human0:35, human1:16, human2:0, human3:0}; pred=3 → {human0:0, human1:0, human2:0, human3:8}
- **adjudicated (n=21)**: pred=2 → {human0:5, human1:8, human2:8, human3:0}; pred=3 → all 0
- **all (n=80)**: pred=2 → {human0:40, human1:24, human2:8, human3:0}; pred=3 → {human0:0, human1:0, human2:0, human3:8}

Human `employment_relevance` distribution (all 80): `{0:40, 1:24, 2:8, 3:8}`.
Arm 1 predicted-bucket distribution (all 80): `{2:72, 3:8}`.

### Structural finding (MEASURED / code-inspection-confirmed)

`app.scoring.score_match` reads only `MatchEvidence.{skills, experience, education,
seniority}`. `MatchEvidence` is built by `app.matching.build_match_evidence`, which never
reads `CandidateEmployment.responsibilities` or `CandidateProfile.projects` — confirmed by
direct inspection of both modules' source, not inference. Consequently:

- Arm 1's predicted bucket is **never 0 or 1** on this corpus and takes only the values
  `{2, 3}`. This mathematically explains every reported metric: since no prediction falls
  below the binary-relevant threshold, TN = FN = 0 in every split, which forces
  precision = accuracy = exact-match-rate identically, and forces recall = 1.0 whenever
  any positive exists in that split — independently of whether the prediction is actually
  correct.
- This collapse is a property of *this corpus's construction* (required skills and
  education were engineered to be present for nearly every candidate, so only
  `experience`/`seniority` vary the score) combined with the matcher's complete
  structural blindness to narrative content — not a general claim about matcher behavior
  on arbitrary real resumes.

---

## 4. Arm 2 — Project Evidence / Oracle Analysis (MEASURED; explicitly NOT a model)

The matcher produces no project-relevance prediction of any kind. This arm characterizes
the human `project_relevance` signal only.

- `project_relevance` distribution (all 80): `{0:26, 1:17, 2:13, 3:24}`.
- **Headroom** — candidates with weak employment evidence (`employment_relevance ≤ 1`)
  but meaningful/strong project evidence (`project_relevance ≥ 2`): **37/80**.
  - Per JD: `{jdp-001:5, jdp-002:5, jdp-003:5, jdp-004:5, jdp-005:5, jdp-006:4, jdp-007:4, jdp-008:4}`
  - Of these 37, **33 are freshers** (see §5).

---

## 5. Fresher analysis (MEASURED)

Fresher = professional months < 12, computed from `employment_history` after excluding
positions whose role title matches `intern|internship|placement|trainee` (regex,
case-insensitive).

- **Freshers: 68/80.**
- Fresher eligibility distribution: `{fail: 68}` — all 68 freshers fail the frozen
  eligibility gate (driven by `total_experience_months`, `required_skills`, `education`,
  `seniority` — untouched by any project field).
- Fresher `employment_relevance`: `{0:40, 1:20, 2:8}`.
- Fresher `project_relevance`: `{0:18, 1:17, 2:13, 3:20}`.
- **Freshers with `project_relevance ≥ 2`: 33/68.**
- **Freshers with `employment_relevance = 0` AND `project_relevance ≥ 2`: 21.**
  Arm 1's predicted bucket for these 21 is uniformly `{2: 21}` — the matcher assigns them
  the same score band as every other fresher, regardless of the strength of their project
  evidence, because it never reads that evidence.

Eligibility × human-relevance cross-tab (all 80; human-relevant = `employment_relevance≥2
OR project_relevance≥2`):

| eligibility | human-relevant | count |
|---|---|---|
| fail | False | 27 |
| fail | True | 45 |
| pass | True | 8 |

No `eligibility=pass, human-relevant=False` row exists (count 0). Projects never altered
eligibility for any candidate — confirmed, not merely asserted (see §8).

---

## 6. Inter-annotator agreement (MEASURED)

Computed directly from the two independent annotation CSVs (Pass A, Pass B), 80 rows each.

- `employment_relevance` A vs. B: **exact agreement = 0.8250**, MAD = 0.1750.
- `project_relevance` A vs. B: **exact agreement = 0.9125**, MAD = 0.0875.

This is the ceiling against which Arm 1's exact-match rate should be read — humans
themselves do not reach 1.0 agreement on `employment_relevance`.

---

## 7. Arm 3 — human-label combination oracle (MEASURED; explicitly NOT model accuracy)

`combined_score = combine_relevance(employment_relevance, project_relevance)`, applied
only to the two **frozen human labels**. No model produced either input, so this arm
measures how far Arm 1 is from a human-judgment upper bound — it is not evidence of any
model's accuracy, current or hypothetical.

- `combined_score` distribution (all 80): `{0:2, 1:25, 2:21, 3:32}`.
- `combination_status` distribution: `{project_compensates:54, employment_sufficient:16,
  employment_only_weak:8, insufficient_evidence:2}`.
- `combined_score > employment_relevance` for **54/80** candidates — i.e. project evidence
  raises the oracle relevance score above employment alone for the majority of the corpus.
- Arm 1 vs. oracle `combined_score` (n=80): exact=0.3625, MAE=0.6625, binary(≥2)
  acc=0.6625, prec=0.6625, rec=1.0000, f1=0.7970 (TP=53, FP=27, TN=FN=0). Confusion
  matrix: pred=2 → {human0:2, human1:25, human2:21, human3:24}; pred=3 → {human0:0,
  human1:0, human2:0, human3:8}.

**Limitation, stated explicitly**: Arm 3 is an *upper bound conditional on a system
reproducing human combined judgment exactly*. It is not a measurement of any implemented
system, and closing the gap between Arm 1 and Arm 3 is not guaranteed to be achievable by
any particular architecture — it only bounds what would be achievable in the best case.

---

## 8. Independent verification (MEASURED)

A second, independently written script (not reusing the first run's in-memory results)
recomputed the following directly from the raw worksheet and CSV files, with the same
`open()` guard active:

- Split sizes (59 agreed + 21 adjudicated = 80 total, confirmed as a true partition by
  reconciling every confusion-matrix cell across the three splits, e.g.
  agreed(35)+adjudicated(5)=all(40) for pred=2/human=0, etc.)
- Exact-match, MAE, and binary acc/prec/rec/F1 for all three Arm-1 splits, and for Arm 3
  vs. Arm 1 — recomputed by direct hand arithmetic over the reported confusion matrices
  and matched to 4 decimal places.
- Headroom (37/80) — recomputed independently: **matched**.
- Fresher count (68/80) and fresher project≥2 count (33/68) — recomputed independently:
  **matched**.
- Fresher employment=0/project≥2 count (21) — recomputed independently: **matched**.
- Inter-annotator agreement (0.8250 / 0.9125) — recomputed independently: **matched**.
- Arm 3 `combined_score` and `combination_status` distributions — recomputed by calling
  `combine_relevance()` fresh over the raw labels: **matched**.
- Confirmed by code inspection that no label field is ever read inside
  `candidate_to_profile()`/`job_to_profile()`, `build_match_evidence`, or `score_match`.
- Confirmed the `open()` guard never fired in either run (no hidden-stratum file was
  opened), and `git status --porcelain` was byte-identical before and after both runs
  (nothing written to disk).

No discrepancy was found between the original run and the independent verification pass.

---

## 9. Leakage / circularity limitations (INTERPRETATION, with measured basis)

- **Corpus authorship contamination.** The 80-candidate corpus was authored by the same
  party (Claude, this assistant) that performed the adjudication of 20 of the 21
  disagreements between Pass A and Pass B, with prior knowledge of intended per-stratum
  target scores. This is a disclosed, structural conflict of interest for the adjudicated
  subset and cannot be fully ruled out even for the agreed subset, since corpus
  *construction* (not labeling) was done by the same party.
- **Why agreed-only (n=59) is the headline.** The 59 agreed-only labels required no
  adjudicator input at all — they are the independent product of two human annotators
  reconciling to the same score without the corpus author's involvement. This is the
  contamination-clean subset and is reported as the primary result. The adjudicated
  (n=21) and all (n=80) splits are reported as secondary, not blended into the headline.
- **Small n.** 80 candidates across 8 JDs; no confidence intervals were computed for this
  phase.
- **Construct mismatch, not necessarily matcher defect.** The matcher was designed to
  measure eligibility-oriented status matching (pass/fail/unknown/partial per dimension),
  not open-ended narrative-quality relevance. Low agreement with `employment_relevance`
  reflects that these are different constructs by design, not proof the matcher is
  miscalibrated for its original purpose.

---

## 10. Exact frozen conditions and files used

**Never modified, accessed for writing, or unblinded:**
- `app/matching.py`, `app/scoring.py` (including `DEFAULT_WEIGHTS`) — read-only, hash/dict
  compared before and after the run: unchanged.
- `evaluation/PROJECT_RUBRIC.md` (v1.1) — governs the frozen human labels; not touched.
- `evaluation/labeling_projects/_blind_key.json` — **never opened** (guarded).
- `evaluation/datasets_projects/*.json` (hidden-stratum fixtures) — **never opened** (guarded).
- No sealed holdout corpus was used in this phase.

**Read-only inputs consumed:**
- `evaluation/labeling_projects/blinded_worksheet_projects.json`
- `evaluation/labeling_projects/adjudicated_annotation_scores.csv`
- `evaluation/labeling_projects/annotation_package_scored.csv`
- `evaluation/labeling_projects/independent_style_annotation_scores.csv`
- `app.matching.build_match_evidence`, `app.scoring.score_match`, `app.schemas.*`
  (unmodified production code)
- `evaluation.schema_projects.combine_relevance` (unmodified evaluation-only code)

**Execution**: two scratchpad-only scripts (`phase3_eval.py`, and an independent
`phase3_verify.py`), never committed to the repository, each with a `builtins.open` guard
active for their full duration. No commit was made at any point in this phase.

---

## 11. Conclusion

### What this experiment demonstrates (measured)

- On this 80-candidate corpus, the frozen employment-only matcher's `overall_score` is
  structurally blind to both employment-narrative quality and all project evidence — it
  scores purely from coded fields (`total_experience_months`, skills, education,
  seniority), never from `responsibilities` or `projects` text.
- This blindness has a measurable cost: the matcher's agreement with human
  `employment_relevance` judgments (13.6% exact match, agreed-only) is far below the
  human-human agreement ceiling (82.5%) — a substantially larger gap than annotator noise
  alone would explain.
- A large fraction of freshers with weak-or-no employment relevance nonetheless have
  meaningful-to-strong human-rated project evidence (33/68 freshers with
  `project_relevance≥2`; 21 with `employment_relevance=0` specifically), and the matcher's
  score is identical for all of them regardless of this variation.
- A human-defined combination rule that incorporates project evidence (Arm 3, oracle only)
  would rate the majority of the corpus (54/80) more relevant than employment evidence
  alone does — indicating human-rated project signal exists that the current architecture
  has no path to use.

### What this experiment does NOT demonstrate (interpretation)

- It does **not** demonstrate that implementing a project-aware scoring feature would
  improve real-world match quality — Arm 2 and Arm 3 are not implemented models; Arm 3 is
  an upper bound conditional on perfectly reproducing human judgment, which no built
  system has been shown to achieve.
- It does **not** establish generalizability beyond this synthetic, author-constructed
  80-candidate corpus; the disclosed authorship-contamination risk applies most directly
  to the adjudicated subset but is not fully excludable for the corpus as a whole.
- It does **not** show the matcher is broken relative to its original design purpose
  (deterministic eligibility scoring) — the measured gap reflects a difference between
  what the matcher was built to measure and what this rubric asked humans to judge.
- It does **not** provide a quantified estimate of how much accuracy a real project-aware
  feature would gain — no such feature was built or measured; only headroom and an oracle
  ceiling were measured.
