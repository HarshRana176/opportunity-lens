# Phase 2 Annotation Handoff — Project-Aware Relevance Corpus

**For the independent annotator. Read this whole document before opening the data file.**

## Why this handoff exists

The 80-candidate corpus you are about to label was constructed synthetically by an AI assistant (Claude) working with the repository owner. That assistant **cannot** serve as the ground-truth labeller for this corpus: it authored every candidate's content with a specific target rubric tier already in mind (its own corpus-generation notes literally named each candidate's intended category — e.g. "strong project", "tutorial project", "title-only project" — before any text was written). No instruction to "ignore the stratum" can undo that prior knowledge, so an independent human who did **not** write this content must provide the ground-truth labels.

**Methodological note, preserved verbatim for the record:**

> The corpus author did not serve as the ground-truth annotator because the author had prior knowledge of the synthetic candidates' intended strata and target rubric tiers. Ground-truth labels therefore require an independent annotator who is blinded to corpus construction targets.

## What you are given

- **`evaluation/labeling_projects/blinded_worksheet_projects.json`** — 8 job descriptions, 10 candidates each (80 total). For every candidate you see: skills, total experience (months), seniority, education, full employment narrative, full project narrative (title, description, technologies, role, outcome — when present). Every candidate is identified only by an opaque `blind_id` (e.g. `pb-a1b2c3d4`).
- **`evaluation/PROJECT_RUBRIC.md`** — the frozen v1.0 rubric. Read it in full before labelling anything. It defines the 0–3 scale, both relevance dimensions, and every specific evidence type you will encounter (title-only, skills-only, tutorial/forked, implementation, deployment, debugging, evaluation, measurable outcomes, architecture, ownership, adjacent technology, multiple-project corroboration).

## What you are NOT given, and must not seek out

- The candidates' **intended stratum** (there is no such field in the worksheet — it was stripped before this handoff).
- Any **intended/target score** for any candidate.
- **`evaluation/labeling_projects/_blind_key.json`** — the private mapping from `blind_id` back to the corpus's internal candidate id. Do not open it.
- Any **corpus-generation code**. (None exists inside this repository — the corpus was built from a script that was never committed here.)
- Any **matcher, scoring, embedding, or model output** — including `app.matching`/`app.scoring` results for these candidates, and any output of the résumé-project extraction pipeline (`app.candidate_extractor`/`app.llm.project_extraction_chain`). None of that has been computed for this corpus and none should be consulted while labelling.
- Any **prior evaluation result** from Phase 1 or from any other phase of this project.

If any of the above is visible to you by some accident, stop and flag it — do not label around it.

## What to produce

For **every one of the 80 candidates**, independently assign:

**A. `employment_relevance`** — 0, 1, 2, or 3, from `evaluation/PROJECT_RUBRIC.md` §2, using **only** the candidate's employment narrative.

**B. `project_relevance`** — 0, 1, 2, or 3, from `evaluation/PROJECT_RUBRIC.md` §3–4, using **only** the candidate's project narrative.

Score these **completely independently of each other**: a strong project must not pull the employment score up, and strong employment must not pull the project score up. Do not let the candidate's skills list alone justify a score above 0 on either dimension — skills-list presence without narrative is not evidence under this rubric (§4). Do not invent or assume information that is not written in the worksheet.

For each of the two scores, also record:
- **`labeller`** — your name or identifier.
- **`labelled_on`** — the date (YYYY-MM-DD).
- **`note`** — one or two sentences citing the *specific* evidence (a phrase or fact from the candidate's text) that drove your score. "Followed a tutorial to build X" is a note; "seems weak" is not.

The worksheet's `labels` object for each candidate already has this exact shape, all null, ready to fill in:

```json
"labels": {
  "employment_relevance": {"score": null, "labeller": null, "labelled_on": null, "note": null},
  "project_relevance":    {"score": null, "labeller": null, "labelled_on": null, "note": null}
}
```

Fill in `score`/`labeller`/`labelled_on`/`note` for both dimensions, for every candidate, directly in this file (or in a copy of it — either way, the returned file must be a complete, valid copy of `blinded_worksheet_projects.json` with every one of the 160 fields above filled in and nothing else changed).

## What happens after you return the labels

The two sub-scores you provide become **frozen human ground truth**. `combined_score`/`combination_status` are **never** something you assign — they are computed afterward, automatically, from your two frozen scores via the already-approved deterministic table in `evaluation/PROJECT_RUBRIC.md` §5. You do not need to think about the combination rule at all; just score employment and project relevance independently and accurately.

## Scope reminder

This corpus, this rubric, and this handoff are **entirely separate** from the project's existing 360-candidate employment-only evaluation (already labelled, frozen, out of scope here) and from that evaluation's sealed 144-candidate final holdout (which remains sealed and unopened throughout this exercise). Nothing you do here touches either of those.
