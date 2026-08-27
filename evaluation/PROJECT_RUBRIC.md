# Phase 2 Project-Aware Relevance Rubric — v1.1 (FROZEN)

> **v1.1 change log (2026-08-26).** Two clarifications were approved AFTER the two
> annotation passes were completed, to resolve conflicts the v1.0 text could not
> settle. Neither changes any tier definition; both only make an already-implied
> boundary explicit. The pre-clarification text is preserved verbatim at
> `evaluation/PROJECT_RUBRIC_v1.0_pre_clarification.md`.
>
> 1. **§4 adjacent technology** — adjacency presupposes at least one shared required
>    technology; a project sharing none is capped at 1 (see the clause itself).
> 2. **§2 vs §4 precedence** — where §2's substantive tier description and §4's
>    operational strength-signal shortcut disagree, **§2 governs**: supervised,
>    team-reviewed work with limited ownership/depth scores **2**, even when it
>    carries two or more technical strength signals. §4's "≥2 signals + direct
>    alignment → 3" is a shortcut for unsupervised/owned work, not an override of
>    the ownership and depth requirement in §2 tier 3.
>
> Neither clarification references hidden strata, intended target scores, or any
> model/matcher output.

**Status: v1.1 — versioned, with an auditable freeze history.**

- **v1.0 was frozen during corpus construction and throughout both annotation passes.** It was fixed before the blinded worksheet was created and was not modified while any label was being assigned, so every label in both original annotation passes was assigned under v1.0 and can be traced to it.
- **v1.0 is preserved unchanged** at `evaluation/PROJECT_RUBRIC_v1.0_pre_clarification.md`, byte-for-byte as it stood during annotation.
- **v1.1 records two explicitly approved post-annotation clarifications** (see the change log above) made only after both passes were complete and their disagreements were audited. It **supersedes only the two affected interpretations** — the adjacency boundary in §4 and the §2-vs-§4 precedence question. Every tier definition, every other rule, and the combination rule in §5 are unchanged from v1.0.
- **Neither clarification was derived from candidate content, hidden strata, intended target scores, or any model/matcher output** — each resolves an internal ambiguity in the v1.0 text itself.

Any future revision must again be a new, separately versioned document (v1.2, v2.0, ...) accompanied by a preserved copy of the superseded version, never a silent in-place edit, so that every label ever assigned remains traceable to the exact rubric version in force when it was assigned.

This rubric governs the **Phase 2 project-aware relevance experiment** only. It is entirely separate from, and does not modify, the existing employment-only relevance rubric already used for the frozen 360-candidate corpus.

**Relevance is not eligibility.** Nothing in this document affects, overrides, or feeds into `total_experience_months`, `required_skills`, `preferred_skills`, `education`, or `seniority` as scored by the frozen `app.matching`/`app.scoring` path. A candidate can score relevance = 3 under this rubric while the same candidate fails the JD's experience-eligibility requirement — that is an expected, not contradictory, outcome. Projects can never manufacture professional experience months or alter any eligibility-affecting field.

---

## 1. The scale

| Score | Meaning |
|---|---|
| **0** | No meaningful evidence |
| **1** | Weak / basic / superficial evidence |
| **2** | Meaningful hands-on evidence |
| **3** | Strong / direct evidence |

Applied **twice, independently**, per candidate per JD:
- `employment_relevance` — scored from `employment_history[*].responsibilities` only.
- `project_relevance` — scored from `projects[*]` only.

A labeller assigns each score **without reference to the other** and **without reference to any model, embedding, or LLM output.** `combined_relevance` is never assigned by a human — it is always computed afterward by the deterministic rule in §5.

---

## 2. Employment relevance — 0/1/2/3

| Score | Definition |
|---|---|
| 0 | No employment records, or none whose responsibilities relate — lexically or substantively — to the JD's required skills/responsibilities |
| 1 | Technology/domain appears, but only administrative, peripheral, or observational involvement ("attended reviews", "coordinated scheduling", "updated documentation") — no demonstrated hands-on ownership |
| 2 | Real hands-on work matching ≥1 JD responsibility, but limited depth, scale, or ownership, or only partial alignment to the JD's core duties |
| 3 | Demonstrated ownership, measurable outcomes, architecture/design decisions, debugging/incident response, or scaled impact directly matching JD responsibilities |

## 3. Project relevance — 0/1/2/3

| Score | Definition |
|---|---|
| 0 | No projects, **or a project's only connection to the JD is a technology named in its title/tags with no substantive description of what was built or done**, or the description is generic/templated with no real content |
| 1 | Technology used superficially: tutorial followed, starter repo cloned/forked, described entirely as following someone else's instructions — real but basic/shallow exposure, no measurable outcome, no architecture/debugging/deployment described |
| 2 | Real implementation work using JD-relevant technology for ≥1 JD responsibility category — candidate describes what they built/decided, has some technical scope, but lacks scale, production concerns, or full ownership |
| 3 | Substantive engineering matching JD responsibilities — architecture decisions, concrete debugging/problem-solving, measurable results, deployment/production concerns, or scope clearly mapping onto the JD's core duties |

**Correction from the design-approval pass — read this before labelling any title-only case:** title/tag-only evidence with no substantive description is **0**, not 1. Only genuine (if shallow) tutorial/basic-exposure evidence — where the candidate did *something*, even trivial — is 1. If the only signal is a project name containing a technology and nothing else describing the work, score it 0.

---

## 4. Definitions of the specific evidence types you will encounter

Read every one of these before labelling. They apply to *both* employment and project relevance unless stated otherwise.

- **Title-only** — the technology or JD-relevant term appears in a project's *title* (or in a skills/tag list attached to the project) but the *description* contains no account of what was actually done. → **Score 0.** A title is not evidence of work; it is a label someone chose.
- **Skills-only** — a technology appears in the candidate's top-level `skills` list but is never mentioned in any employment or project narrative. → **Not evidence for this rubric at all — score 0 on whichever dimension (employment/project) you're scoring, since there is no narrative to read.** Skills-list presence is already captured by the existing, separate `required_skills`/`preferred_skills` dimensions; do not credit it again here.
- **Tutorial / forked project** — the description explicitly or implicitly indicates the candidate followed an external course/tutorial, or forked/cloned a starter repository without describing independent extension. → **Score 1**, never higher, regardless of how many technologies are named.
- **Implementation** — the candidate describes building something themselves: writing the logic, designing a component, making a technical decision. → A genuine implementation description is a **strength signal**; on its own it supports a **floor of 2**.
- **Deployment** — the candidate describes shipping/running the thing somewhere real (containerized, deployed to a cloud service, published, released) rather than only running it locally once. → **Strength signal**, contributes toward 3 when combined with another strength signal.
- **Debugging** — the candidate describes finding and fixing a specific problem (a bug, a performance issue, a failure mode), not just "debugged issues" in the abstract. Vague claims ("did lots of debugging") without any specific problem named are **not** a strength signal. → **Strength signal** when concrete.
- **Evaluation** — the candidate describes measuring or testing the result (accuracy, test coverage, a benchmark, a comparison before/after). → **Strength signal.**
- **Measurable outcomes** — a concrete number, metric, or comparison tied to the work (e.g. "cut latency from 800ms to 140ms", "reduced crash rate by 95%"). Vague claims of impact with no number are **not** measurable outcomes. → **Strength signal.**
- **Architecture** — the candidate describes a structural/design decision (how components fit together, why a particular approach was chosen), not just "used X". → **Strength signal.**
- **Ownership** — explicit language indicating the candidate drove the work themselves ("designed", "built", "owned", "led") as opposed to passive/observational language ("attended", "helped", "was part of", "shadowed"). Passive language alone caps a score at **1**, regardless of what technologies are named.
- **Scoring rule for strength signals:** presence of **any one** genuine strength signal (implementation, deployment, debugging, evaluation, measurable outcome, or architecture) with real hands-on description → floor of **2**. Presence of **≥2 distinct** strength-signal categories, **and** direct alignment to a specific JD responsibility (not just shared vocabulary) → **3**. **Clarification (v1.1, 2026-08-26):** this is a shortcut, not an override. Where it conflicts with §2's substantive tier description, **§2 governs** — supervised, team-reviewed work with limited ownership or depth scores **2** even when two or more strength signals are present.
- **Adjacent technology** — the narrative demonstrates a JD *responsibility* (e.g. "built a REST API", "deployed a containerized service") but using a technology that is not the one the JD requires (e.g. Go instead of Python, Vue instead of React, Flutter instead of native Android). → **Capped at 2.** Technology match and responsibility match are both required for 3; a responsibility match alone, on the wrong stack, is meaningful but not strong-direct evidence for *this* JD. **Clarification (v1.1, 2026-08-26):** adjacent technology presupposes at least one of the JD's required technologies is still present in the project. A project sharing **none** of the JD's required technologies is not 'adjacent' — §3 tier 2's "using JD-relevant technology" is not satisfied, so such a project is scored on responsibility evidence alone and **capped at 1**.
- **Multiple-project corroboration** — more than one project independently supports the same JD responsibility category. → A single project already scoring 3 needs no corroboration. A project scoring 2 may be upgraded to **3** *only* if **≥2 independently qualifying projects** (each already scoring ≥2 on its own, before considering corroboration) support the *same* responsibility. Repetition of *weak* (0/1) evidence across multiple projects does **not** upgrade the score — quantity of weak evidence is not strength.

---

## 5. Combined relevance — the experimental combination rule

**This is a hypothesis being tested in Phase 2, not an established truth.** It is applied mechanically, in code (`evaluation.schema_projects.combine_relevance`), never by a human labeller, and always *after* both `employment_relevance` and `project_relevance` are frozen. `employment_score`, `project_score`, `combined_score`, and `combination_status` are **always retained separately** — the combined value is derived, never a replacement for its inputs, specifically so an alternative combination rule can be tested later against the same 80 labels without relabelling anything.

The rule, exhaustively, for every `(employment, project)` pair:

| employment ↓ / project → | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **3** | 3, employment_sufficient | 3, employment_sufficient | 3, employment_sufficient | 3, employment_sufficient |
| **2** | 2, employment_sufficient | 2, employment_sufficient | 2, corroborated | 3, corroborated |
| **1** | 1, employment_only_weak | 1, corroborated_weak | 2, project_compensates | 3, project_compensates |
| **0** | 0, insufficient_evidence | 1, project_compensates | 2, project_compensates | 3, project_compensates |

No employment records at all (`employment = None`, distinct from `employment = 0`): combined is driven entirely by `project` (status `project_compensates`, or `insufficient_evidence` if project is also 0/absent). No projects at all (`project = None`): combined is driven entirely by `employment` (`employment_sufficient` if ≥2, `employment_only_weak` if 1, `insufficient_evidence` if 0).

**On conflict:** `conflicting_unresolved` and `conflicting_resolved_favor_stronger` remain valid status values in the schema, but the deterministic function above **never emits them** — a genuine content-level contradiction between the two narratives is a qualitative judgement no bare pair of integers can safely infer, and an earlier draft of this rule that tried to infer it from `|employment − project| ≥ 2` was found to be wrong (it incorrectly flagged clean cases like employment=3/project=0 as conflicting). Those statuses are reserved for a human reviewer to assign manually, with reasoning recorded in the label's `note` field, never inferred automatically.

---

## 6. What the labeller sees and does not see

**Visible:** JD (title, seniority, required/preferred skills, experience/education requirements, responsibilities), candidate skills, candidate total experience months and seniority, candidate education, full employment narrative, full project narrative (title, description, technologies, role, outcome when present).

**Not visible:** stratum, candidate source/original ID, any expected label, any model/matcher score, any pre-computed employment/project/combined score.

---

## 7. Provenance

Every label carries `score`, `labeller`, `labelled_on`, `note` — see `evaluation/schema_projects.py::EmploymentRelevanceLabel` / `ProjectRelevanceLabel`. `combined_score`/`combination_status` are never hand-entered; they are always the output of `combine_relevance()` applied to the two frozen sub-scores.
