"""
Evaluation reporting (Task 8C-1).

Renders harness output as plain text. Reports numbers and their
uncertainty; it does NOT decide anything. Applying the decision gate is
a human judgement made against pre-registered thresholds, and this
module deliberately prints no verdict -- a report that announced a
winner would invite reading the conclusion off a favourable-looking
number.

Every interval is a CLUSTER bootstrap over job descriptions. Where the
number of JDs is small the interval will be wide, and that width is the
honest message, not a defect to be tuned away.

Usage (from the repo root):

    python -m evaluation.report --split dev
    python -m evaluation.report --split dev --real     # needs Ollama
"""
import argparse
from pathlib import Path
from typing import Optional

from evaluation.harness import (
    ArmMetrics,
    build_provider,
    cluster_bootstrap_ci,
    default_arms,
    run,
)
from evaluation.schema import Dataset, UnlabelledDatasetError, load_dataset

DATASET_DIR = Path(__file__).parent / "datasets"


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _fmt_ci(ci: Optional[tuple[float, float]]) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render(results: dict[str, ArmMetrics], split: str, dataset: Dataset) -> str:
    cases = dataset.dev if split == "dev" else dataset.holdout
    candidates = sum(len(c.candidates) for c in cases)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"Task 8C-1 evaluation report -- split: {split}")
    lines.append("=" * 78)
    lines.append(f"job descriptions (clusters): {len(cases)}")
    lines.append(f"candidates                 : {candidates}")

    baseline = results.get("A_baseline")
    if baseline is not None:
        lines.append(f"structured-tied pairs      : {baseline.tied_pair_count}")
    lines.append("")

    lines.append("STATISTICAL RESOLUTION")
    lines.append("-" * 78)
    lines.append(
        "  The independent unit is the JOB DESCRIPTION, not the pair: candidates"
    )
    lines.append(
        "  within one JD are correlated. All intervals below resample JDs."
    )
    if len(cases) < 8:
        lines.append("")
        lines.append(
            f"  ** {len(cases)} clusters is a SMALL sample. Intervals will be wide and"
        )
        lines.append(
            "     a result consistent with 'no difference' should be read as"
        )
        lines.append("     INCONCLUSIVE, not as evidence of no effect. **")
    lines.append("")

    header = (
        f"{'arm':<20} {'nDCG@5':>8} {'nDCG@5 CI':>16} {'P@3':>7} "
        f"{'tau':>7} {'tied acc':>9} {'tied CI':>16} {'elig':>6}"
    )
    lines.append("RESULTS")
    lines.append("-" * 78)
    lines.append(header)
    lines.append("-" * len(header))

    for name, metrics in results.items():
        ndcg_ci = cluster_bootstrap_ci(metrics.per_jd_ndcg_at_5)
        tied_ci = cluster_bootstrap_ci(metrics.per_jd_tied_accuracy)
        lines.append(
            f"{name:<20} {_fmt(metrics.ndcg_at_5):>8} {_fmt_ci(ndcg_ci):>16} "
            f"{_fmt(metrics.precision_at_3, 3):>7} {_fmt(metrics.kendall_tau, 3):>7} "
            f"{_fmt(metrics.tied_pair_accuracy, 3):>9} {_fmt_ci(tied_ci):>16} "
            f"{'OK' if metrics.eligibility_respected else 'FAIL':>6}"
        )

    lines.append("")
    lines.append("GUARDRAILS")
    lines.append("-" * 78)
    violations = {
        name: m.eligibility_violations
        for name, m in results.items()
        if not m.eligibility_respected
    }
    if violations:
        for name, jds in violations.items():
            lines.append(
                f"  DISQUALIFIED {name}: ranked a labelled-ineligible candidate above "
                f"an eligible one in {', '.join(jds)}"
            )
    else:
        lines.append("  eligibility respected by every arm (no ineligible outranked an eligible)")

    if baseline is not None and baseline.tied_pair_count == 0:
        lines.append("")
        lines.append(
            "  NOTE: zero structured-tied pairs -- the primary question cannot be"
        )
        lines.append(
            "  answered by this dataset. Add candidates the 8A score cannot separate."
        )

    lines.append("")
    lines.append("READING THIS REPORT")
    lines.append("-" * 78)
    lines.append("  - No verdict is printed on purpose. Apply the pre-registered gate.")
    lines.append("  - 'tied acc' is the primary signal: 0.5 is a coin flip.")
    lines.append("  - Overlapping intervals mean INCONCLUSIVE, not 'no difference'.")
    lines.append("  - An arm marked FAIL on eligibility is disqualified outright.")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Task 8C-1 evaluation.")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the real Ollama embedding model instead of the offline fake.",
    )
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)

    if not dataset.cases:
        print(f"No evaluation cases found in {args.dataset}.\n")
        print(
            "The corpus is empty. Underscore-prefixed files (e.g. _TEMPLATE.json)\n"
            "are deliberately excluded, so copy the template to a real filename,\n"
            "replace its content, and assign labels:\n"
        )
        print(f"    cp {DATASET_DIR / '_TEMPLATE.json'} {DATASET_DIR / 'jd-001.json'}\n")
        return 2

    cases_in_split = dataset.dev if args.split == "dev" else dataset.holdout
    if not cases_in_split:
        print(
            f"No cases assigned to split {args.split!r} "
            f"(dev: {len(dataset.dev)}, holdout: {len(dataset.holdout)}).\n"
        )
        print("Set \"split\" to \"dev\" or \"holdout\" on each case. Split by JD, never")
        print("by candidate: candidates for one JD are correlated and would leak.")
        return 2

    try:
        results = run(
            dataset,
            split=args.split,
            provider=build_provider(real=args.real),
            arms=default_arms(),
        )
    except UnlabelledDatasetError as exc:
        print("Cannot evaluate: the dataset is not fully labelled.\n")
        print(str(exc))
        print(
            "\nLabels must be assigned by a human, without reference to any "
            "semantic score."
        )
        return 2

    print(render(results, args.split, dataset))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
