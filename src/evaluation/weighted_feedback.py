"""
Weighted aggregation of user feedback from model_feedback.csv.

Different user groups have different domain expertise, so their ratings are
weighted accordingly.  An administrator's opinion on answer accuracy is far more
valuable than a first-year student's.

User type weights
-----------------
    admin          10  — dean's office staff, faculty administration
    phd             5  — PhD students
    master          3  — master's students
    student_senior  2  — 2nd / 3rd year bachelor's students
    student_junior  1  — 1st year bachelor's students
    (unknown)       1  — fallback for missing / unrecognised user_type

Usage
-----
    export PYTHONPATH=src
    python -m evaluation.weighted_feedback
    python -m evaluation.weighted_feedback --feedback-csv path/to/model_feedback.csv
    python -m evaluation.weighted_feedback --output-csv results/weighted_summary.csv
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEEDBACK_CSV = "src/data/feedback/model_feedback.csv"

USER_TYPE_WEIGHTS: dict[str, int] = {
    "admin": 10,
    "phd": 5,
    "master": 3,
    "student_senior": 2,
    "student_junior": 1,
}
DEFAULT_WEIGHT = 1

# Rating criteria expected inside the `ratings` JSON column.
# Each is a 1-5 integer (or absent).
RATING_CRITERIA = ["usefulness", "accuracy", "conciseness"]


def _weight(user_type: str) -> int:
    return USER_TYPE_WEIGHTS.get((user_type or "").strip().lower(), DEFAULT_WEIGHT)


def _parse_ratings(raw: str) -> dict[str, float]:
    """Parse the `ratings` JSON column into a dict of criterion → float."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if v is not None}
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def load_feedback(path: str) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        logger.warning("Feedback file not found: %s", path)
        return rows
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    logger.info("Loaded %d feedback rows from %s", len(rows), path)
    return rows


def aggregate(rows: list[dict]) -> dict:
    """
    Compute weighted and unweighted means for all rating signals,
    broken down by user_type, language, and model.

    Returns a nested dict:
        {
          "overall": {"rating": {...}, "usefulness": {...}, ...},
          "by_user_type": {user_type: {...}},
          "by_language":  {lang: {...}},
          "by_model":     {model: {...}},
        }

    Each inner dict has keys: weighted_mean, unweighted_mean, count, total_weight.
    """

    def _accumulator():
        return {"w_sum": 0.0, "w_count": 0.0, "u_sum": 0.0, "u_count": 0}

    def _add(acc, value, weight):
        acc["w_sum"] += value * weight
        acc["w_count"] += weight
        acc["u_sum"] += value
        acc["u_count"] += 1

    def _finalise(acc) -> dict:
        return {
            "weighted_mean": acc["w_sum"] / acc["w_count"] if acc["w_count"] else None,
            "unweighted_mean": acc["u_sum"] / acc["u_count"] if acc["u_count"] else None,
            "count": acc["u_count"],
            "total_weight": acc["w_count"],
        }

    signals = ["rating"] + RATING_CRITERIA

    overall = {s: _accumulator() for s in signals}
    by_user_type: dict[str, dict] = defaultdict(lambda: {s: _accumulator() for s in signals})
    by_language: dict[str, dict] = defaultdict(lambda: {s: _accumulator() for s in signals})
    by_model: dict[str, dict] = defaultdict(lambda: {s: _accumulator() for s in signals})

    for row in rows:
        user_type = row.get("user_type", "")
        language = row.get("language", "unknown") or "unknown"
        model = row.get("model", "unknown") or "unknown"
        w = _weight(user_type)

        # Overall numeric rating (single value)
        raw_rating = row.get("rating", "")
        if raw_rating not in ("", None):
            try:
                val = float(raw_rating)
                for acc in [overall["rating"], by_user_type[user_type]["rating"],
                             by_language[language]["rating"], by_model[model]["rating"]]:
                    _add(acc, val, w)
            except ValueError:
                pass

        # Per-criterion ratings from JSON column
        criteria_vals = _parse_ratings(row.get("ratings", ""))
        for criterion in RATING_CRITERIA:
            if criterion in criteria_vals:
                val = criteria_vals[criterion]
                for acc in [overall[criterion], by_user_type[user_type][criterion],
                             by_language[language][criterion], by_model[model][criterion]]:
                    _add(acc, val, w)

    return {
        "overall": {s: _finalise(overall[s]) for s in signals},
        "by_user_type": {ut: {s: _finalise(accs[s]) for s in signals} for ut, accs in by_user_type.items()},
        "by_language": {lang: {s: _finalise(accs[s]) for s in signals} for lang, accs in by_language.items()},
        "by_model": {model: {s: _finalise(accs[s]) for s in signals} for model, accs in by_model.items()},
    }


def _fmt(val) -> str:
    return f"{val:.4f}" if val is not None else "—"


def print_report(results: dict) -> None:
    signals = ["rating"] + RATING_CRITERIA

    print("\n" + "=" * 60)
    print("WEIGHTED FEEDBACK SUMMARY")
    print("=" * 60)

    print("\n── Overall ──────────────────────────────────────────────")
    print(f"  {'Signal':<16} {'Weighted':>10} {'Unweighted':>12} {'N':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6}")
    for s in signals:
        d = results["overall"][s]
        if d["count"] == 0:
            continue
        print(f"  {s:<16} {_fmt(d['weighted_mean']):>10} {_fmt(d['unweighted_mean']):>12} {d['count']:>6}")

    for section_key, section_label in [
        ("by_user_type", "By user type"),
        ("by_language", "By language"),
        ("by_model", "By model"),
    ]:
        section = results[section_key]
        if not section:
            continue
        print(f"\n── {section_label} {'─' * (54 - len(section_label))}")
        for group, group_data in sorted(section.items()):
            w_label = f"(weight={_weight(group)})" if section_key == "by_user_type" else ""
            print(f"\n  {group} {w_label}")
            print(f"    {'Signal':<16} {'Weighted':>10} {'Unweighted':>12} {'N':>6}")
            for s in signals:
                d = group_data[s]
                if d["count"] == 0:
                    continue
                print(f"    {s:<16} {_fmt(d['weighted_mean']):>10} {_fmt(d['unweighted_mean']):>12} {d['count']:>6}")

    print()


def save_csv(results: dict, path: str) -> None:
    """Save a flat CSV with one row per (section, group, signal)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals = ["rating"] + RATING_CRITERIA

    rows = []
    for section_key in ("overall", "by_user_type", "by_language", "by_model"):
        section = results[section_key]
        if section_key == "overall":
            items = [("overall", section)]
        else:
            items = list(section.items())
        for group, group_data in items:
            for s in signals:
                d = group_data[s]
                rows.append({
                    "section": section_key,
                    "group": group,
                    "signal": s,
                    "weighted_mean": d["weighted_mean"],
                    "unweighted_mean": d["unweighted_mean"],
                    "count": d["count"],
                    "total_weight": d["total_weight"],
                })

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "group", "signal", "weighted_mean",
                         "unweighted_mean", "count", "total_weight"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved summary to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute weighted feedback metrics.")
    parser.add_argument("--feedback-csv", default=FEEDBACK_CSV)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="If provided, save summary table to this CSV path.",
    )
    args = parser.parse_args()

    rows = load_feedback(args.feedback_csv)
    if not rows:
        logger.warning("No feedback rows found — nothing to aggregate.")
        return

    results = aggregate(rows)
    print_report(results)

    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
