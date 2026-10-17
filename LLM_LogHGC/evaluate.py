import argparse
import csv
from collections import Counter
from pathlib import Path


def evaluate(result_file):
    truth_counts = Counter()
    predicted_counts = Counter()
    predicted_truth = {}
    mixed_groups = set()
    total = 0
    failed = 0

    with Path(result_file).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"EventId", "TemplateGroupId"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            total += 1
            truth = (row.get("EventId") or "").strip()
            predicted = (row.get("TemplateGroupId") or "").strip()
            if not truth:
                raise ValueError(f"Missing EventId at data row {total}.")
            truth_counts[truth] += 1
            if not predicted or (row.get("ParseStatus") or "").strip() == "failed":
                failed += 1
                continue

            predicted_counts[predicted] += 1
            if predicted not in predicted_truth:
                predicted_truth[predicted] = truth
            elif predicted_truth[predicted] != truth:
                mixed_groups.add(predicted)

    if total == 0:
        raise ValueError("The CSV contains no log rows; GA is undefined.")
    correct = sum(
        count
        for group, count in predicted_counts.items()
        if group not in mixed_groups and count == truth_counts[predicted_truth[group]]
    )
    return {"total": total, "correct": correct, "failed": failed, "ga": correct / total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate log grouping accuracy (GA).")
    parser.add_argument("result_file", nargs="?", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    try:
        result = evaluate(args.result_file)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"File: {args.result_file.resolve()}")
    print(f"Total logs: {result['total']}")
    print(f"Correctly grouped logs: {result['correct']}")
    print(f"Failed/unassigned logs: {result['failed']}")
    print(f"GA: {result['ga']:.6f} ({result['ga']:.4%})")


if __name__ == "__main__":
    DEFAULT_RESULT = (
        Path(__file__).resolve().parent / "datasets/HDFS/HDFS_parsed_results.csv"
    )
    main()
