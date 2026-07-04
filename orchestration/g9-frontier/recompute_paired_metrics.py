#!/usr/bin/env python3
"""Recompute paired CIFAR-100 speedrun metrics from local raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


RUN_DIR_RE = re.compile(r"^seed(?P<pair_index>\d{4})_(?P<slot>[AB])_(?P<role>baseline|candidate)$")
REQUIRED_RUN_FILES = (
    "metrics.csv",
    "config.json",
    "summary.json",
    "warmup.json",
    "repro_metadata.json",
)
SUMMARY_FLOAT_KEYS = (
    "mean_val_acc_delta",
    "mean_time_delta",
    "mean_time_ratio",
)
FLOAT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RunArtifact:
    pair_index: int
    slot: str
    role: str
    path: Path
    metrics: dict[str, Any]
    config: dict[str, Any]
    summary: dict[str, Any]
    warmup: dict[str, Any]
    repro_metadata: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.metrics["seed"])

    @property
    def val_acc(self) -> float:
        return float(self.metrics["val_acc"])

    @property
    def time_seconds(self) -> float:
        return float(self.metrics["time_seconds"])

    @property
    def target_hit(self) -> int:
        return int(float(self.metrics["target_hit"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute paired CIFAR-100 metrics from a local artifact directory."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-validation-source")
    parser.add_argument("--expected-record-mode", choices=("true", "false"))
    parser.add_argument("--expected-paired-seeds", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one measured row, found {len(rows)}")
    row = rows[0]
    for key in ("seed", "val_acc", "target_hit", "time_seconds"):
        if key not in row or row[key] == "":
            raise ValueError(f"{path} is missing required column value {key!r}")
    return row


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def float_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def prepare_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and not force:
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def discover_run_dirs(run_root: Path) -> tuple[list[tuple[int, str, str, Path]], list[str]]:
    discovered: list[tuple[int, str, str, Path]] = []
    warnings: list[str] = []
    if not run_root.exists():
        raise FileNotFoundError(f"run root does not exist: {run_root}")
    if not run_root.is_dir():
        raise NotADirectoryError(f"run root is not a directory: {run_root}")
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_RE.match(child.name)
        if match is None:
            if child.name.startswith("seed"):
                warnings.append(f"ignored seed-like directory with unsupported name: {child.name}")
            continue
        discovered.append(
            (
                int(match.group("pair_index")),
                match.group("slot"),
                match.group("role"),
                child,
            )
        )
    return discovered, warnings


def load_run_artifacts(
    run_dirs: list[tuple[int, str, str, Path]]
) -> tuple[list[RunArtifact], list[str], list[str]]:
    artifacts: list[RunArtifact] = []
    failures: list[str] = []
    warnings: list[str] = []
    seen_names: set[str] = set()

    for pair_index, slot, role, path in run_dirs:
        if path.name in seen_names:
            failures.append(f"duplicate run directory name: {path.name}")
            continue
        seen_names.add(path.name)

        missing = [name for name in REQUIRED_RUN_FILES if not (path / name).is_file()]
        if missing:
            failures.append(f"{path.name} missing required artifacts: {', '.join(missing)}")
            continue

        try:
            metrics = load_metrics(path / "metrics.csv")
            config = load_json(path / "config.json")
            summary = load_json(path / "summary.json")
            warmup = load_json(path / "warmup.json")
            repro_metadata = load_json(path / "repro_metadata.json")
        except Exception as exc:  # noqa: BLE001 - report parse errors without aborting all output.
            failures.append(f"{path.name} parse error: {exc}")
            continue

        artifact = RunArtifact(
            pair_index=pair_index,
            slot=slot,
            role=role,
            path=path,
            metrics=metrics,
            config=config,
            summary=summary,
            warmup=warmup,
            repro_metadata=repro_metadata,
        )
        artifacts.append(artifact)
        warnings.extend(validate_single_artifact(artifact))

    return artifacts, failures, warnings


def validate_single_artifact(artifact: RunArtifact) -> list[str]:
    warnings: list[str] = []
    summary_checks = {
        "val_acc_mean": artifact.val_acc,
        "time_seconds_mean": artifact.time_seconds,
    }
    for key, expected in summary_checks.items():
        if key in artifact.summary and not float_close(float(artifact.summary[key]), expected):
            warnings.append(
                f"{artifact.path.name} summary {key}={artifact.summary[key]} "
                f"does not match metrics.csv {expected}"
            )
    if "target_hit_count" in artifact.summary and int(artifact.summary["target_hit_count"]) != artifact.target_hit:
        warnings.append(
            f"{artifact.path.name} summary target_hit_count={artifact.summary['target_hit_count']} "
            f"does not match metrics.csv {artifact.target_hit}"
        )
    if "seed_base" in artifact.config and int(artifact.config["seed_base"]) != artifact.seed:
        warnings.append(
            f"{artifact.path.name} config seed_base={artifact.config['seed_base']} "
            f"does not match metrics.csv seed {artifact.seed}"
        )
    if artifact.warmup.get("validation_evaluated") is True:
        warnings.append(f"{artifact.path.name} warmup.json reports validation_evaluated=true")
    return warnings


def build_pairs(
    artifacts: list[RunArtifact],
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, RunArtifact]]]:
    grouped: dict[int, dict[str, RunArtifact]] = {}
    failures: list[str] = []
    for artifact in artifacts:
        by_role = grouped.setdefault(artifact.pair_index, {})
        if artifact.role in by_role:
            failures.append(f"seed{artifact.pair_index:04d} has duplicate {artifact.role} artifacts")
        by_role[artifact.role] = artifact

    pairs: list[dict[str, Any]] = []
    for pair_index in sorted(grouped):
        by_role = grouped[pair_index]
        if set(by_role) != {"baseline", "candidate"}:
            failures.append(
                f"seed{pair_index:04d} has incomplete pair roles: {sorted(by_role)}"
            )
            continue
        baseline = by_role["baseline"]
        candidate = by_role["candidate"]
        if baseline.seed != candidate.seed:
            failures.append(
                f"seed{pair_index:04d} baseline seed {baseline.seed} "
                f"does not match candidate seed {candidate.seed}"
            )
            continue
        if baseline.time_seconds <= 0:
            failures.append(f"seed{pair_index:04d} baseline time is not positive")
            continue
        order = (
            f"{baseline.role}/{candidate.role}"
            if baseline.slot == "A"
            else f"{candidate.role}/{baseline.role}"
        )
        pair = {
            "pair_index": pair_index,
            "seed": baseline.seed,
            "order": order,
            "baseline_dir": baseline.path.name,
            "candidate_dir": candidate.path.name,
            "baseline_val_acc": baseline.val_acc,
            "candidate_val_acc": candidate.val_acc,
            "val_acc_delta": candidate.val_acc - baseline.val_acc,
            "baseline_time": baseline.time_seconds,
            "candidate_time": candidate.time_seconds,
            "time_delta": candidate.time_seconds - baseline.time_seconds,
            "time_ratio": candidate.time_seconds / baseline.time_seconds,
            "baseline_target_hit": baseline.target_hit,
            "candidate_target_hit": candidate.target_hit,
        }
        pairs.append(pair)

    by_pair = {f"seed{pair_index:04d}": by_role for pair_index, by_role in grouped.items()}
    return pairs, failures, by_pair


def summarize_role(pairs: list[dict[str, Any]], role: str) -> dict[str, Any]:
    val_key = f"{role}_val_acc"
    time_key = f"{role}_time"
    hit_key = f"{role}_target_hit"
    vals = [float(pair[val_key]) for pair in pairs]
    times = [float(pair[time_key]) for pair in pairs]
    hits = [int(pair[hit_key]) for pair in pairs]
    return {
        "count": len(pairs),
        "val_acc_mean": mean(vals) if vals else None,
        "val_acc_min": min(vals) if vals else None,
        "time_seconds_mean": mean(times) if times else None,
        "time_seconds_min": min(times) if times else None,
        "time_seconds_max": max(times) if times else None,
        "target_hit_count": sum(hits),
    }


def recompute_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    val_deltas = [float(pair["val_acc_delta"]) for pair in pairs]
    time_deltas = [float(pair["time_delta"]) for pair in pairs]
    ratios = [float(pair["time_ratio"]) for pair in pairs]
    return {
        "paired_seeds": len(pairs),
        "baseline": summarize_role(pairs, "baseline"),
        "candidate": summarize_role(pairs, "candidate"),
        "mean_val_acc_delta": mean(val_deltas) if val_deltas else None,
        "mean_time_delta": mean(time_deltas) if time_deltas else None,
        "mean_time_ratio": mean(ratios) if ratios else None,
    }


def load_paired_order(run_root: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    path = run_root / "paired_order.csv"
    if not path.is_file():
        return [], [], ["paired_order.csv is absent; order validation skipped"]
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:  # noqa: BLE001
        return [], [f"paired_order.csv parse error: {exc}"], []
    required = {"pair_index", "seed", "first", "second"}
    if not rows:
        return [], ["paired_order.csv is empty"], []
    missing_columns = required.difference(rows[0])
    if missing_columns:
        return [], [f"paired_order.csv missing columns: {sorted(missing_columns)}"], []
    return rows, [], []


def validate_paired_order(
    rows: list[dict[str, str]],
    pairs: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    by_index = {int(pair["pair_index"]): pair for pair in pairs}
    seen: set[int] = set()
    for row in rows:
        try:
            pair_index = int(row["pair_index"])
            seed = int(row["seed"])
        except ValueError:
            failures.append(f"paired_order.csv has non-integer row: {row}")
            continue
        seen.add(pair_index)
        pair = by_index.get(pair_index)
        if pair is None:
            failures.append(f"paired_order.csv references missing pair_index {pair_index}")
            continue
        expected_order = f"{row['first']}/{row['second']}"
        if pair["order"] != expected_order:
            failures.append(
                f"paired_order.csv pair_index {pair_index} order {expected_order} "
                f"does not match directories {pair['order']}"
            )
        if int(pair["seed"]) != seed:
            failures.append(
                f"paired_order.csv pair_index {pair_index} seed {seed} "
                f"does not match metrics seed {pair['seed']}"
            )
    for pair_index in sorted(set(by_index).difference(seen)):
        failures.append(f"paired_order.csv missing pair_index {pair_index}")
    return failures


def validate_expected_values(
    artifacts: list[RunArtifact],
    paired_summary: dict[str, Any] | None,
    expected_validation_source: str | None,
    expected_record_mode: bool | None,
    expected_paired_seeds: int | None,
    recomputed: dict[str, Any],
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    if expected_validation_source is not None:
        config_values = sorted(
            {
                str(artifact.config.get("validation_source"))
                for artifact in artifacts
                if "validation_source" in artifact.config
            }
        )
        if config_values != [expected_validation_source]:
            failures.append(
                "config validation_source values "
                f"{config_values} do not match expected {expected_validation_source!r}"
            )
        metadata_values = sorted(
            {
                str(artifact.repro_metadata.get("environment", {}).get("C100_VALIDATION_SOURCE"))
                for artifact in artifacts
                if artifact.repro_metadata.get("environment", {}).get("C100_VALIDATION_SOURCE") is not None
            }
        )
        if metadata_values and metadata_values != [expected_validation_source]:
            failures.append(
                "repro_metadata C100_VALIDATION_SOURCE values "
                f"{metadata_values} do not match expected {expected_validation_source!r}"
            )
        if paired_summary is not None and paired_summary.get("validation_source") != expected_validation_source:
            failures.append(
                "paired_summary validation_source "
                f"{paired_summary.get('validation_source')!r} does not match expected "
                f"{expected_validation_source!r}"
            )

    if expected_record_mode is not None:
        if paired_summary is None:
            failures.append("expected record_mode was supplied, but paired_summary.json is absent")
        elif "record_mode" not in paired_summary:
            failures.append("expected record_mode was supplied, but paired_summary.json lacks record_mode")
        elif bool(paired_summary["record_mode"]) is not expected_record_mode:
            failures.append(
                f"paired_summary record_mode={paired_summary['record_mode']!r} "
                f"does not match expected {expected_record_mode!r}"
            )
    elif paired_summary is None or "record_mode" not in paired_summary:
        warnings.append("record_mode unavailable because paired_summary.json is absent or lacks the key")

    if expected_paired_seeds is not None:
        if recomputed["paired_seeds"] != expected_paired_seeds:
            failures.append(
                f"recomputed paired_seeds={recomputed['paired_seeds']} "
                f"does not match expected {expected_paired_seeds}"
            )
        if paired_summary is not None and paired_summary.get("paired_seeds") != expected_paired_seeds:
            failures.append(
                f"paired_summary paired_seeds={paired_summary.get('paired_seeds')} "
                f"does not match expected {expected_paired_seeds}"
            )

    return failures, warnings


def compare_paired_summary(
    paired_summary: dict[str, Any] | None,
    pairs: list[dict[str, Any]],
    recomputed: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    if paired_summary is None:
        return [], ["paired_summary.json is absent; summary comparison skipped"], {}

    failures: list[str] = []
    warnings: list[str] = []
    comparisons: dict[str, Any] = {}

    if paired_summary.get("paired_seeds") != recomputed["paired_seeds"]:
        failures.append(
            f"paired_summary paired_seeds={paired_summary.get('paired_seeds')} "
            f"does not match recomputed {recomputed['paired_seeds']}"
        )

    for key in SUMMARY_FLOAT_KEYS:
        actual = recomputed[key]
        reported = paired_summary.get(key)
        comparisons[key] = {"reported": reported, "recomputed": actual}
        if reported is None or actual is None or not float_close(float(reported), float(actual)):
            failures.append(
                f"paired_summary {key}={reported!r} does not match recomputed {actual!r}"
            )

    reported_pairs = paired_summary.get("pairs")
    if not isinstance(reported_pairs, list):
        failures.append("paired_summary pairs is absent or not a list")
        return failures, warnings, comparisons
    if len(reported_pairs) != len(pairs):
        failures.append(
            f"paired_summary pairs length {len(reported_pairs)} does not match recomputed {len(pairs)}"
        )

    reported_by_seed = {int(pair.get("seed")): pair for pair in reported_pairs if "seed" in pair}
    for pair in pairs:
        reported = reported_by_seed.get(int(pair["seed"]))
        if reported is None:
            failures.append(f"paired_summary lacks pair for seed {pair['seed']}")
            continue
        for key in (
            "baseline_val_acc",
            "candidate_val_acc",
            "val_acc_delta",
            "baseline_time",
            "candidate_time",
            "time_ratio",
        ):
            if key not in reported:
                failures.append(f"paired_summary seed {pair['seed']} lacks {key}")
                continue
            if not float_close(float(reported[key]), float(pair[key])):
                failures.append(
                    f"paired_summary seed {pair['seed']} {key}={reported[key]!r} "
                    f"does not match recomputed {pair[key]!r}"
                )
        if reported.get("order") != pair["order"]:
            failures.append(
                f"paired_summary seed {pair['seed']} order={reported.get('order')!r} "
                f"does not match recomputed {pair['order']!r}"
            )

    return failures, warnings, comparisons


def write_metrics_table(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = (
        "pair_index",
        "seed",
        "order",
        "baseline_dir",
        "candidate_dir",
        "baseline_val_acc",
        "candidate_val_acc",
        "val_acc_delta",
        "baseline_time",
        "candidate_time",
        "time_delta",
        "time_ratio",
        "baseline_target_hit",
        "candidate_target_hit",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow({field: pair[field] for field in fields})


def write_verification(path: Path, verification: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2, sort_keys=True)
        handle.write("\n")
    shutil.move(str(tmp_path), str(path))


def main() -> int:
    args = parse_args()
    expected_record_mode = parse_bool(args.expected_record_mode)
    failures: list[str] = []
    warnings: list[str] = []
    pairs: list[dict[str, Any]] = []
    recomputed = recompute_summary([])
    paired_summary: dict[str, Any] | None = None
    summary_comparisons: dict[str, Any] = {}

    try:
        prepare_out_dir(args.out_dir, args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        run_dirs, discover_warnings = discover_run_dirs(args.run_root)
        warnings.extend(discover_warnings)
        if not run_dirs:
            failures.append("no per-run directories matching seedNNNN_[AB]_(baseline|candidate) found")

        artifacts, load_failures, load_warnings = load_run_artifacts(run_dirs)
        failures.extend(load_failures)
        warnings.extend(load_warnings)

        pairs, pair_failures, _ = build_pairs(artifacts)
        failures.extend(pair_failures)
        recomputed = recompute_summary(pairs)

        order_rows, order_failures, order_warnings = load_paired_order(args.run_root)
        failures.extend(order_failures)
        warnings.extend(order_warnings)
        if order_rows:
            failures.extend(validate_paired_order(order_rows, pairs))

        summary_path = args.run_root / "paired_summary.json"
        if summary_path.is_file():
            try:
                paired_summary = load_json(summary_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"paired_summary.json parse error: {exc}")

        expected_failures, expected_warnings = validate_expected_values(
            artifacts=artifacts,
            paired_summary=paired_summary,
            expected_validation_source=args.expected_validation_source,
            expected_record_mode=expected_record_mode,
            expected_paired_seeds=args.expected_paired_seeds,
            recomputed=recomputed,
        )
        failures.extend(expected_failures)
        warnings.extend(expected_warnings)

        summary_failures, summary_warnings, summary_comparisons = compare_paired_summary(
            paired_summary, pairs, recomputed
        )
        failures.extend(summary_failures)
        warnings.extend(summary_warnings)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"unhandled verifier error: {exc}")

    status = "pass" if not failures else "fail"
    verification = {
        "status": status,
        "run_root": str(args.run_root),
        "out_dir": str(args.out_dir),
        "expected": {
            "validation_source": args.expected_validation_source,
            "record_mode": expected_record_mode,
            "paired_seeds": args.expected_paired_seeds,
        },
        "failures": failures,
        "warnings": warnings,
        "recomputed": recomputed,
        "paired_summary_present": paired_summary is not None,
        "paired_summary_comparison": summary_comparisons,
        "pairs": pairs,
    }
    write_metrics_table(args.out_dir / "metrics-table.tsv", pairs)
    write_verification(args.out_dir / "verification.json", verification)

    print(f"{status.upper()} wrote {args.out_dir / 'verification.json'}")
    print(f"{status.upper()} wrote {args.out_dir / 'metrics-table.tsv'}")
    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    if warnings:
        print("Warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
