from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_ovr_auc",
]


def load_fold(root: Path, fold: int) -> pd.DataFrame:
    path = root / f"fold_{fold}" / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    required = {"path", "subtype", "case_id", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def get_prob_cols(df: pd.DataFrame, classes: list[str]) -> list[str]:
    cols = [f"p_{c}" for c in classes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    return cols


def verify_same_test(a: pd.DataFrame, b: pd.DataFrame, name: str) -> None:
    key = ["path", "case_id", "subtype", "y_true"]
    aa = a[key].sort_values(key).reset_index(drop=True)
    bb = b[key].sort_values(key).reset_index(drop=True)
    if not aa.equals(bb):
        raise AssertionError(f"Test-set mismatch: {name}")


def calc_metrics(
    df: pd.DataFrame,
    classes: list[str],
) -> dict:
    labels = np.arange(len(classes))
    y = df["y_true"].to_numpy(dtype=int)
    prob = df[get_prob_cols(df, classes)].to_numpy(dtype=float)
    pred = prob.argmax(axis=1)

    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    try:
        out["macro_ovr_auc"] = float(
            roc_auc_score(
                y,
                prob,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except ValueError:
        out["macro_ovr_auc"] = np.nan

    out["confusion_matrix"] = confusion_matrix(
        y,
        pred,
        labels=labels,
    ).tolist()

    return out


def aggregate_cases(
    df: pd.DataFrame,
    classes: list[str],
) -> pd.DataFrame:
    prob_cols = get_prob_cols(df, classes)

    meta = (
        df.groupby("case_id", as_index=False)
        .agg(
            subtype=("subtype", "first"),
            y_true=("y_true", "first"),
        )
    )

    probs = (
        df.groupby("case_id")[prob_cols]
        .mean()
        .reset_index()
    )

    out = meta.merge(
        probs,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = (
        out[prob_cols].to_numpy().argmax(axis=1)
    )
    return out


def stratified_case_indices(
    case_df: pd.DataFrame,
    classes: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for label in range(len(classes)):
        idx = np.flatnonzero(
            case_df["y_true"].to_numpy(dtype=int) == label
        )
        if len(idx) == 0:
            raise ValueError(
                f"No cases for class index {label}."
            )
        pieces.append(
            rng.choice(
                idx,
                size=len(idx),
                replace=True,
            )
        )
    return np.concatenate(pieces)


def metric_from_arrays(
    y: np.ndarray,
    prob: np.ndarray,
    metric: str,
    n_classes: int,
) -> float:
    labels = np.arange(n_classes)
    pred = prob.argmax(axis=1)

    if metric == "accuracy":
        return float(accuracy_score(y, pred))

    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(y, pred)
        )

    if metric == "macro_f1":
        return float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    if metric == "weighted_f1":
        return float(
            f1_score(
                y,
                pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        )

    if metric == "macro_ovr_auc":
        try:
            return float(
                roc_auc_score(
                    y,
                    prob,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            return np.nan

    raise ValueError(metric)


def pooled_case_bootstrap_difference(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    classes: list[str],
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    ref = aggregate_cases(reference, classes)
    cand = aggregate_cases(candidate, classes)

    ref = ref.sort_values("case_id").reset_index(drop=True)
    cand = cand.sort_values("case_id").reset_index(drop=True)

    if not ref["case_id"].equals(cand["case_id"]):
        raise AssertionError(
            "Case-level paired keys do not match."
        )

    n_classes = len(classes)
    ref_prob = ref[get_prob_cols(ref, classes)].to_numpy(float)
    cand_prob = cand[get_prob_cols(cand, classes)].to_numpy(float)
    ref_y = ref["y_true"].to_numpy(int)
    cand_y = cand["y_true"].to_numpy(int)

    observed = (
        metric_from_arrays(
            cand_y, cand_prob, metric, n_classes
        )
        - metric_from_arrays(
            ref_y, ref_prob, metric, n_classes
        )
    )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = stratified_case_indices(
            ref,
            classes,
            rng,
        )
        diffs[b] = (
            metric_from_arrays(
                cand_y[idx],
                cand_prob[idx],
                metric,
                n_classes,
            )
            - metric_from_arrays(
                ref_y[idx],
                ref_prob[idx],
                metric,
                n_classes,
            )
        )

    finite = np.isfinite(diffs)
    diffs = diffs[finite]

    return {
        "observed_difference": float(observed),
        "ci95_low": float(
            np.percentile(diffs, 2.5)
        ),
        "ci95_high": float(
            np.percentile(diffs, 97.5)
        ),
        "n_boot": int(len(diffs)),
        "unit": "case_stratified_paired",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Final matched analysis for BreakHis image-only, "
            "DO-hierarchy, and shuffled-DO models."
        )
    )
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--do", type=Path, required=True)
    ap.add_argument("--shuffled", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    roots = {
        "E0_image_only": args.baseline,
        "E_DO": args.do,
        "E_DO_shuffled": args.shuffled,
    }

    classes = None
    image_rows = []
    case_rows = []
    pooled = {k: [] for k in roots}

    for fold in range(args.n_splits):
        fold_data = {
            name: load_fold(root, fold)
            for name, root in roots.items()
        }

        ref = fold_data["E0_image_only"]

        if classes is None:
            classes = sorted(
                ref["subtype"].astype(str).unique().tolist()
            )

        for name, df in fold_data.items():
            verify_same_test(
                ref,
                df,
                f"{name}/fold_{fold}",
            )
            pooled[name].append(df)

            im = calc_metrics(df, classes)
            image_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    **{
                        m: im[m]
                        for m in METRICS
                    },
                }
            )

            cases = aggregate_cases(
                df,
                classes,
            )
            cm = calc_metrics(
                cases,
                classes,
            )
            case_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "n_cases": len(cases),
                    **{
                        m: cm[m]
                        for m in METRICS
                    },
                }
            )

    image_df = pd.DataFrame(image_rows)
    case_df = pd.DataFrame(case_rows)

    image_df.to_csv(
        args.out / "image_level_fold_metrics.csv",
        index=False,
    )
    case_df.to_csv(
        args.out / "case_level_fold_metrics.csv",
        index=False,
    )

    (
        image_df.groupby("model")[METRICS]
        .agg(["mean", "std"])
        .to_csv(
            args.out / "image_level_aggregate.csv"
        )
    )

    (
        case_df.groupby("model")[METRICS]
        .agg(["mean", "std"])
        .to_csv(
            args.out / "case_level_aggregate.csv"
        )
    )

    pooled = {
        name: pd.concat(
            frames,
            ignore_index=True,
        )
        for name, frames in pooled.items()
    }

    for name in (
        "E_DO",
        "E_DO_shuffled",
    ):
        verify_same_test(
            pooled["E0_image_only"],
            pooled[name],
            name,
        )

    pooled_rows = []
    pooled_cases = {}

    for name, df in pooled.items():
        im = calc_metrics(df, classes)
        cases = aggregate_cases(
            df,
            classes,
        )
        pooled_cases[name] = cases
        cm = calc_metrics(
            cases,
            classes,
        )

        pooled_rows.append(
            {
                "model": name,
                "unit": "image",
                **{
                    m: im[m]
                    for m in METRICS
                },
            }
        )
        pooled_rows.append(
            {
                "model": name,
                "unit": "case",
                **{
                    m: cm[m]
                    for m in METRICS
                },
            }
        )

    pooled_df = pd.DataFrame(
        pooled_rows
    )
    pooled_df.to_csv(
        args.out / "pooled_out_of_fold_metrics.csv",
        index=False,
    )
    pairs = [
        ("E_DO", "E0_image_only"),
        ("E_DO", "E_DO_shuffled"),
        ("E_DO_shuffled", "E0_image_only"),
    ]

    bootstrap_rows = []

    for candidate, reference in pairs:
        for metric in METRICS:
            result = pooled_case_bootstrap_difference(
                pooled[reference],
                pooled[candidate],
                classes,
                metric,
                args.bootstrap,
                args.seed,
            )
            bootstrap_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    **result,
                }
            )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )
    bootstrap_df.to_csv(
        args.out
        / "case_stratified_paired_bootstrap.csv",
        index=False,
    )

    metadata = {
        "classes": classes,
        "n_cases": int(
            len(pooled_cases["E0_image_only"])
        ),
        "n_images": int(
            len(pooled["E0_image_only"])
        ),
        "n_splits": args.n_splits,
        "bootstrap_replicates": args.bootstrap,
        "case_grouped": True,
        "primary_inference_unit": "case",
        "bootstrap_type": (
            "paired stratified case bootstrap"
        ),
        "case_probability_aggregation": (
            "mean image probabilities within each case"
        ),
        "paired_difference": (
            "candidate metric minus reference metric"
        ),
        "pairs": pairs,
    }

    (args.out / "analysis_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== IMAGE-LEVEL MEAN ± SD ===")
    print(
        image_df.groupby("model")[METRICS]
        .agg(["mean", "std"])
    )

    print("\n=== CASE-LEVEL MEAN ± SD ===")
    print(
        case_df.groupby("model")[METRICS]
        .agg(["mean", "std"])
    )

    print("\n=== POOLED OUT-OF-FOLD ===")
    print(
        pooled_df.to_string(index=False)
    )

    print("\n=== CASE-STRATIFIED PAIRED BOOTSTRAP ===")
    print(
        bootstrap_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
