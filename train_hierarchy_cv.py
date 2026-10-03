from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from .knowledge.do_semantics import build_semantic_records
from .knowledge.do_hierarchy import (
    build_relation_matrix,
    ontology_aware_loss,
    save_relation_matrix,
    shuffled_relation_matrix,
)
from .utils import seed_everything


class BreakHisDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx

        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        with Image.open(str(row["path"])) as img:
            x = self.transform(img.convert("RGB"))
        y = self.class_to_idx[str(row["subtype"])]
        return x, y


class ImageOnlyResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def class_weights(train_df: pd.DataFrame, classes: list[str]) -> torch.Tensor:
    counts = train_df["subtype"].value_counts().reindex(classes).astype(float)
    if counts.isna().any() or (counts <= 0).any():
        raise ValueError(f"Missing training classes: {counts.to_dict()}")
    w = len(train_df) / (len(classes) * counts.to_numpy())
    return torch.tensor(w, dtype=torch.float32)


def validate_manifest(df: pd.DataFrame, n_splits: int) -> None:
    required = {"path", "subtype", "case_id", "magnification", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    if sorted(df["fold"].unique().tolist()) != list(range(n_splits)):
        raise ValueError("Unexpected fold IDs.")

    if (df.groupby("case_id")["fold"].nunique() > 1).any():
        raise AssertionError("Case leakage detected.")

    classes = set(df["subtype"].astype(str))
    for fold in range(n_splits):
        present = set(
            df.loc[df["fold"] == fold, "subtype"].astype(str)
        )
        if present != classes:
            raise AssertionError(f"Fold {fold} does not contain every subtype.")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    relation_matrix: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    neighbor_mass: float,
    ontology_weight: float,
) -> tuple[float, float, float]:
    model.train()
    losses: list[float] = []
    ce_losses: list[float] = []
    do_losses: list[float] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(x)
            ce = criterion(logits, y)
            do_loss = ontology_aware_loss(
                logits,
                y,
                relation_matrix,
                neighbor_mass=neighbor_mass,
                ordinary_label_smoothing=0.0,
            )
            loss = ce + float(ontology_weight) * do_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce.detach().cpu()))
        do_losses.append(float(do_loss.detach().cpu()))

    return (
        float(np.mean(losses)),
        float(np.mean(ce_losses)),
        float(np.mean(do_losses)),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for x, y in loader:
        logits = model(
            x.to(device, non_blocking=True)
        )
        probs.append(
            torch.softmax(logits, dim=-1).cpu().numpy()
        )
        ys.append(y.numpy())

    if not ys:
        raise RuntimeError("Prediction loader returned no samples.")

    return np.concatenate(ys), np.concatenate(probs)


def metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    try:
        result["macro_ovr_auc"] = float(
            roc_auc_score(y, prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["macro_ovr_auc"] = None
    return result


class HierarchyRelationProxy:
    def __init__(
        self,
        classes: list[str],
        matrix: torch.Tensor,
        direct_links,
        relation_definition: str,
    ) -> None:
        self.classes = classes
        self.matrix = matrix
        self.direct_links = direct_links
        self.relation_definition = relation_definition


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BreakHis ResNet-50 with Human Disease Ontology hierarchical "
            "regularization. Uses the same frozen 3-fold case-grouped CV "
            "as the image-only baseline."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["do_hierarchy", "shuffled_do_hierarchy"],
        default="do_hierarchy",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--neighbor-mass",
        type=float,
        default=0.15,
        help="Probability mass assigned to ontology-supported neighbors.",
    )
    parser.add_argument(
        "--ontology-weight",
        type=float,
        default=0.20,
        help="Weight of the ontology hierarchical loss added to CE.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    df = pd.read_csv(args.manifest)
    validate_manifest(df, args.n_splits)

    classes = sorted(df["subtype"].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but unavailable.")

    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))
        print(
            "CUDA memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB",
        )

    records = build_semantic_records(args.ontology, args.mapping)
    relation = build_relation_matrix(
        classes,
        records,
        shared_ancestor_weight=0.5,
        direct_relation_weight=1.0,
        self_weight=1.0,
    )

    if args.mode == "shuffled_do_hierarchy":
        relation_matrix, permutation = shuffled_relation_matrix(
            relation.matrix,
            args.shuffle_seed,
        )
    else:
        relation_matrix = relation.matrix
        permutation = list(range(len(classes)))

    args.out.mkdir(parents=True, exist_ok=True)

    save_relation_matrix(
        HierarchyRelationProxy(
            classes,
            relation_matrix,
            relation.direct_links,
            relation.relation_definition,
        ),
        args.out / "relation_matrix.json",
    )

    print("Classes:", classes)
    print("Mode:", args.mode)
    print("Neighbor mass:", args.neighbor_mass)
    print("Ontology loss weight:", args.ontology_weight)
    print("Ontology relation matrix:\n", relation_matrix)
    print("Case-grouped outer CV:", args.n_splits)
    print("Gold target used in model forward: false")
    print("Ontology used only as fixed training-time hierarchical supervision.")
    print("Final objective: CE + ontology_weight * DO_hierarchy_loss")

    if args.mode == "shuffled_do_hierarchy":
        print("Relation permutation:", permutation)

    fold_rows: list[dict] = []

    for test_fold in range(args.n_splits):
        seed_everything(args.seed + test_fold)

        train_df = df[df["fold"] != test_fold].copy()
        test_df = df[df["fold"] == test_fold].copy()

        train_cases = set(train_df["case_id"])
        test_cases = set(test_df["case_id"])

        if train_cases & test_cases:
            raise AssertionError(
                f"Case leakage detected in outer fold {test_fold}."
            )

        print(
            f"\n=== Outer Fold {test_fold}: "
            f"train={len(train_df)} test={len(test_df)} ==="
        )
        print(
            f"cases: train={len(train_cases)} "
            f"test={len(test_cases)}"
        )
        print("device=" + str(device))

        common = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.workers > 0,
        }
        generator = torch.Generator().manual_seed(args.seed + test_fold)

        train_loader = DataLoader(
            BreakHisDataset(
                train_df,
                class_to_idx,
                args.image_size,
                True,
            ),
            shuffle=True,
            generator=generator,
            **common,
        )
        test_loader = DataLoader(
            BreakHisDataset(
                test_df,
                class_to_idx,
                args.image_size,
                False,
            ),
            shuffle=False,
            **common,
        )

        model = ImageOnlyResNet50(
            len(classes),
            pretrained=True,
        ).to(device)

        weights = class_weights(train_df, classes).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=0.05,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
        scaler = (
            GradScaler("cuda", enabled=True)
            if device.type == "cuda"
            else None
        )

        rm = relation_matrix.to(device)
        history: list[dict] = []
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            loss, ce_loss, do_loss = train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                relation_matrix=rm,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                neighbor_mass=args.neighbor_mass,
                ontology_weight=args.ontology_weight,
            )
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "ce_loss": ce_loss,
                    "do_hierarchy_loss": do_loss,
                    "lr": lr,
                }
            )

            print(
                f"epoch={epoch:03d} "
                f"loss={loss:.4f} "
                f"ce={ce_loss:.4f} "
                f"do={do_loss:.4f} "
                f"lr={lr:.8f}"
            )

        y, prob = predict(model, test_loader, device)
        m = metrics(y, prob)

        fold_dir = args.out / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(history).to_csv(
            fold_dir / "history.csv",
            index=False,
        )

        prediction = test_df.reset_index(drop=True).copy()
        prediction["y_true"] = y
        prediction["y_pred"] = prob.argmax(axis=1)
        for i, cls in enumerate(classes):
            prediction[f"p_{cls}"] = prob[:, i]
        prediction.to_csv(
            fold_dir / "test_predictions.csv",
            index=False,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "mode": args.mode,
                "relation_matrix": relation_matrix.tolist(),
                "relation_permutation": permutation,
                "ontology_weight": args.ontology_weight,
                "neighbor_mass": args.neighbor_mass,
                "epochs": args.epochs,
            },
            fold_dir / "final.pt",
        )

        metadata = {
            "experiment": "BreakHis ResNet-50 + DO hierarchical regularization",
            "mode": args.mode,
            "test_fold": test_fold,
            "metrics": m,
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "train_images": len(train_df),
            "test_images": len(test_df),
            "fixed_epochs": args.epochs,
            "neighbor_mass": args.neighbor_mass,
            "ontology_weight": args.ontology_weight,
            "gold_target_used_in_model_forward": False,
            "ontology_used_for_training_supervision": True,
            "objective": "CE + ontology_weight * DO_hierarchy_loss",
            "elapsed_sec": time.time() - start,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("Fold test metrics:")
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_ovr_auc",
        ):
            print(f"  {key}: {m[key]}")

        fold_rows.append(
            {
                "fold": test_fold,
                **m,
                "train_cases": len(train_cases),
                "test_cases": len(test_cases),
                "train_images": len(train_df),
                "test_images": len(test_df),
                "evaluation_epoch": args.epochs,
            }
        )

    summary = pd.DataFrame(fold_rows)
    summary.to_csv(
        args.out / "fold_metrics.csv",
        index=False,
    )

    aggregate: dict = {}
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_ovr_auc",
    ):
        values = pd.to_numeric(
            summary[metric],
            errors="coerce",
        ).dropna().to_numpy()
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        }

    aggregate["protocol"] = {
        "n_splits": args.n_splits,
        "fixed_epochs": args.epochs,
        "case_grouped": True,
        "test_untouched_until_final_evaluation": True,
        "ontology_source": Path(args.ontology).name,
        "mapping_source": Path(args.mapping).name,
        "mode": args.mode,
        "neighbor_mass": args.neighbor_mass,
        "ontology_weight": args.ontology_weight,
        "seed": args.seed,
        "shuffle_seed": (
            args.shuffle_seed
            if args.mode == "shuffled_do_hierarchy"
            else None
        ),
        "objective": "CE + ontology_weight * DO_hierarchy_loss",
    }

    (args.out / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print("\n=== Aggregate ===")
    print(summary.to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
