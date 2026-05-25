"""
VINS-Fusion-SAM result analysis.

This version can run directly on the zipped results in All_Datasets-results.

Examples:
    python analyze_dataset.py --results_root All_Datasets-results
    python analyze_dataset.py --results_root All_Datasets-results --name corridor1_512_16

VIO CSVs are treated as the single source for both trajectory and telemetry.
The older explicit metric CSV arguments are still supported as overrides.
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.switch_backend("Agg")


VIO_COLS = [
    "timestamp",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "vel_x",
    "vel_y",
    "vel_z",
]

ROS_VIO_COLS = {
    "timestamp": "%time",
    "pos_x": "field.pose.pose.position.x",
    "pos_y": "field.pose.pose.position.y",
    "pos_z": "field.pose.pose.position.z",
    "quat_x": "field.pose.pose.orientation.x",
    "quat_y": "field.pose.pose.orientation.y",
    "quat_z": "field.pose.pose.orientation.z",
    "quat_w": "field.pose.pose.orientation.w",
}

METRIC_COLS = [
    "timestamp",
    "frame_id",
    "frame_processing_time",
    "feature_tracking_time",
    "optimization_time",
    "sam_invoked",
    "sam_start_time",
    "sam_end_time",
    "sam_duration",
    "cpu_usage",
    "gpu_usage",
    "covariance_value",
    "gate_blocked",
    "cov_threshold",
    "mask_iou",
]


@dataclass
class DatasetInputs:
    name: str
    cov_zip: Path | None = None
    timed_zip: Path | None = None
    baseline_csv: Path | None = None
    baseline_vio: Path | None = None
    timed_csv: Path | None = None
    cov_csv: Path | None = None
    timed_vio: Path | None = None
    cov_vio: Path | None = None
    timed_masks: Path | None = None
    cov_masks: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="Dataset name, or omit to process all discovered datasets")
    parser.add_argument("--results_root", default="All_Datasets-results", help="Folder containing dataset zip results")
    parser.add_argument("--timed_csv", default=None)
    parser.add_argument("--cov_csv", default=None)
    parser.add_argument("--baseline_csv", default=None)
    parser.add_argument("--timed_vio", default=None)
    parser.add_argument("--cov_vio", default=None)
    parser.add_argument("--baseline_vio", default=None)
    parser.add_argument("--timed_masks", default=None, help="Folder of timed .npy masks")
    parser.add_argument("--cov_masks", default=None, help="Folder of covariance .npy masks")
    parser.add_argument("--out", default="figures", help="Output folder for figures")
    return parser.parse_args()


def is_numeric(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def read_zip_member(zip_path: Path, member_name: str) -> bytes:
    with ZipFile(zip_path) as zipf:
        return zipf.read(member_name)


def csv_members(zip_path: Path) -> list[str]:
    with ZipFile(zip_path) as zipf:
        return [name for name in zipf.namelist() if name.lower().endswith(".csv")]


def mask_members(zip_path: Path, folder_name: str) -> list[str]:
    prefix = folder_name.rstrip("/") + "/"
    with ZipFile(zip_path) as zipf:
        return [
            name
            for name in zipf.namelist()
            if name.startswith(prefix) and Path(name).name.startswith("mask_") and name.endswith(".npy")
        ]


def choose_csv_member(zip_path: Path, prefer_vio: bool) -> str | None:
    members = csv_members(zip_path)
    if prefer_vio:
        vio_members = [name for name in members if "vio" in Path(name).stem.lower()]
        return vio_members[0] if vio_members else (members[0] if members else None)

    non_vio_members = [name for name in members if "vio" not in Path(name).stem.lower()]
    return non_vio_members[0] if non_vio_members else (members[0] if members else None)


def load_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def load_headered_csv_bytes(data: bytes) -> pd.DataFrame | None:
    if not data.strip():
        return None

    first_line = data.splitlines()[0].decode("utf-8", errors="replace")
    if not any(char.isalpha() for char in first_line):
        return None
    return pd.read_csv(io.BytesIO(data))


def load_metrics_from_vio_path(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None

    df = load_headered_csv_bytes(path.read_bytes())
    if df is None:
        return None
    return df


def load_metrics_from_zip(zip_path: Path | None) -> pd.DataFrame | None:
    if zip_path is None:
        return None
    member = choose_csv_member(zip_path, prefer_vio=True)
    if member is None:
        return None

    return load_headered_csv_bytes(read_zip_member(zip_path, member))


def load_explicit_metrics(metrics_path: Path | None, vio_path: Path | None) -> pd.DataFrame | None:
    metrics_df = load_csv(metrics_path)
    if metrics_df is not None:
        return metrics_df
    return load_metrics_from_vio_path(vio_path)


def normalize_ros_vio(df: pd.DataFrame) -> pd.DataFrame | None:
    if not all(col in df.columns for col in ROS_VIO_COLS.values()):
        return None

    out = pd.DataFrame({target: pd.to_numeric(df[source], errors="coerce") for target, source in ROS_VIO_COLS.items()})
    for col in ("vel_x", "vel_y", "vel_z"):
        out[col] = np.nan
    return out.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)


def normalize_headerless_vio(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None

    # ROS odometry exported without the header:
    # %time, seq, stamp, frame_id, child_frame_id, pos_x, pos_y, pos_z, qx, qy, qz, qw, ...
    if df.shape[1] >= 12 and not is_numeric(df.iloc[0, 3]):
        out = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                "pos_x": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
                "pos_y": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
                "pos_z": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
                "quat_x": pd.to_numeric(df.iloc[:, 8], errors="coerce"),
                "quat_y": pd.to_numeric(df.iloc[:, 9], errors="coerce"),
                "quat_z": pd.to_numeric(df.iloc[:, 10], errors="coerce"),
                "quat_w": pd.to_numeric(df.iloc[:, 11], errors="coerce"),
                "vel_x": np.nan,
                "vel_y": np.nan,
                "vel_z": np.nan,
            }
        )
        return out.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)

    # Compact VINS output:
    # timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z[,]
    if df.shape[1] >= 11:
        out = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
                "pos_x": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                "pos_y": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
                "pos_z": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
                "quat_x": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
                "quat_y": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
                "quat_z": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
                "quat_w": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
                "vel_x": pd.to_numeric(df.iloc[:, 8], errors="coerce"),
                "vel_y": pd.to_numeric(df.iloc[:, 9], errors="coerce"),
                "vel_z": pd.to_numeric(df.iloc[:, 10], errors="coerce"),
            }
        )
        return out.dropna(subset=["pos_x", "pos_y", "pos_z"]).reset_index(drop=True)

    return None


def load_vio(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None

    first_line = path.read_text(errors="replace").splitlines()[0] if path.stat().st_size else ""
    if "field.pose.pose.position.x" in first_line:
        return normalize_ros_vio(pd.read_csv(path))

    df = pd.read_csv(path, header=None)
    return normalize_headerless_vio(df)


def load_vio_from_zip(zip_path: Path | None) -> pd.DataFrame | None:
    if zip_path is None:
        return None
    member = choose_csv_member(zip_path, prefer_vio=True)
    if member is None:
        return None

    data = read_zip_member(zip_path, member)
    if not data.strip():
        return None

    first_line = data.splitlines()[0].decode("utf-8", errors="replace")
    if "field.pose.pose.position.x" in first_line:
        return normalize_ros_vio(pd.read_csv(io.BytesIO(data)))

    df = pd.read_csv(io.BytesIO(data), header=None)
    return normalize_headerless_vio(df)


def resize_mask(mask: np.ndarray, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    if mask.shape[:2] == size:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8))
    img = img.resize((size[1], size[0]), Image.NEAREST)
    return np.array(img).astype(bool)


def frame_id_from_mask_name(name: str) -> int:
    stem = Path(name).stem
    return int(stem.split("_", 1)[1])


def load_masks_from_folder(folder: str | Path | None) -> list[tuple[int, np.ndarray]]:
    if folder is None:
        return []
    folder = Path(folder)
    if not folder.exists():
        return []

    masks: list[tuple[int, np.ndarray]] = []
    for path in folder.glob("mask_*.npy"):
        mask = np.load(path, allow_pickle=False)
        masks.append((frame_id_from_mask_name(path.name), resize_mask(mask)))
    masks.sort(key=lambda item: item[0])
    return masks


def load_masks_from_zip(zip_path: Path | None, folder_name: str) -> list[tuple[int, np.ndarray]]:
    if zip_path is None:
        return []

    masks: list[tuple[int, np.ndarray]] = []
    with ZipFile(zip_path) as zipf:
        for member in mask_members(zip_path, folder_name):
            mask = np.load(io.BytesIO(zipf.read(member)), allow_pickle=False)
            masks.append((frame_id_from_mask_name(member), resize_mask(mask)))
    masks.sort(key=lambda item: item[0])
    return masks


def compute_iou_series(masks: list[tuple[int, np.ndarray]]) -> list[tuple[int, float]]:
    ious: list[tuple[int, float]] = []
    for idx in range(1, len(masks)):
        fid_curr, mask_curr = masks[idx]
        _, mask_prev = masks[idx - 1]
        intersection = np.logical_and(mask_prev, mask_curr).sum()
        union = np.logical_or(mask_prev, mask_curr).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append((fid_curr, float(iou)))
    return ious


def rte(vio_df: pd.DataFrame) -> np.ndarray:
    pos = vio_df[["pos_x", "pos_y", "pos_z"]].to_numpy()
    return np.linalg.norm(np.diff(pos, axis=0), axis=1)


def discover_datasets(results_root: str | Path, only_name: str | None) -> list[DatasetInputs]:
    root = Path(results_root)
    if not root.exists():
        return []

    datasets: list[DatasetInputs] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        name = dataset_dir.name
        if only_name and name != only_name:
            continue

        zips = sorted(dataset_dir.glob("*.zip"))
        cov_zip = next((path for path in zips if "timed" not in path.stem.lower()), None)
        timed_zip = next((path for path in zips if "timed" in path.stem.lower()), None)
        if cov_zip or timed_zip:
            datasets.append(DatasetInputs(name=name, cov_zip=cov_zip, timed_zip=timed_zip))
    return datasets


def explicit_dataset(args: argparse.Namespace) -> DatasetInputs | None:
    explicit_args = [
        args.timed_csv,
        args.cov_csv,
        args.baseline_csv,
        args.timed_vio,
        args.cov_vio,
        args.baseline_vio,
        args.timed_masks,
        args.cov_masks,
    ]
    if not any(explicit_args):
        return None

    if not args.name:
        raise SystemExit("--name is required when using explicit CSV/mask arguments")

    return DatasetInputs(
        name=args.name,
        timed_csv=Path(args.timed_csv) if args.timed_csv else None,
        cov_csv=Path(args.cov_csv) if args.cov_csv else None,
        baseline_csv=Path(args.baseline_csv) if args.baseline_csv else None,
        timed_vio=Path(args.timed_vio) if args.timed_vio else None,
        cov_vio=Path(args.cov_vio) if args.cov_vio else None,
        baseline_vio=Path(args.baseline_vio) if args.baseline_vio else None,
        timed_masks=Path(args.timed_masks) if args.timed_masks else None,
        cov_masks=Path(args.cov_masks) if args.cov_masks else None,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def analyze_dataset(dataset: DatasetInputs, out_root: Path) -> dict[str, object]:
    out_dir = out_root / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Analyzing: {dataset.name} ===")
    print(f"Output folder: {out_dir}")

    timed_vio = load_vio_from_zip(dataset.timed_zip) if dataset.timed_zip else load_vio(dataset.timed_vio)
    cov_vio = load_vio_from_zip(dataset.cov_zip) if dataset.cov_zip else load_vio(dataset.cov_vio)
    base_vio = load_vio(dataset.baseline_vio)

    timed_csv = (
        load_metrics_from_zip(dataset.timed_zip)
        if dataset.timed_zip
        else load_explicit_metrics(dataset.timed_csv, dataset.timed_vio)
    )
    cov_csv = (
        load_metrics_from_zip(dataset.cov_zip)
        if dataset.cov_zip
        else load_explicit_metrics(dataset.cov_csv, dataset.cov_vio)
    )
    base_csv = load_explicit_metrics(dataset.baseline_csv, dataset.baseline_vio)

    timed_masks = (
        load_masks_from_zip(dataset.timed_zip, "masks_timed")
        if dataset.timed_zip
        else load_masks_from_folder(dataset.timed_masks)
    )
    cov_masks = (
        load_masks_from_zip(dataset.cov_zip, "masks_cov")
        if dataset.cov_zip
        else load_masks_from_folder(dataset.cov_masks)
    )

    summary: dict[str, object] = {
        "dataset": dataset.name,
        "timed_masks": len(timed_masks),
        "cov_masks": len(cov_masks),
    }

    if timed_csv is not None and cov_csv is not None and "gpu_usage" in timed_csv.columns and "gpu_usage" in cov_csv.columns:
        print("Generating Figure: GPU Usage Over Time")
        fig, ax = plt.subplots(figsize=(12, 4))
        if base_csv is not None and "gpu_usage" in base_csv.columns:
            ax.plot(base_csv.index, base_csv["gpu_usage"], label="Baseline", color="steelblue", alpha=0.7)
        ax.plot(timed_csv.index, timed_csv["gpu_usage"], label="Timed SAM", color="tomato", alpha=0.7)
        ax.plot(cov_csv.index, cov_csv["gpu_usage"], label="Cov SAM", color="seagreen", alpha=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("GPU Usage (%)")
        ax.set_title(f"GPU Usage Over Time - {dataset.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(fig, out_dir / f"{dataset.name}_gpu_usage.png")
    else:
        print("Skipping GPU figure: telemetry CSVs/columns are not present")

    if cov_csv is not None and "covariance_value" in cov_csv.columns:
        print("Generating Figure: Gate Behavior")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(cov_csv.index, cov_csv["covariance_value"], color="steelblue", label="Angular Velocity", alpha=0.8)
        if "gate_blocked" in cov_csv.columns:
            blocked = cov_csv[cov_csv["gate_blocked"] == 1]
            ax.scatter(blocked.index, blocked["covariance_value"], color="red", s=10, label="Gate Blocked", zorder=5)
        if "sam_invoked" in cov_csv.columns:
            invoked = cov_csv[cov_csv["sam_invoked"] == 1]
            ax.scatter(invoked.index, invoked["covariance_value"], color="green", marker="*", s=100, label="SAM Invoked")
        if "cov_threshold" in cov_csv.columns:
            threshold = cov_csv["cov_threshold"].iloc[0]
            ax.axhline(threshold, color="orange", linestyle="--", label=f"Threshold ({threshold:.2f})")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.set_title(f"Gate Behavior Over Time - {dataset.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(fig, out_dir / f"{dataset.name}_gate_behavior.png")
    else:
        print("Skipping gate figure: covariance telemetry is not present")

    if len(timed_masks) > 1 and len(cov_masks) > 1:
        print("Generating Figure: Mask IoU Comparison")
        timed_ious = compute_iou_series(timed_masks)
        cov_ious = compute_iou_series(cov_masks)
        timed_frames = [item[0] for item in timed_ious]
        timed_vals = [item[1] for item in timed_ious]
        cov_frames = [item[0] for item in cov_ious]
        cov_vals = [item[1] for item in cov_ious]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timed_frames, timed_vals, "o-", color="tomato", label=f"Timed SAM (mean={np.mean(timed_vals):.3f})")
        ax.plot(cov_frames, cov_vals, "s-", color="seagreen", label=f"Cov SAM (mean={np.mean(cov_vals):.3f})")
        ax.axhline(0.3, color="gray", linestyle="--", label="Failure threshold (IoU=0.3)")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Mask IoU")
        ax.set_title(f"Mask Tracking IoU - {dataset.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        save_figure(fig, out_dir / f"{dataset.name}_mask_iou.png")

        timed_failures = sum(value < 0.3 for value in timed_vals)
        cov_failures = sum(value < 0.3 for value in cov_vals)
        summary.update(
            {
                "timed_iou_mean": float(np.mean(timed_vals)),
                "timed_iou_median": float(np.median(timed_vals)),
                "timed_iou_failures": int(timed_failures),
                "timed_iou_count": len(timed_vals),
                "cov_iou_mean": float(np.mean(cov_vals)),
                "cov_iou_median": float(np.median(cov_vals)),
                "cov_iou_failures": int(cov_failures),
                "cov_iou_count": len(cov_vals),
            }
        )
        print(
            f"  Timed SAM: mean={np.mean(timed_vals):.3f}, "
            f"failures={timed_failures}/{len(timed_vals)}"
        )
        print(f"  Cov SAM: mean={np.mean(cov_vals):.3f}, failures={cov_failures}/{len(cov_vals)}")
    else:
        print("Skipping mask IoU figure: not enough masks in one or both result sets")

    vios = {}
    if base_vio is not None and len(base_vio) > 1:
        vios["Baseline"] = base_vio
    if timed_vio is not None and len(timed_vio) > 1:
        vios["Timed SAM"] = timed_vio
    if cov_vio is not None and len(cov_vio) > 1:
        vios["Cov SAM"] = cov_vio

    if vios:
        print("Generating Figure: Trajectory")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"Baseline": "steelblue", "Timed SAM": "tomato", "Cov SAM": "seagreen"}
        for name, vio in vios.items():
            color = colors.get(name, "black")
            ax.plot(vio["pos_x"], vio["pos_y"], label=name, color=color, alpha=0.8, linewidth=1.2)
            ax.scatter(vio["pos_x"].iloc[0], vio["pos_y"].iloc[0], color=color, marker="o", s=50, zorder=5)
            ax.scatter(vio["pos_x"].iloc[-1], vio["pos_y"].iloc[-1], color=color, marker="X", s=50, zorder=5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Trajectory Top View - {dataset.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        save_figure(fig, out_dir / f"{dataset.name}_trajectory.png")

        print("  Trajectory RTE:")
        for name, vio in vios.items():
            distances = rte(vio)
            if len(distances) == 0:
                continue
            key = name.lower().replace(" ", "_")
            summary[f"{key}_rte_mean_cm"] = float(distances.mean() * 100)
            summary[f"{key}_rte_std_cm"] = float(distances.std() * 100)
            summary[f"{key}_rte_max_cm"] = float(distances.max() * 100)
            print(
                f"  {name}: mean={distances.mean() * 100:.2f}cm, "
                f"std={distances.std() * 100:.2f}cm, max={distances.max() * 100:.2f}cm"
            )
    else:
        print("Skipping trajectory figure: no usable VIO trajectory files")

    return summary


def print_summary_table(summaries: list[dict[str, object]]) -> None:
    print("\n=== SUMMARY ===")
    for summary in summaries:
        print(f"\n{summary['dataset']}")
        print(f"  Masks saved: timed={summary.get('timed_masks', 0)}, cov={summary.get('cov_masks', 0)}")
        if "timed_iou_mean" in summary and "cov_iou_mean" in summary:
            print(f"  Timed IoU mean/median: {summary['timed_iou_mean']:.3f}/{summary['timed_iou_median']:.3f}")
            print(f"  Cov IoU mean/median:   {summary['cov_iou_mean']:.3f}/{summary['cov_iou_median']:.3f}")
            print(
                "  IoU failures: "
                f"timed={summary['timed_iou_failures']}/{summary['timed_iou_count']}, "
                f"cov={summary['cov_iou_failures']}/{summary['cov_iou_count']}"
            )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    explicit = explicit_dataset(args)
    datasets = [explicit] if explicit else discover_datasets(args.results_root, args.name)
    if not datasets:
        raise SystemExit(f"No datasets found. Check --results_root ({args.results_root}) and --name.")

    summaries = [analyze_dataset(dataset, out_root) for dataset in datasets]
    pd.DataFrame(summaries).to_csv(out_root / "summary_metrics.csv", index=False)
    print_summary_table(summaries)
    print(f"\nAll figures and summary_metrics.csv saved under: {out_root}")


if __name__ == "__main__":
    main()
