"""
Compute Absolute Trajectory Error (ATE) for VINS-Fusion against EuRoC ground truth.

Ground truth CSV:
    Header starts with #timestamp. Timestamps are nanoseconds.
    Required columns:
        #timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z

VIO CSV:
    No header. 12 columns with a trailing comma:
        integer_timestamp, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z,
        quat_w, vel_x, vel_y, vel_z, empty
    VIO timestamps are integer seconds and are converted to nanoseconds.

Example:
    python compute_ate.py --gt mh05_gt.csv --timed mh05_timed_vio.csv \
        --cov mh05_cov_vio.csv --name MH05
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

NS_PER_SECOND = 1_000_000_000


@dataclass
class Trajectory:
    timestamps_ns: np.ndarray
    positions: np.ndarray


@dataclass
class AteResult:
    mode: str
    matched_pairs: int
    timestamps_ns: np.ndarray
    gt_positions: np.ndarray
    aligned_vio_positions: np.ndarray
    errors: np.ndarray
    rmse: float
    mean: float
    max: float
    std: float
    rotation: np.ndarray
    translation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ATE between VINS-Fusion VIO CSVs and EuRoC ground truth."
    )
    parser.add_argument("--gt", required=True, help="EuRoC ground truth CSV")
    parser.add_argument("--timed", required=True, help="Timed SAM VIO CSV")
    parser.add_argument("--cov", required=True, help="Intelligent/Cov SAM VIO CSV")
    parser.add_argument("--name", required=True, help="Dataset name used in plot titles")
    parser.add_argument(
        "--max_time_diff",
        type=float,
        default=0.05,
        help="Maximum nearest-neighbor timestamp difference in seconds",
    )
    parser.add_argument("--out", default=".", help="Output folder for ate_trajectory.png and ate_error.png")
    return parser.parse_args()


def normalize_gt_column_name(name: object) -> str:
    cleaned = str(name).strip().lstrip("#").strip()
    if "[" in cleaned:
        cleaned = cleaned.split("[", 1)[0].strip()
    return cleaned


def clean_trajectory(timestamps_ns: np.ndarray, positions: np.ndarray) -> Trajectory:
    valid = np.isfinite(timestamps_ns) & np.isfinite(positions).all(axis=1)
    timestamps_ns = timestamps_ns[valid].astype(np.int64)
    positions = positions[valid].astype(float)
    order = np.argsort(timestamps_ns)
    return Trajectory(timestamps_ns=timestamps_ns[order], positions=positions[order])


def load_ground_truth(path: str | Path) -> Trajectory:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [normalize_gt_column_name(col) for col in df.columns]

    required = ["timestamp", "p_RS_R_x", "p_RS_R_y", "p_RS_R_z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing ground-truth columns: {', '.join(missing)}")

    timestamps_ns = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    positions = df[["p_RS_R_x", "p_RS_R_y", "p_RS_R_z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return clean_trajectory(timestamps_ns, positions)


def load_vio(path: str | Path) -> Trajectory:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 11:
        raise ValueError(f"{csv_path} must have at least 11 VIO columns")

    timestamps_seconds = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    timestamps_ns = np.rint(timestamps_seconds * NS_PER_SECOND)
    positions = df.iloc[:, 1:4].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return clean_trajectory(timestamps_ns, positions)


def align_by_nearest_timestamp(
    gt: Trajectory,
    vio: Trajectory,
    max_time_diff_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_diff_ns = int(max_time_diff_seconds * NS_PER_SECOND)
    gt_t = gt.timestamps_ns
    vio_t = vio.timestamps_ns

    matched_times: list[int] = []
    matched_gt: list[np.ndarray] = []
    matched_vio: list[np.ndarray] = []

    insert_indices = np.searchsorted(gt_t, vio_t)
    for vio_idx, insert_idx in enumerate(insert_indices):
        candidates = []
        if insert_idx < len(gt_t):
            candidates.append(insert_idx)
        if insert_idx > 0:
            candidates.append(insert_idx - 1)
        if not candidates:
            continue

        best_gt_idx = min(candidates, key=lambda idx: abs(int(gt_t[idx]) - int(vio_t[vio_idx])))
        if abs(int(gt_t[best_gt_idx]) - int(vio_t[vio_idx])) <= max_diff_ns:
            matched_times.append(int(vio_t[vio_idx]))
            matched_gt.append(gt.positions[best_gt_idx])
            matched_vio.append(vio.positions[vio_idx])

    if not matched_times:
        raise ValueError(
            f"No matches found within {max_time_diff_seconds:.3f}s. "
            "Check that VIO integer-second timestamps correspond to the GT timestamps."
        )

    return np.array(matched_times, dtype=np.int64), np.vstack(matched_gt), np.vstack(matched_vio)


def horn_se3_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t that minimize ||R @ source + t - target|| without scale."""
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if source.shape[0] < 3:
        raise ValueError("Need at least 3 matched poses for SE3 alignment")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (rotation @ points.T).T + translation


def compute_stats(errors: np.ndarray) -> tuple[float, float, float, float]:
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean = float(np.mean(errors))
    max_error = float(np.max(errors))
    std = float(np.std(errors))
    return rmse, mean, max_error, std


def compute_ate(mode: str, gt: Trajectory, vio: Trajectory, max_time_diff_seconds: float) -> AteResult:
    timestamps_ns, gt_positions, vio_positions = align_by_nearest_timestamp(gt, vio, max_time_diff_seconds)
    rotation, translation = horn_se3_alignment(vio_positions, gt_positions)
    aligned_vio = apply_transform(vio_positions, rotation, translation)
    errors = np.linalg.norm(aligned_vio - gt_positions, axis=1)
    rmse, mean, max_error, std = compute_stats(errors)

    return AteResult(
        mode=mode,
        matched_pairs=len(errors),
        timestamps_ns=timestamps_ns,
        gt_positions=gt_positions,
        aligned_vio_positions=aligned_vio,
        errors=errors,
        rmse=rmse,
        mean=mean,
        max=max_error,
        std=std,
        rotation=rotation,
        translation=translation,
    )


def print_comparison_table(results: list[AteResult]) -> None:
    headers = ["Mode", "Pairs", "ATE RMSE (m)", "ATE Mean (m)", "ATE Max (m)", "ATE Std (m)"]
    rows = [
        [
            result.mode,
            str(result.matched_pairs),
            f"{result.rmse:.6f}",
            f"{result.mean:.6f}",
            f"{result.max:.6f}",
            f"{result.std:.6f}",
        ]
        for result in results
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\nATE comparison after SE3 alignment")
    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in rows:
        print(fmt(row))


def plot_trajectory(results: list[AteResult], name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Timed SAM": "tomato", "Intelligent SAM": "seagreen"}

    first = results[0]
    ax.plot(first.gt_positions[:, 0], first.gt_positions[:, 1], color="black", linewidth=2.0, label="Ground Truth")

    for result in results:
        ax.plot(
            result.aligned_vio_positions[:, 0],
            result.aligned_vio_positions[:, 1],
            color=colors.get(result.mode),
            linewidth=1.4,
            alpha=0.85,
            label=f"{result.mode} aligned VIO",
        )

    ax.set_title(f"ATE Aligned Trajectories - {name}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error(results: list[AteResult], name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"Timed SAM": "tomato", "Intelligent SAM": "seagreen"}

    for result in results:
        time_seconds = (result.timestamps_ns - result.timestamps_ns[0]) / NS_PER_SECOND
        ax.plot(time_seconds, result.errors, color=colors.get(result.mode), linewidth=1.2, label=result.mode)

    ax.set_title(f"ATE Error Over Time - {name}")
    ax.set_xlabel("Time from first match (s)")
    ax.set_ylabel("ATE (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(args.gt)
    timed = load_vio(args.timed)
    cov = load_vio(args.cov)

    results = [
        compute_ate("Timed SAM", gt, timed, args.max_time_diff),
        compute_ate("Intelligent SAM", gt, cov, args.max_time_diff),
    ]

    print_comparison_table(results)

    trajectory_path = output_dir / "ate_trajectory.png"
    error_path = output_dir / "ate_error.png"
    plot_trajectory(results, args.name, trajectory_path)
    plot_error(results, args.name, error_path)

    print(f"\nSaved trajectory plot: {trajectory_path}")
    print(f"Saved error plot: {error_path}")


if __name__ == "__main__":
    main()
