import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Task 1 bundle adjustment results")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--views", type=int, nargs="+", default=[0, 12, 25, 37, 49])
    parser.add_argument("--max-points-3d", type=int, default=5000)
    parser.add_argument("--max-points-2d", type=int, default=3000)
    return parser.parse_args()


def euler_xyz_to_matrix_np(euler_angles: np.ndarray) -> np.ndarray:
    x_angle, y_angle, z_angle = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    cx, cy, cz = np.cos(x_angle), np.cos(y_angle), np.cos(z_angle)
    sx, sy, sz = np.sin(x_angle), np.sin(y_angle), np.sin(z_angle)

    num_views = euler_angles.shape[0]
    rx = np.zeros((num_views, 3, 3), dtype=np.float32)
    ry = np.zeros((num_views, 3, 3), dtype=np.float32)
    rz = np.zeros((num_views, 3, 3), dtype=np.float32)

    rx[:, 0, 0] = 1.0
    rx[:, 1, 1] = cx
    rx[:, 1, 2] = -sx
    rx[:, 2, 1] = sx
    rx[:, 2, 2] = cx

    ry[:, 0, 0] = cy
    ry[:, 0, 2] = sy
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0] = -sy
    ry[:, 2, 2] = cy

    rz[:, 0, 0] = cz
    rz[:, 0, 1] = -sz
    rz[:, 1, 0] = sz
    rz[:, 1, 1] = cz
    rz[:, 2, 2] = 1.0

    return rz @ ry @ rx


def project_points(points3d: np.ndarray, euler_angles: np.ndarray, translations: np.ndarray, focal: float) -> np.ndarray:
    rotation_matrices = euler_xyz_to_matrix_np(euler_angles)
    camera_points = np.einsum("vij,pj->vpi", rotation_matrices, points3d) + translations[:, None, :]

    z_c = np.minimum(camera_points[..., 2], -1e-6)
    u = -focal * camera_points[..., 0] / z_c + 512.0
    v = focal * camera_points[..., 1] / z_c + 512.0
    return np.stack([u, v], axis=-1)


def load_observations(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    points2d_npz = np.load(data_dir / "points2d.npz")
    view_names = sorted(points2d_npz.files)
    stacked = np.stack([points2d_npz[name] for name in view_names], axis=0).astype(np.float32)
    observations = stacked[:, :, :2]
    visibility = stacked[:, :, 2] > 0.5
    return observations, visibility


def camera_centers(rotation_matrices: np.ndarray, translations: np.ndarray) -> np.ndarray:
    rotation_transpose = np.transpose(rotation_matrices, (0, 2, 1))
    return -np.einsum("vij,vj->vi", rotation_transpose, translations)


def plot_point_cloud(points3d: np.ndarray, colors: np.ndarray, centers: np.ndarray, output_path: Path, max_points: int) -> None:
    if len(points3d) > max_points:
        indices = np.linspace(0, len(points3d) - 1, max_points, dtype=int)
        points3d = points3d[indices]
        colors = colors[indices]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=colors, s=1.0, alpha=0.9)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=20, label="Camera centers")
    ax.set_title("Reconstructed Point Cloud and Camera Centers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reprojections(
    observations: np.ndarray,
    visibility: np.ndarray,
    projections: np.ndarray,
    output_path: Path,
    views: list[int],
    max_points: int,
) -> None:
    fig, axes = plt.subplots(len(views), 2, figsize=(10, 4 * len(views)))
    if len(views) == 1:
        axes = np.asarray([axes])

    for row, view_idx in enumerate(views):
        visible = visibility[view_idx]
        obs = observations[view_idx][visible]
        pred = projections[view_idx][visible]

        if len(obs) > max_points:
            indices = np.linspace(0, len(obs) - 1, max_points, dtype=int)
            obs = obs[indices]
            pred = pred[indices]

        ax_obs = axes[row, 0]
        ax_pred = axes[row, 1]

        ax_obs.scatter(obs[:, 0], obs[:, 1], s=2, c=np.arange(len(obs)), cmap="hsv")
        ax_obs.set_title(f"Observed view_{view_idx:03d}")
        ax_obs.set_xlim(0, 1024)
        ax_obs.set_ylim(1024, 0)

        ax_pred.scatter(pred[:, 0], pred[:, 1], s=2, c=np.arange(len(pred)), cmap="hsv")
        ax_pred.set_title(f"Predicted view_{view_idx:03d}")
        ax_pred.set_xlim(0, 1024)
        ax_pred.set_ylim(1024, 0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points3d = np.load(args.result_dir / "optimized_points3d.npy")
    euler_angles = np.load(args.result_dir / "optimized_euler_angles.npy")
    translations = np.load(args.result_dir / "optimized_translations.npy")
    colors = np.load(args.data_dir / "points3d_colors.npy")

    summary = (args.result_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
    focal_line = next(line for line in summary if line.startswith("focal_length="))
    focal = float(focal_line.split("=", 1)[1])

    observations, visibility = load_observations(args.data_dir)
    projections = project_points(points3d, euler_angles, translations, focal)
    rotations = euler_xyz_to_matrix_np(euler_angles)
    centers = camera_centers(rotations, translations)

    plot_point_cloud(
        points3d=points3d,
        colors=colors,
        centers=centers,
        output_path=output_dir / "point_cloud_overview.png",
        max_points=args.max_points_3d,
    )
    plot_reprojections(
        observations=observations,
        visibility=visibility,
        projections=projections,
        output_path=output_dir / "reprojection_overview.png",
        views=args.views,
        max_points=args.max_points_2d,
    )

    per_view_error = np.linalg.norm(projections - observations, axis=-1)
    visible_error = per_view_error[visibility]
    report = "\n".join(
        [
            f"visible_points={int(visibility.sum())}",
            f"mean_reprojection_error={visible_error.mean():.6f}",
            f"median_reprojection_error={np.median(visible_error):.6f}",
            f"max_reprojection_error={visible_error.max():.6f}",
        ]
    )
    (output_dir / "reprojection_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved visualizations to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
