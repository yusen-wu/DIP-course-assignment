import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: Bundle Adjustment with PyTorch")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--init-distance", type=float, default=2.5)
    parser.add_argument("--init-fov-deg", type=float, default=60.0)
    parser.add_argument("--point-init-scale", type=float, default=0.3)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--resume-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    points2d_npz = np.load(data_dir / "points2d.npz")
    view_names = sorted(points2d_npz.files)
    stacked = np.stack([points2d_npz[name] for name in view_names], axis=0).astype(np.float32)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)

    observations = torch.from_numpy(stacked[:, :, :2])
    visibility = torch.from_numpy(stacked[:, :, 2] > 0.5)
    return observations, visibility, colors


def euler_xyz_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    x_angle, y_angle, z_angle = euler_angles.unbind(dim=-1)
    cx, cy, cz = torch.cos(x_angle), torch.cos(y_angle), torch.cos(z_angle)
    sx, sy, sz = torch.sin(x_angle), torch.sin(y_angle), torch.sin(z_angle)

    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return rz @ ry @ rx


class BundleAdjustmentModel(torch.nn.Module):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        image_size: tuple[int, int],
        init_distance: float,
        init_fov_deg: float,
        point_init_scale: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        width, height = image_size
        self.width = float(width)
        self.height = float(height)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        focal_init = self.height / (2.0 * np.tan(np.deg2rad(init_fov_deg) / 2.0))
        self.log_focal = torch.nn.Parameter(torch.log(torch.tensor([focal_init], dtype=torch.float32, device=device)))
        self.euler_angles = torch.nn.Parameter(torch.zeros(num_views, 3, dtype=torch.float32, device=device))

        translations = torch.zeros(num_views, 3, dtype=torch.float32, device=device)
        translations[:, 2] = -init_distance
        self.translations = torch.nn.Parameter(translations)

        points = point_init_scale * torch.randn(num_points, 3, dtype=torch.float32, device=device)
        self.points3d = torch.nn.Parameter(points)

    def load_state_arrays(
        self,
        points3d: np.ndarray,
        euler_angles: np.ndarray,
        translations: np.ndarray,
        focal_length: float,
    ) -> None:
        with torch.no_grad():
            self.points3d.copy_(torch.from_numpy(points3d).to(self.points3d.device, dtype=self.points3d.dtype))
            self.euler_angles.copy_(torch.from_numpy(euler_angles).to(self.euler_angles.device, dtype=self.euler_angles.dtype))
            self.translations.copy_(torch.from_numpy(translations).to(self.translations.device, dtype=self.translations.dtype))
            focal_tensor = torch.tensor([focal_length], device=self.log_focal.device, dtype=self.log_focal.dtype)
            self.log_focal.copy_(torch.log(focal_tensor))

    @property
    def focal_length(self) -> torch.Tensor:
        return torch.exp(self.log_focal)[0]

    def project(self) -> torch.Tensor:
        rotation_matrices = euler_xyz_to_matrix(self.euler_angles)
        camera_points = torch.einsum("vij,pj->vpi", rotation_matrices, self.points3d) + self.translations[:, None, :]

        x_c = camera_points[..., 0]
        y_c = camera_points[..., 1]
        z_c = camera_points[..., 2].clamp(max=-1e-6)

        u = -self.focal_length * x_c / z_c + self.cx
        v = self.focal_length * y_c / z_c + self.cy
        return torch.stack([u, v], dim=-1)

    def forward(self, observations: torch.Tensor, visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project()
        residuals = projected - observations
        visible_residuals = residuals[visibility]
        loss = (visible_residuals ** 2).mean()
        return loss, projected


def save_obj(points3d: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for point, color in zip(points3d, colors):
            f.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def save_loss_curve(loss_history: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Mean Squared Reprojection Error")
    plt.title("Bundle Adjustment Optimization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations, visibility, colors = load_data(args.data_dir)
    observations = observations.to(device)
    visibility = visibility.to(device)

    num_views, num_points, _ = observations.shape
    image_size = (1024, 1024)

    model = BundleAdjustmentModel(
        num_views=num_views,
        num_points=num_points,
        image_size=image_size,
        init_distance=args.init_distance,
        init_fov_deg=args.init_fov_deg,
        point_init_scale=args.point_init_scale,
        device=device,
    )

    previous_steps = 0
    if args.resume_dir is not None:
        model.load_state_arrays(
            points3d=np.load(args.resume_dir / "optimized_points3d.npy"),
            euler_angles=np.load(args.resume_dir / "optimized_euler_angles.npy"),
            translations=np.load(args.resume_dir / "optimized_translations.npy"),
            focal_length=float(
                next(
                    line.split("=", 1)[1]
                    for line in (args.resume_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
                    if line.startswith("focal_length=")
                )
            ),
        )
        summary_lines = (args.resume_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
        previous_steps = int(
            next(line.split("=", 1)[1] for line in summary_lines if line.startswith("steps="))
        )
        print(f"Resumed from {args.resume_dir.resolve()} after {previous_steps} steps")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_history: list[float] = []

    progress_bar = tqdm(range(1, args.steps + 1), desc="Optimizing", dynamic_ncols=True)
    for local_step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(observations, visibility)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        progress_bar.set_postfix(loss=f"{loss_value:.4f}", f=f"{float(model.focal_length.item()):.2f}")
        total_step = previous_steps + local_step

        if local_step % args.print_every == 0 or local_step == 1 or local_step == args.steps:
            print(
                f"step={total_step:04d} "
                f"loss={loss_value:.6f} "
                f"focal={float(model.focal_length.item()):.3f}"
            )

    optimized_points = model.points3d.detach().cpu().numpy()
    optimized_rotations = model.euler_angles.detach().cpu().numpy()
    optimized_translations = model.translations.detach().cpu().numpy()

    save_obj(optimized_points, colors, args.output_dir / "optimized_points3d.obj")
    save_loss_curve(loss_history, args.output_dir / "loss_curve.png")

    np.save(args.output_dir / "optimized_points3d.npy", optimized_points)
    np.save(args.output_dir / "optimized_euler_angles.npy", optimized_rotations)
    np.save(args.output_dir / "optimized_translations.npy", optimized_translations)
    np.save(args.output_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float32))

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"device={device}",
                f"steps={previous_steps + args.steps}",
                f"final_loss={loss_history[-1]:.6f}",
                f"focal_length={float(model.focal_length.item()):.6f}",
                f"num_views={num_views}",
                f"num_points={num_points}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
