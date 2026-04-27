import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: 3D reconstruction with COLMAP")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workspace-dir", type=Path, default=Path("outputs/colmap"))
    parser.add_argument("--colmap-bin", type=str, default="colmap")
    parser.add_argument("--camera-model", type=str, default="PINHOLE")
    parser.add_argument("--single-camera", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dense", action="store_true", help="Run dense reconstruction steps")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="GPU index for dense reconstruction. Leave unset for COLMAP default.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Delete the existing workspace directory before running.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Skip sparse reconstruction and continue from an existing sparse model in workspace-dir/sparse/0.",
    )
    parser.add_argument(
        "--skip-undistorter",
        action="store_true",
        help="Skip image_undistorter and assume dense workspace is already prepared.",
    )
    parser.add_argument(
        "--skip-patch-match",
        action="store_true",
        help="Skip patch_match_stereo and run only the remaining dense steps.",
    )
    return parser.parse_args()


def ensure_colmap_available(colmap_bin: str) -> None:
    if Path(colmap_bin).exists():
        return
    if shutil.which(colmap_bin) is not None:
        return
    raise FileNotFoundError(
        "COLMAP executable was not found. Install COLMAP first, then rerun with "
        f"--colmap-bin <path-to-colmap.exe> if it is not on PATH."
    )


def run_command(command: list[str], log_path: Path) -> None:
    print(f"\n>>> Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {' '.join(command)}",
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    ensure_colmap_available(args.colmap_bin)

    image_dir = args.data_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    workspace_dir = args.workspace_dir
    if args.force_clean and workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    sparse_dir = workspace_dir / "sparse"
    dense_dir = workspace_dir / "dense"
    logs_dir = workspace_dir / "logs"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if args.dense:
        dense_dir.mkdir(parents=True, exist_ok=True)

    database_path = workspace_dir / "database.db"

    commands: list[tuple[str, list[str]]] = []

    if not args.dense_only:
        commands.extend(
            [
                (
                    "01_feature_extractor",
                    [
                        args.colmap_bin,
                        "feature_extractor",
                        "--database_path",
                        str(database_path),
                        "--image_path",
                        str(image_dir),
                        "--ImageReader.camera_model",
                        args.camera_model,
                        "--ImageReader.single_camera",
                        str(args.single_camera),
                    ],
                ),
                (
                    "02_exhaustive_matcher",
                    [
                        args.colmap_bin,
                        "exhaustive_matcher",
                        "--database_path",
                        str(database_path),
                    ],
                ),
                (
                    "03_mapper",
                    [
                        args.colmap_bin,
                        "mapper",
                        "--database_path",
                        str(database_path),
                        "--image_path",
                        str(image_dir),
                        "--output_path",
                        str(sparse_dir),
                    ],
                ),
            ]
        )

    if args.dense:
        sparse_model_dir = sparse_dir / "0"
        if not sparse_model_dir.exists():
            raise FileNotFoundError(
                f"Sparse model not found: {sparse_model_dir}. Run sparse reconstruction first."
            )
        if not args.skip_undistorter:
            commands.append(
                (
                    "04_image_undistorter",
                    [
                        args.colmap_bin,
                        "image_undistorter",
                        "--image_path",
                        str(image_dir),
                        "--input_path",
                        str(sparse_model_dir),
                        "--output_path",
                        str(dense_dir),
                    ],
                )
            )
        if not args.skip_patch_match:
            commands.append(
                (
                    "05_patch_match_stereo",
                    [
                        args.colmap_bin,
                        "patch_match_stereo",
                        "--workspace_path",
                        str(dense_dir),
                    ]
                    + (
                        ["--PatchMatchStereo.gpu_index", str(args.gpu_index)]
                        if args.gpu_index is not None
                        else []
                    ),
                )
            )
        commands.append(
            (
                "06_stereo_fusion",
                [
                    args.colmap_bin,
                    "stereo_fusion",
                    "--workspace_path",
                    str(dense_dir),
                    "--output_path",
                    str(dense_dir / "fused.ply"),
                ],
            )
        )

    for step_name, command in commands:
        run_command(command, logs_dir / f"{step_name}.log")

    summary_lines = [
        f"workspace_dir={workspace_dir.resolve()}",
        f"database_path={database_path.resolve()}",
        f"sparse_dir={sparse_dir.resolve()}",
        f"dense_enabled={args.dense}",
        f"dense_only={args.dense_only}",
        f"skip_undistorter={args.skip_undistorter}",
        f"skip_patch_match={args.skip_patch_match}",
    ]
    if args.dense:
        summary_lines.append(f"dense_dir={dense_dir.resolve()}")
        summary_lines.append(f"fused_ply={(dense_dir / 'fused.ply').resolve()}")

    (workspace_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nCOLMAP pipeline finished successfully.")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
