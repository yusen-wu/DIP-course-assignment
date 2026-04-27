# Assignment 3 - Bundle Adjustment

This directory contains the implementation of Assignment 03 of DIP:
- `Task 1`: Bundle Adjustment with PyTorch
- `Task 2`: 3D reconstruction with COLMAP

## Contents

- `data/`: original assignment data
- `task1_bundle_adjustment.py`: Task 1 optimization script
- `visualize_task1_results.py`: Task 1 result visualization
- `task2_colmap.py`: Task 2 COLMAP pipeline script
- `visualize_data.py`: raw observation visualization
- `requirements.txt`: Python dependencies
- `outputs/run_2000/`: final Task 1 results
- `outputs/colmap/`: Task 2 reconstruction results

## Requirements

Install Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Task 1

Run Bundle Adjustment:

```bash
python task1_bundle_adjustment.py --steps 2000 --device cpu --output-dir outputs/run_2000
```

Visualize the final result:

```bash
python visualize_task1_results.py --result-dir outputs/run_2000
```

Final Task 1 outputs:
- `outputs/run_2000/optimized_points3d.obj`
- `outputs/run_2000/loss_curve.png`
- `outputs/run_2000/reprojection_overview.png`
- `outputs/run_2000/point_cloud_overview.png`

## Task 2

Sparse reconstruction with COLMAP:

```bash
python task2_colmap.py --colmap-bin "D:\study\DIP\colmap-x64-windows-cuda\bin\colmap.exe" --force-clean
```

Dense reconstruction with COLMAP:

```bash
python task2_colmap.py --colmap-bin "D:\study\DIP\colmap-x64-windows-cuda\bin\colmap.exe" --dense --dense-only --skip-undistorter
```

Main Task 2 outputs:
- `outputs/colmap/sparse/0/`
- `outputs/colmap/dense/fused.ply`

## Current Results

Task 1 final summary:
- `steps = 2000`
- `final_loss = 0.000049`
- `focal_length = 885.654053`
- `mean reprojection error = 0.009807 px`

Task 2 current dense result:
- `outputs/colmap/dense/fused.ply`
