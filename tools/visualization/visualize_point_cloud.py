import numpy as np
from typing import Tuple, List, Dict, Any
import glob
import open3d as o3d
import matplotlib.pyplot as plt

NUM_DIMENSIONS = 2

X_RANGE = [-51.2, 51.2]
Y_RANGE = [-51.2, 51.2]
Z_RANGE = [-5, 3]
VOXEL_RESOLUTIONS = np.array([0.1, 0.1, 0.2])


def get_prallel_lines(x_range, y_range, reolution, is_vertical=False):
    intervals = np.arange(x_range[0], x_range[1] + reolution, reolution)
    intervals = np.repeat(intervals, NUM_DIMENSIONS)
    min_max = np.full_like(intervals, y_range[0])
    min_max[::NUM_DIMENSIONS] = y_range[1]
    heights = np.zeros_like(intervals) - 5
    parallel_lines = np.vstack((intervals, min_max, heights)).T
    if is_vertical:
        parallel_lines[:, [1, 0, 2]] = parallel_lines.copy()

    return parallel_lines


def gen_grid(x_range, y_range, raster_resolution: np.ndarray) -> o3d.geometry.LineSet:

    horizontal_lines = get_prallel_lines(
        x_range, y_range, raster_resolution[0], is_vertical=False
    )
    vertical_lines = get_prallel_lines(
        y_range, x_range, raster_resolution[1], is_vertical=True
    )

    line_set = o3d.geometry.LineSet()
    points = np.vstack((vertical_lines, horizontal_lines))
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = np.arange(points.shape[0]).reshape(-1, NUM_DIMENSIONS)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color((0.5, 0.5, 0.5))

    # o3d.visualization.RenderOption.line_width = 0.01

    return line_set


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_voxelize(
    points,
    voxel_resolutions: Tuple[float, ...],
) -> List[np.ndarray]:

    coords = np.floor(points / voxel_resolutions).astype(np.int32)
    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    return coords, indices


def visualize_points(points: np.ndarray, x_range, y_range, voxe_resolutions) -> None:
    if points.shape[1] == 2:
        z = np.zeros(points.shape[0])
        points = np.stack([points[:, 0], points[:, 1], z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    line_set_grid = gen_grid(x_range, y_range, raster_resolution=voxe_resolutions[:2])
    o3d.visualization.draw_geometries(
        [pcd, line_set_grid],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )


def get_occupany_ratio(num_points: int, num_coord: int) -> float:
    return num_coord / num_points * 100


def analize_point_clouds(
    points,
    num_voxels,
    voxel_resolutions,
    x_range,
    y_range,
    visualize=False,
) -> Dict[str, Any]:
    # Voxel
    voxel_coords, voxel_indices = sparse_voxelize(points, voxel_resolutions)

    # Raster
    raster_coords, raster_indices = sparse_voxelize(
        points[:, :2], voxel_resolutions[:2]
    )

    # Count
    num_points = len(points)
    num_occupied_voxels = len(voxel_coords)
    num_occupied_raster = len(raster_coords)

    points_compress_ratio_voxelize = num_occupied_voxels / num_points * 100
    points_compress_ratio_raster = num_occupied_raster / num_points * 100

    voxel_occupancy_ratio = num_occupied_voxels / np.prod(num_voxels) * 100
    raster_occupany_ratio = num_occupied_raster / np.prod(num_voxels[:2]) * 100

    print(f"num_voxels={num_voxels} [x,y,z]")
    print(f"voxel_resolutions={voxel_resolutions} [x,y,z]")
    print(f"num_points={num_points:.0f} [N]")
    print(f"num_occupied_voxels={num_occupied_voxels:.0f} [N]")
    print(f"num_occupied_raster={num_occupied_raster:.0f} [N]")
    print(f"num_voxels={np.prod(num_voxels):.0f} [N]")
    print(f"num_rasters={np.prod(num_voxels[:2]):.0f} [N]")

    print(f"points_compress_ratio_voxelize={points_compress_ratio_voxelize:.2f} [%]")
    print(f"points_compress_ratio_raster={points_compress_ratio_raster:.2f} [%]")

    print(f"voxel_occupancy_ratio={voxel_occupancy_ratio:.2f} [%]")
    print(f"raster_occupany_ratio={raster_occupany_ratio:.2f} [%]")

    # Visuaization
    if visualize:
        points = points[raster_indices]
        visualize_points(points, x_range, y_range, voxel_resolutions)

    return {"voxel_occupancy_ratio": voxel_occupancy_ratio}


def main():

    # pcd_paths = [
    #     "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin"
    # ]

    pcd_paths = glob.glob("./data/nuscenes/samples/LIDAR_TOP/*.bin")

    voxel_occupancy_ratio_list = []

    for pcd_path in pcd_paths:
        print("------------------------------------------")
        print(pcd_path)
        points = np.fromfile(
            pcd_path,
            dtype=np.float32,
        )

        # Points
        points = points.reshape(-1, 5)[:, :3]

        # scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        scales = [1]

        for scale in scales:

            voxel_resolutions = VOXEL_RESOLUTIONS * scale

            num_voxels = (
                np.array(
                    [
                        X_RANGE[1] - X_RANGE[0],
                        Y_RANGE[1] - Y_RANGE[0],
                        Z_RANGE[1] - Z_RANGE[0],
                    ]
                )
                / voxel_resolutions
            )
            results = analize_point_clouds(
                points,
                num_voxels,
                voxel_resolutions,
                X_RANGE,
                Y_RANGE,
                visualize=False,
            )

            voxel_occupancy_ratio_list.append(results["voxel_occupancy_ratio"])

    # Plot Result
    fig = plt.figure()
    plt.suptitle("voxel_occupancy_ratio histogram in NuScenes", fontsize=16, y=1)
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(voxel_occupancy_ratio_list, bins=50)
    ax.set_title(
        f"X_RANGE={X_RANGE}[m], Y_RANGE={Y_RANGE}[m], Z_RANGE={Z_RANGE}[m], VOXEL_RESOLUTIONS={VOXEL_RESOLUTIONS}[m]",
        fontsize=6,
    )
    ax.set_xlabel("voxel_occupancy_ratio[%]")
    ax.set_ylabel("frequency[N]")
    fig.savefig("hist.jpg")
    fig.show()


if __name__ == "__main__":
    main()
