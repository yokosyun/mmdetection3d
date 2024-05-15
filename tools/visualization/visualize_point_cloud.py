import numpy as np

import open3d as o3d


NUM_DIMENSIONS = 2
GRID_SIZE = 1.0
MAX_DIST = 100


def gen_grid(grid_size: float, max_value: float) -> o3d.geometry.LineSet:

    line_set = o3d.geometry.LineSet()

    # Y dimension
    x = np.arange(-max_value, max_value + grid_size, grid_size)
    x = np.repeat(x, NUM_DIMENSIONS)
    y = np.full_like(x, -max_value)
    y[::NUM_DIMENSIONS] = max_value
    z = np.zeros_like(x)
    points_Y = np.vstack((x, y, z)).T

    # X dimension
    y = np.arange(-max_value, max_value + grid_size, grid_size)
    y = np.repeat(y, NUM_DIMENSIONS)
    x = np.full_like(y, -max_value)
    x[::NUM_DIMENSIONS] = max_value
    z = np.zeros_like(y)
    points_X = np.vstack((x, y, z)).T

    # XYZ dimension
    points = np.vstack((points_X, points_Y))
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = np.arange(points.shape[0]).reshape(-1, NUM_DIMENSIONS)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.paint_uniform_color((0.5, 0.5, 0.5))

    return line_set


points = np.fromfile(
    "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
    dtype=np.float32,
)
points = points.reshape(-1, 5)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
line_set_grid = gen_grid(grid_size=GRID_SIZE, max_value=MAX_DIST)

voxel_pcd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)
# print(type(voxel_pcd.property))
# # voxel_pcd[:, 3] += 1.0
o3d.visualization.draw_geometries(
    [pcd, line_set_grid, voxel_pcd],
    zoom=0.3412,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[2.6172, 2.0475, 1.532],
    up=[-0.0694, -0.9768, 0.2024],
)
