import numpy as np
import torch

from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer

points = np.fromfile(
    'demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin',
    dtype=np.float32,
)
print(points.shape)
points = points.reshape(-1, 5)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
# bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900, -1.5808]])
# )
# Draw 3D bboxes
# visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()
