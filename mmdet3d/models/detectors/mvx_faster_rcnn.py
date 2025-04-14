# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Dict, List, Optional, Sequence

from torch import Tensor

from mmdet3d.registry import MODELS
from .mvx_two_stage import MVXTwoStageDetector


@MODELS.register_module()
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)


@MODELS.register_module()
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_pts_bbox:
            return None
        start_time = time.time()
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'], points, img_feats,
            batch_input_metas)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        print(f'pts_voxel_encoder ={elapsed:.3f}[ms]')

        batch_size = voxel_dict['coors'][-1, 0] + 1

        start_time = time.time()
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        print(f'pts_middle_encoder ={elapsed:.3f}[ms]')

        start_time = time.time()
        x = self.pts_backbone(x)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        print(f'pts_backbone ={elapsed:.3f}[ms]')

        start_time = time.time()
        if self.with_pts_neck:
            x = self.pts_neck(x)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000
        print(f'pts_neck ={elapsed:.3f}[ms]')

        return x
