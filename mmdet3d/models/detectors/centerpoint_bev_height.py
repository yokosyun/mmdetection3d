# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

from torch import Tensor

from mmdet3d.registry import MODELS
from .mvx_two_stage import MVXTwoStageDetector


@MODELS.register_module()
class CenterPointBEVHEIGHT(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet.

    Args:
        pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.p
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_fusion_layer: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):

        super(CenterPointBEVHEIGHT,
              self).__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor, **kwargs)

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
        x = self.pts_voxel_encoder(points)
        # batch_size = voxel_dict['coors'][-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
        #                             batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
