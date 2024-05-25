_base_ = ['./centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))

_base_.visualizer.vis_backends = [
    # dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
