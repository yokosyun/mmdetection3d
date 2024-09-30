_base_ = ['./centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=10, save_last=True))
