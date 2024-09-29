_base_ = [
    './centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    data_preprocessor=dict(voxel_type='quantize', ),
    pts_voxel_encoder=dict(type='SkipVFE'))
