_base_ = [
    './centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='quantize',
        # voxel_layer=dict(
        #     max_num_points=10,
        #     voxel_size=voxel_size,
        #     max_voxels=(90000, 120000))
    ),
    pts_voxel_encoder=dict(type='SkipVFE'))
