_base_ = ['./centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel_type='quantize',
    ),
    pts_voxel_encoder=dict(type='SkipVFE', num_features=5),
)
