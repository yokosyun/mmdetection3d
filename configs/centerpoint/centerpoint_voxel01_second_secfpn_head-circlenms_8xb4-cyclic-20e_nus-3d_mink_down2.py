_base_ = [
    './centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d_mink.py'
]

model = dict(
    pts_middle_encoder=dict(
        down_kernel_size=2,
        encoder_paddings=((0, 0, [1, 0, 0]), (0, 0, [1, 0, 0]),
                          (0, 0, [0, 0, 0]), (0, 0)),
    ))
