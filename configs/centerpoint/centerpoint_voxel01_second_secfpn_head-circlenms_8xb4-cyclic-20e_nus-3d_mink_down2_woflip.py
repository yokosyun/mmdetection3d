_base_ = [
    './centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d_mink_down2.py'
]

model = dict(data_preprocessor=dict(flip=False), )
