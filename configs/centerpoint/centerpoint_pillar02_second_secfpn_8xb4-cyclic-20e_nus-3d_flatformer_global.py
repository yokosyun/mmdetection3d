_base_ = ['./centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

model = dict(
    pts_middle_encoder=dict(
        type='FlatFormer',
        in_channels=64,
        num_heads=8,
        num_blocks=2,
        activation='gelu',
        window_shape=(16, 16, 1),
        sparse_shape=(512, 512, 1),
        output_shape=(512, 512),
        pos_temperature=10000,
        normalize_pos=False,
        group_size=64,
        with_global=False,
    ), )
