# Install

```
export PYTHONPATH=${PYTHONPATH}:"/home/yoko/dev/mmdetection3d"
pip install typing-extensions --upgrade
mim install mmcv=='2.2.0'
pip install -v -e .
```

```
cd mmcv
pip install -v -e .
```

# Data Conversion

```
unlink data/nuscenes
ln -s  /media/yoko/SSD-PGU3/workspace/datasets/nuscenes/v1.0-mini data/nuscenes
ln -s  /media/yoko/SSD-PGU3/workspace/datasets/nuscenes/v1.0-full data/nuscenes
```

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

for centerpoint

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --only-gt-database
```

# MMCV Version Fix

mmdet3d/__init__.py
/home/yoko/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmdet/__init__.py

```
mmcv_maximum_version = "2.3.0"
```

# Inference

## KITTI

```
python demo/pcd_demo.py \
demo/data/kitti/000008.bin \
configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py \
inputs/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth \
--show
```

## NuScene

### train

```
python3.8 tools/train.py configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py --amp
```

```
python3.8 tools/train.py configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py --amp
```

```
python3.8 tools/train.py configs/flatformer/flatformer_quantize_voxel02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer.py --amp
```

### inference

```
python3.8 demo/pcd_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
inputs/ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
--show
```

```
python3.8 demo/pcd_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer.py \
inputs/ckpts/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_epoch_20.pth \
--show
```

```
python3.8 demo/pcd_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_dynamic.py \
inputs/ckpts/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_dynamic_epoch_1.pth \
--show
```

```
python3.8 demo/pcd_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin configs/flatformer/flatformer_voxel02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_quantize.py inputs/ckpts/flatformer_voxel02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_quantize_epoch_5.pth --show
```

### Benchmark

```
python3.8 tools/analysis_tools/benchmark.py \
configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
inputs/ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
--amp
```

```
python3.8 tools/analysis_tools/benchmark.py \
configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer.py \
inputs/ckpts/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_epoch_20.pth \
--amp
```

```
python3.8 tools/analysis_tools/benchmark.py \
configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_dynamic.py \
inputs/ckpts/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_dynamic_epoch_1.pth \
--amp
```

```
python3.8 tools/analysis_tools/benchmark.py \
configs/flatformer/flatformer_voxel02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_quantize.py \
inputs/ckpts/flatformer_voxel02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_quantize_epoch_5.pth \
--amp
```

### Test

```
python3.8 tools/test.py \
configs/centerpoint/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer.py \
inputs/ckpts/centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d_flatformer_epoch_20.pth
```

work_dirs/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d/epoch_20.pth \\

browse dataset

```
python3.8 tools/misc/browse_dataset.py configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py --task lidar_det --output-dir outputs/
```

/home/yoko/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmcv/cnn/bricks/conv.py

```
if layer_type == "TorchSparseConv3d":
    layer = conv_layer(*args, **kwargs)
else:
    layer = conv_layer(*args, **kwargs, **cfg_)
```

visualize log

```
 python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4
-cyclic-20e_nus-3d/20240523_184305/vis_data/scalars.json
```

loss

```
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d/20240524_085003/vis_data/20240524_085003.json --keys loss
plot curve of work_dirs/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d/20240524_085003/vis_data/20240524_085003.json, metric is los
```

# Mean Voxel Feature

- taking mean of voxel feature doesn't effect performance at all

```
class HardSimpleVFE
if True:
    points_mean = features[:, :, :self.num_features].sum(
        dim=1, keepdim=False) / num_points.type_as(features).view(
            -1, 1)
    num_points, _ = points_mean.shape
    # rand_x = torch.rand(num_points, device="cuda") * 10
    # points_mean[:, 0] -= rand_x
else:
    points_mean = features[:, 0, :self.num_features]
return points_mean.contiguous()
```

# Reference

[centerpoint++](https://cvpr2021.wad.vision/)
