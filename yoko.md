# Install

```
export PYTHONPATH=${PYTHONPATH}:"/home/yoko/dev/mmdetection3d"
pip install typing-extensions --upgrade
mim install mmcv=='2.2.0'
```

# Data Conversion

```
ln -s  /media/yoko/SSD-PGU3/workspace/datasets/nuscenes/v1.0-mini data/nuscenes
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini

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

```
python demo/pcd_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
inputs/ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
--show
```

```
python tools/misc/browse_dataset.py configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py --task lidar_det --output-dir outputs/
```
