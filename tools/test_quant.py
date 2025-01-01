# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import modelopt.torch.quantization as mtq
import torch
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
# from pytorch_quantization import nn as quant_nn
from modelopt.torch.quantization.nn.modules.tensor_quantizer import \
    TensorQuantizer

from mmdet3d.utils import replace_ceph_backend


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def save_model_to_onnx(model, input_shape, file_path):
    # Create a dummy input tensor
    dummy_input = torch.randn(
        input_shape, device=next(model.parameters()).device)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        file_path,  # where to save the model
        export_params=
        True,  # store the trained parameter weights inside the model file
        # opset_version=11,    # the ONNX version to export the model to
        do_constant_folding=
        True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        # dynamic_axes={
        #     "input": {0: "batch_size"},  # variable length axes
        #     "output": {0: "batch_size"},
        # },
    )

    print(f'Model saved to {file_path}')


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.model.eval()
    runner.load_or_resume()

    # m = runner.model.pts_backbone.blocks[0][0]
    # print(torch.max(runner.model.pts_backbone.blocks[0][0].weight.data[:, 28,:,:]))
    # m.weight.data[:, 28,:,:] *= 10.0
    # m.weight.data[:, 62,:,:] *= 10.0
    # m.weight.data[:, 12,:,:] *= 10.0
    # m.weight.data[:, 19,:,:] *= 10.0
    # m.weight.data[:, 6,:,:] *= 10.0
    # print(torch.max(runner.model.pts_backbone.blocks[0][0].weight.data[:, 28,:,:]))

    if True:
        modules_to_fuse = [
            ['pts_backbone.blocks.0.0', 'pts_backbone.blocks.0.1'],
            ['pts_backbone.blocks.0.3', 'pts_backbone.blocks.0.4'],
            ['pts_backbone.blocks.0.6', 'pts_backbone.blocks.0.7'],
            ['pts_backbone.blocks.0.9', 'pts_backbone.blocks.0.10'],
            ['pts_backbone.blocks.1.0', 'pts_backbone.blocks.1.1'],
            ['pts_backbone.blocks.1.3', 'pts_backbone.blocks.1.4'],
            ['pts_backbone.blocks.1.6', 'pts_backbone.blocks.1.7'],
            ['pts_backbone.blocks.1.9', 'pts_backbone.blocks.1.10'],
            ['pts_backbone.blocks.1.12', 'pts_backbone.blocks.1.13'],
            ['pts_backbone.blocks.1.15', 'pts_backbone.blocks.1.16'],
            ['pts_backbone.blocks.2.0', 'pts_backbone.blocks.2.1'],
            ['pts_backbone.blocks.2.3', 'pts_backbone.blocks.2.4'],
            ['pts_backbone.blocks.2.6', 'pts_backbone.blocks.2.7'],
            ['pts_backbone.blocks.2.9', 'pts_backbone.blocks.2.10'],
            ['pts_backbone.blocks.2.12', 'pts_backbone.blocks.2.13'],
            ['pts_backbone.blocks.2.15', 'pts_backbone.blocks.2.16'],
            ['pts_neck.deblocks.0.0', 'pts_neck.deblocks.0.1'],
            ['pts_neck.deblocks.1.0', 'pts_neck.deblocks.1.1'],
            # ConvTran+Bn+ReLU is not supported https://github.com/pytorch/ao/issues/1462
            ['pts_neck.deblocks.2.0', 'pts_neck.deblocks.2.1'],
            ['pts_bbox_head.shared_conv.conv', 'pts_bbox_head.shared_conv.bn'],
            [
                'pts_bbox_head.task_heads.0.reg.0.conv',
                'pts_bbox_head.task_heads.0.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.0.height.0.conv',
                'pts_bbox_head.task_heads.0.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.0.dim.0.conv',
                'pts_bbox_head.task_heads.0.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.0.rot.0.conv',
                'pts_bbox_head.task_heads.0.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.0.vel.0.conv',
                'pts_bbox_head.task_heads.0.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.0.heatmap.0.conv',
                'pts_bbox_head.task_heads.0.heatmap.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.reg.0.conv',
                'pts_bbox_head.task_heads.1.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.height.0.conv',
                'pts_bbox_head.task_heads.1.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.dim.0.conv',
                'pts_bbox_head.task_heads.1.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.rot.0.conv',
                'pts_bbox_head.task_heads.1.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.vel.0.conv',
                'pts_bbox_head.task_heads.1.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.1.heatmap.0.conv',
                'pts_bbox_head.task_heads.1.heatmap.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.reg.0.conv',
                'pts_bbox_head.task_heads.2.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.height.0.conv',
                'pts_bbox_head.task_heads.2.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.dim.0.conv',
                'pts_bbox_head.task_heads.2.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.rot.0.conv',
                'pts_bbox_head.task_heads.2.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.vel.0.conv',
                'pts_bbox_head.task_heads.2.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.2.heatmap.0.conv',
                'pts_bbox_head.task_heads.2.heatmap.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.reg.0.conv',
                'pts_bbox_head.task_heads.3.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.height.0.conv',
                'pts_bbox_head.task_heads.3.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.dim.0.conv',
                'pts_bbox_head.task_heads.3.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.rot.0.conv',
                'pts_bbox_head.task_heads.3.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.vel.0.conv',
                'pts_bbox_head.task_heads.3.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.3.heatmap.0.conv',
                'pts_bbox_head.task_heads.3.heatmap.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.reg.0.conv',
                'pts_bbox_head.task_heads.4.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.height.0.conv',
                'pts_bbox_head.task_heads.4.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.dim.0.conv',
                'pts_bbox_head.task_heads.4.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.rot.0.conv',
                'pts_bbox_head.task_heads.4.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.vel.0.conv',
                'pts_bbox_head.task_heads.4.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.4.heatmap.0.conv',
                'pts_bbox_head.task_heads.4.heatmap.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.reg.0.conv',
                'pts_bbox_head.task_heads.5.reg.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.height.0.conv',
                'pts_bbox_head.task_heads.5.height.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.dim.0.conv',
                'pts_bbox_head.task_heads.5.dim.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.rot.0.conv',
                'pts_bbox_head.task_heads.5.rot.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.vel.0.conv',
                'pts_bbox_head.task_heads.5.vel.0.bn'
            ],
            [
                'pts_bbox_head.task_heads.5.heatmap.0.conv',
                'pts_bbox_head.task_heads.5.heatmap.0.bn'
            ],
        ]

        runner.model = torch.ao.quantization.fuse_modules(
            runner.model, modules_to_fuse)

        new_linear = torch.nn.utils.fuse_linear_bn_eval(
            runner.model.pts_voxel_encoder.pfn_layers[0].linear,
            runner.model.pts_voxel_encoder.pfn_layers[0].norm)
        runner.model.pts_voxel_encoder.pfn_layers[0].linear = new_linear
        runner.model.pts_voxel_encoder.pfn_layers[0].norm = torch.nn.Identity()

    if True:
        # config = mtq.INT8_SMOOTHQUANT_CFG
        from modelopt.torch.quantization.config import \
            _default_disabled_quantizer_cfg
        config = {
            'quant_cfg': {
                'pts_voxel_encoder*weight_quantizer': {
                    'enable': False,
                    'num_bits': 8,
                    'axis': 0
                },
                'pts_backbone.blocks.0*weight_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': 0
                },
                'pts_backbone.blocks.1*weight_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': 0
                },
                'pts_backbone.blocks.2*weight_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': 0
                },
                'pts_neck.*weight_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': 0
                },
                'pts_bbox_head.*weight_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': 0
                },

                # "*weight_quantizer": {"enable": True, "num_bits": 8, "axis": None, "calibrator": "max"},
                # "*input_quantizer": {"enable": True, "num_bits": 8, "axis": None, "calibrator": "histogram"},
                'pts_voxel_encoder*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None
                },
                'pts_backbone.blocks.0.0*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None,
                    'calibrator': 'max'
                },
                'pts_backbone.blocks.0.3*input_quantizer': {
                    'enable': False,
                    'num_bits': 8,
                    'axis': None,
                    'calibrator': 'max'
                },
                'pts_backbone.blocks.0.6*input_quantizer': {
                    'enable': False,
                    'num_bits': 8,
                    'axis': None,
                    'calibrator': 'max'
                },
                'pts_backbone.blocks.0.9*input_quantizer': {
                    'enable': False,
                    'num_bits': 8,
                    'axis': None,
                    'calibrator': 'max'
                },
                'pts_backbone.blocks.1*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None
                },
                'pts_backbone.blocks.2*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None
                },
                'pts_neck.*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None
                },
                'pts_bbox_head.*input_quantizer': {
                    'enable': True,
                    'num_bits': 8,
                    'axis': None
                },

                # "*lm_head*": {"enable": False},
                # "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
                # "*router*": {"enable": False},  # Skip the MOE router
                # "*output_layer*": {"enable": False},
                # "output.*": {"enable": False},
                **_default_disabled_quantizer_cfg,
                # "default": {"enable": False},
            },
            'algorithm':
            'max',  # ["max", "smoothquant", "awq_lite", "awq_clip", "awq_full", "real_quantize"]
        }
        import modelopt
        from modelopt.torch.opt import apply_mode
        from modelopt.torch.quantization.mode import QuantizeModeRegistry

        runner.model = apply_mode(
            runner.model,
            mode=[('quantize', config)],
            registry=QuantizeModeRegistry)

        runner._val_loop = runner.build_val_loop(runner._val_loop)
        from modelopt.torch.quantization.model_quant import calibrate
        calibrate(
            runner.model,
            config['algorithm'],
            forward_loop=runner.val_loop.run)

        if True:
            for name, module in runner.model.named_modules():
                if isinstance(module, TensorQuantizer):
                    if module.is_enabled:
                        calibrator = module._calibrator
                        if isinstance(
                                calibrator, modelopt.torch.quantization.calib.
                                max.MaxCalibrator):
                            # print(name, calibrator)
                            pass
                        elif isinstance(
                                calibrator, modelopt.torch.quantization.calib.
                                histogram.HistogramCalibrator):
                            name = name.replace('.activation_post_process', '')
                            name = name.replace('pts_voxel_encoder', 'b')
                            name = name.replace('pts_backbone', 'b')
                            name = name.replace('pts_neck', 'n')
                            name = name.replace('pts_bbox_head.shared_conv',
                                                'h.s')
                            name = name.replace('pts_bbox_head.task_heads',
                                                'h.t')

                            bin_centers = (calibrator._calib_bin_edges[:-1] +
                                           calibrator._calib_bin_edges[1:]) / 2
                            plt.figure()
                            plt.plot(
                                bin_centers.cpu(),
                                calibrator._calib_hist.cpu(),
                                '--',
                                linewidth=2,
                                markersize=2)
                            plt.xlabel('Value')
                            plt.ylabel('Frequency')
                            plt.title('Histogram: ' + name)
                            plt.axvline(
                                x=module.amax.cpu(),
                                color='r',
                                linestyle='--',
                                label='99th Percentile')
                            os.makedirs('outputs/hist/', exist_ok=True)
                            plt.savefig('outputs/hist/' + name + '.jpg')

        if True:
            print('calibrated model params')
            # mtq.print_quant_summary(runner.model)
            print('---runner._test_loop---')
            runner._test_loop = runner.build_test_loop(runner._test_loop)
            runner.test_loop.run()

        if True:
            from centerpoint_onnx_converter import PFNONNX, CenterPointHeadONNX

            pfn_onnx = PFNONNX(
                runner.model.pts_voxel_encoder.pfn_layers[0].linear)
            input_features = torch.rand([5268, 20, 11]).cuda()
            pth_onnx_pve = os.path.join(
                'outputs', 'pts_voxel_encoder_centerpoint_custom.onnx')
            torch.onnx.export(
                pfn_onnx,
                (input_features, ),
                f=pth_onnx_pve,
                input_names=('input_features', ),
                output_names=('pillar_features', ),
                # dynamic_axes={
                #     'input_features': {
                #         0: 'num_voxels',
                #         1: 'num_max_points'
                #     },
                #     'pillar_features': {
                #         0: 'num_voxels'
                #     },
                # },
                verbose=False,
                opset_version=17,
            )

            pts_backbone_neck_head = CenterPointHeadONNX(
                runner.model.pts_backbone,
                runner.model.pts_neck,
                runner.model.pts_bbox_head,
            )
            x = torch.rand([1, 64, 512, 512]).cuda()
            pth_onnx_backbone_neck_head = os.path.join(
                'outputs', 'pts_backbone_neck_head_centerpoint_custom.onnx')
            output_names = list(
                runner.model.pts_bbox_head.task_heads[0].heads.keys())
            for task_head in runner.model.pts_bbox_head.task_heads:
                rot_heads = {
                    k: None
                    for k in task_head.heads.keys() if 'rot' in k
                }
                task_head.heads = {
                    'heatmap': None,
                    'reg': None,
                    'height': None,
                    'dim': None,
                    **rot_heads,
                    'vel': None,
                }

            torch.onnx.export(
                pts_backbone_neck_head,
                (x, ),
                f=pth_onnx_backbone_neck_head,
                input_names=('spatial_features', ),
                output_names=tuple(output_names),
                # dynamic_axes={
                #     name: {
                #         0: 'batch_size',
                #         2: 'H',
                #         3: 'W'
                #     }
                #     for name in ['spatial_features'] +
                #     output_names
                # },
                verbose=False,
                opset_version=17,
            )
            print(f'Saved pts_backbone_neck_head onnx model:'
                  f' {pth_onnx_backbone_neck_head}')

    else:
        import torch.ao.quantization as quant
        from torch.ao.quantization.observer import (HistogramObserver,
                                                    MinMaxObserver)
        from torch.ao.quantization.qconfig import QConfig

        qconfig = QConfig(
            activation=HistogramObserver.with_args(dtype=torch.qint8),
            # activation=MinMaxObserver.with_args(dtype=torch.qint8),
            weight=HistogramObserver.with_args(dtype=torch.qint8),
        )

        runner.model.qconfig = qconfig

        # disable quantization for output
        for idx in range(6):
            runner.model.pts_bbox_head.task_heads[idx].dim[1].qconfig = None
            runner.model.pts_bbox_head.task_heads[idx].heatmap[
                1].qconfig = None
            runner.model.pts_bbox_head.task_heads[idx].height[1].qconfig = None
            runner.model.pts_bbox_head.task_heads[idx].reg[1].qconfig = None
            runner.model.pts_bbox_head.task_heads[idx].rot[1].qconfig = None
            runner.model.pts_bbox_head.task_heads[idx].vel[1].qconfig = None

        runner.model = quant.prepare(runner.model)

        runner.val_loop.run()

        vis_hist = True

        if vis_hist:
            layer_names = []
            min_values = []
            max_values = []

            os.makedirs('outputs/', exist_ok=True)
            for name, module in runner.model.named_modules():
                # print(name, module)
                if isinstance(module,
                              torch.ao.quantization.observer.MinMaxObserver):
                    layer_names.append(name)
                    min_values.append(module.min_val.item())
                    max_values.append(module.max_val.item())
                elif isinstance(
                        module,
                        torch.ao.quantization.observer.HistogramObserver):
                    name = name.replace('.activation_post_process', '')
                    name = name.replace('pts_voxel_encoder', 'b')
                    name = name.replace('pts_backbone', 'b')
                    name = name.replace('pts_neck', 'n')
                    name = name.replace('pts_bbox_head.shared_conv', 'h.s')
                    name = name.replace('pts_bbox_head.task_heads', 'h.t')

                    layer_names.append(name)
                    max_val = module.max_val.item()
                    min_val = module.min_val.item()
                    min_values.append(min_val)
                    max_values.append(max_val)

                    percentile_99 = get_99th_percentile_from_histogram_observer(
                        module)

                    step = (max_val - min_val) / module.bins
                    bins = torch.arange(min_val, max_val, step)
                    hist = module.histogram
                    if abs(min_val) < 1e-4:
                        bins = bins[1:]
                        hist = hist[1:]

                    plt.figure()
                    plt.plot(
                        bins.cpu(),
                        hist.cpu(),
                        '--',
                        linewidth=2,
                        markersize=2)
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    # plt.yscale('log')
                    plt.title('Histogram: ' + name)
                    plt.axvline(
                        x=percentile_99.cpu(),
                        color='r',
                        linestyle='--',
                        label='99th Percentile')
                    os.makedirs('outputs/hist/', exist_ok=True)
                    plt.savefig('outputs/hist/' + name + '.jpg')

            # plt.figure()
            plt.figure(figsize=(12, 6))
            plt.vlines(layer_names, min_values, max_values)
            plt.xlabel('Layer Name')
            plt.ylabel('Value Range')
            plt.title('Min and Max Values per Layer (Quantization)')
            # plt.xticks(rotation=45, ha="right")
            plt.xticks(rotation=60, ha='right', fontsize=6.0)
            plt.tight_layout()
            plt.legend()
            plt.tight_layout()
            plt.savefig('outputs/hist/a.min_max_act.jpg')
        else:
            # runner.model = torch.ao.quantization.convert(runner.model)
            # runner.model.cpu()
            # print(runner.model)
            # self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

            # self.call_hook('before_run')
            runner.test_loop.run()

    # else:
    #     runner.test()


def get_99th_percentile_from_histogram_observer(observer, percentile=0.999):
    """Calculates the 99th percentile from a HistogramObserver.

    Args:
        observer: A torch.ao.quantization.observer.HistogramObserver instance.

    Returns:
        The 99th percentile value.
    """
    # Calculate the cumulative sum of the histogram
    cdf = torch.cumsum(observer.histogram, dim=0)
    cdf = cdf / torch.sum(observer.histogram)  # Normalize to get CDF

    indices = (cdf >= percentile).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        # If no indices are found (CDF never reaches 99%), return the maximum value
        return observer.max_val
    percentile_index = indices[0]

    # Calculate the fraction of the bin that corresponds to the 99th percentile
    cdf_at_index = cdf[percentile_index - 1]
    fraction_in_bin = (percentile - cdf_at_index) / (
        cdf[percentile_index] - cdf_at_index)

    # Interpolate within the bin
    bin_width = (observer.max_val - observer.min_val) / observer.bins
    percentile_value = observer.min_val + (percentile_index +
                                           fraction_in_bin) * bin_width

    return percentile_value


if __name__ == '__main__':
    main()
