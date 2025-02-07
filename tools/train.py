# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


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

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

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

            # "*weight_quantizer": {"enable": True, "num_bits": 8, "axis": 0, "calibrator": "max"},
            # "*input_quantizer": {"enable": False, "num_bits": 8, "axis": None, "calibrator": "max"},
            'pts_voxel_encoder*input_quantizer': {
                'enable': False,
                'num_bits': 8,
                'axis': None,
            },
            'pts_backbone.blocks.0.0*input_quantizer': {
                'enable': True,
                'num_bits': 8,
                'axis': None,
                # 'calibrator': 'max',
                # "unsigned": True,
            },
            'pts_backbone.blocks.0.3*input_quantizer': {
                'enable': True,
                'num_bits': 8,
                'axis': None,
                # 'calibrator': 'max',
                # "unsigned": True,
            },
            'pts_backbone.blocks.0.6*input_quantizer': {
                'enable': True,
                'num_bits': 8,
                'axis': None,
                # 'calibrator': 'max',
                # "unsigned": True,
            },
            'pts_backbone.blocks.0.9*input_quantizer': {
                'enable': True,
                'num_bits': 8,
                'axis': None,
                # 'calibrator': 'max',
                # "unsigned": True,
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

    runner.call_hook('before_run')
    runner.model.eval()
    runner.load_or_resume()
    runner.model = apply_mode(
        runner.model,
        mode=[('quantize', config)],
        registry=QuantizeModeRegistry)

    runner._val_loop = runner.build_val_loop(runner._val_loop)
    from modelopt.torch.quantization.model_quant import calibrate
    calibrate(
        runner.model, config['algorithm'], forward_loop=runner.val_loop.run)

    runner._test_loop = runner.build_test_loop(runner._test_loop)
    runner.test_loop.run()

    # start training
    runner.train()


if __name__ == '__main__':
    main()
