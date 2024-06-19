# Copyright (c) OpenMMLab. All rights reserved.
from .torchsparse_wrapper import register_torchsparse

USE_TORCH_SPARSE = False
if USE_TORCH_SPARSE:
    try:
        import torchsparse  # noqa
    except ImportError:
        IS_TORCHSPARSE_AVAILABLE = False
    else:
        IS_TORCHSPARSE_AVAILABLE = register_torchsparse()
else:
    IS_TORCHSPARSE_AVAILABLE = False

__all__ = ['IS_TORCHSPARSE_AVAILABLE']
