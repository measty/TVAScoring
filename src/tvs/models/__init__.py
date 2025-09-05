r"""Model architectures for TubuloVillous scoring.

This subpackage contains small convolutional and transformer based
models used to regress a score in ``[-1, 1]`` indicating the
`tubular` \(-1\) to `villous` \(+1\) architecture of a patch.
"""

from .cnn_baseline import CNNBaseline
from .conch_head import ConchHead
from .factory import make_model
from .resnet import ResNet18Baseline

__all__ = ["CNNBaseline", "ResNet18Baseline", "ConchHead", "make_model"]
