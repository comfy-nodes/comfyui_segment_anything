from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "Grounding DINO SAM Segment": GroundingDinoSAMSegment,
    "Is Mask Empty?": IsMaskEmptyNode,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
