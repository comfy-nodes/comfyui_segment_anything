from .node import *
from .install import *

NODE_CLASS_MAPPINGS = {
    "SAM Model Loader (Segment Anything)": SAMModelLoader,
    "Grounding DINO Model Loader": GroundingDinoModelLoader,
    "Grounding DINO SAM Segment": GroundingDinoSAMSegment,
    'InvertMask (segment anything)': InvertMask,
    "Is Mask Empty?": IsMaskEmptyNode,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
