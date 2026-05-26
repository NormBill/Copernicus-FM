from .evidence_memory import TaskEvidenceMemory
from .ot_matcher import OTMatcher
from .validity_calibrator import ValidityCalibrator

try:
    from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
    from .ops.modules import MSDeformAttn
except (ImportError, ModuleNotFoundError, OSError):
    SpatialPriorModule = None
    InteractionBlock = None
    deform_inputs = None
    MSDeformAttn = None
