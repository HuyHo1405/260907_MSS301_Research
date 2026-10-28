try:
    from .detector import YOLOPlateDetector
except ImportError:
    YOLOPlateDetector = None

from .detector_onnx import ONNXPlateDetector
from .recognizer_rapid import RapidOCRRecognizer

try:
    from .recognizer_lprnet import LPRNetRecognizer
except ImportError:
    LPRNetRecognizer = None

from .postprocess import format_vietnamese_plate
from .alpr_pipeline import ALPRPipeline

__all__ = [
    'ALPRPipeline',
    'YOLOPlateDetector',
    'RapidOCRRecognizer',
    'LPRNetRecognizer',
    'format_vietnamese_plate'
]
