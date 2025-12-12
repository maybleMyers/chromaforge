# GLM-4.6V model implementation
# Adapted from HuggingFace transformers for local use

from .configuration_glm46v import Glm46VConfig, Glm4vVisionConfig, Glm4vTextConfig
from .modeling_glm46v import (
    Glm46VForConditionalGeneration,
    Glm46VModel,
    Glm46VPreTrainedModel,
)
from .processing_glm46v import Glm46VProcessor
from .image_processing_glm46v import Glm46VImageProcessor

# Optional imports that may fail on older transformers
try:
    from .image_processing_glm46v_fast import Glm46VImageProcessorFast
except (ImportError, TypeError):
    Glm46VImageProcessorFast = None

try:
    from .video_processing_glm46v import Glm46VVideoProcessor
except (ImportError, TypeError):
    Glm46VVideoProcessor = None

__all__ = [
    "Glm46VConfig",
    "Glm46VForConditionalGeneration",
    "Glm46VModel",
    "Glm46VPreTrainedModel",
    "Glm46VProcessor",
    "Glm46VImageProcessor",
    "Glm46VImageProcessorFast",
    "Glm46VVideoProcessor",
]
