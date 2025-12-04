"""
Constants for pathology-llava wrapper.
"""

# Default HF identifiers for third-party models.
DEFAULT_VISION_TOWER = "vinid/plip"
DEFAULT_LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_PATHOLOGY_LLAVA_REPO = "OpenFace-CQUPT/Pathology-LLaVA"

# Subfolders inside the Pathology-LLaVA raw layout (after extraction).
DEFAULT_INSTRUCTION_TUNING_SUBFOLDER = "instruction_tuning_weight_ft"
DEFAULT_PROJECTOR_SUBFOLDER = "instruction_tuning_weight_ft/projector"

# Environment variables to override paths/names at runtime.
ENV_VISION_TOWER = "PATHOLOGY_LLAVA_VISION_NAME_OR_PATH"
ENV_LLM_NAME = "PATHOLOGY_LLAVA_LLM_NAME_OR_PATH"
ENV_PATHOLOGY_LLAVA_PATH = "PATHOLOGY_LLAVA_ROOT"
ENV_PROJECTOR_SUBFOLDER = "PATHOLOGY_LLAVA_PROJECTOR_SUBFOLDER"
