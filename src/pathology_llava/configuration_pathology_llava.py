from typing import Optional

from transformers import PretrainedConfig

from .constants import (
    DEFAULT_VISION_TOWER,
    DEFAULT_LLM_NAME,
    DEFAULT_PATHOLOGY_LLAVA_REPO,
    DEFAULT_PROJECTOR_SUBFOLDER,
)


class PathologyLLaVAConfig(PretrainedConfig):
    """
    Configuration for the Pathology-LLaVA multimodal model.

    This config does NOT include any third-party weights. It only stores
    the identifiers or local paths needed to load:
      - the PLIP vision tower,
      - the LLaMA language model,
      - the projector / Q-Former weights from Pathology-LLaVA.

    Parameters
    ----------
    vision_tower_name_or_path: str
        HF repo id or local path for PLIP (CLIP-style vision model).
    llm_name_or_path: str
        HF repo id or local path for the base LLaMA model.
    pathology_llava_root: Optional[str]
        Local root directory where Pathology-LLaVA weights were extracted.
        If None, the code may download from the official HF repo at runtime.
    projector_subfolder: str
        Relative path (under pathology_llava_root) pointing to the projector.
    image_size: int
        Input image resolution for PLIP.
    max_image_patches: int
        Max number of image patches per example for the projector.
    """

    model_type = "pathology-llava"

    def __init__(
        self,
        vision_tower_name_or_path: str = DEFAULT_VISION_TOWER,
        llm_name_or_path: str = DEFAULT_LLM_NAME,
        pathology_llava_repo_id: str = DEFAULT_PATHOLOGY_LLAVA_REPO,
        pathology_llava_root: Optional[str] = None,
        projector_subfolder: str = DEFAULT_PROJECTOR_SUBFOLDER,
        image_size: int = 224,
        max_image_patches: int = 196,
        use_anyres: bool = True,
        torch_dtype: Optional[str] = "float16",
        **kwargs,
    ):
        self.vision_tower_name_or_path = vision_tower_name_or_path
        self.llm_name_or_path = llm_name_or_path
        self.pathology_llava_repo_id = pathology_llava_repo_id
        self.pathology_llava_root = pathology_llava_root
        self.projector_subfolder = projector_subfolder
        self.image_size = image_size
        self.max_image_patches = max_image_patches
        self.use_anyres = use_anyres
        self.torch_dtype = torch_dtype

        super().__init__(**kwargs)
