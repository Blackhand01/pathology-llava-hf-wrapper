import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from ..constants import (
    DEFAULT_VISION_TOWER,
    DEFAULT_LLM_NAME,
    DEFAULT_PATHOLOGY_LLAVA_REPO,
    DEFAULT_PROJECTOR_SUBFOLDER,
    ENV_VISION_TOWER,
    ENV_LLM_NAME,
    ENV_PATHOLOGY_LLAVA_PATH,
    ENV_PROJECTOR_SUBFOLDER,
)


def _default_cache_root() -> Path:
    # Dedicated cache dir under HF cache (or home).
    base = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(base) / "pathology-llava"


def resolve_vision_tower_path(name_or_path: Optional[str] = None) -> str:
    # Allow override via ENV, else config, else default.
    env_val = os.environ.get(ENV_VISION_TOWER)
    if env_val:
        return env_val
    if name_or_path:
        return name_or_path
    return DEFAULT_VISION_TOWER


def resolve_llm_path(name_or_path: Optional[str] = None) -> str:
    env_val = os.environ.get(ENV_LLM_NAME)
    if env_val:
        return env_val
    if name_or_path:
        return name_or_path
    return DEFAULT_LLM_NAME


def resolve_projector_path(
    pathology_llava_root: Optional[str],
    projector_subfolder: Optional[str] = None,
    pathology_llava_repo_id: str = DEFAULT_PATHOLOGY_LLAVA_REPO,
    cache_root: Optional[Path] = None,
) -> str:
    """
    Resolve local path to the Pathology-LLaVA projector directory.

    If `pathology_llava_root` is None, this will download from the
    official HF repo into a dedicated cache directory.
    """
    env_root = os.environ.get(ENV_PATHOLOGY_LLAVA_PATH)
    if env_root:
        root = Path(env_root)
    elif pathology_llava_root is not None:
        root = Path(pathology_llava_root)
    else:
        # Download only the necessary files.
        cache_root = cache_root or _default_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        root = Path(
            snapshot_download(
                repo_id=pathology_llava_repo_id,
                cache_dir=str(cache_root),
                # allow HF to handle revisions; keep it simple.
            )
        )

    subfolder = (
        os.environ.get(ENV_PROJECTOR_SUBFOLDER)
        or projector_subfolder
        or DEFAULT_PROJECTOR_SUBFOLDER
    )
    projector_dir = root / subfolder
    if not projector_dir.exists():
        raise FileNotFoundError(
            f"Expected projector directory at {projector_dir}. "
            "Check PATHOLOGY_LLAVA_PROJECTOR_SUBFOLDER or config.projector_subfolder."
        )
    return str(projector_dir)
