1. **Sintesi minima (3–5 righe)**

Costruiamo un progetto pubblico “code-only” che espone PA-LLaVA come modello multimodale HF compatibile (`AutoModelForCausalLM` + `AutoProcessor`), ma senza ridistribuire pesi di PLIP, LLaMA-3 o Pathology-LLaVA.
Il codice wrapper sarà rilasciato sotto Apache-2.0, con NOTICE che accredita Pathology-LLaVA (Apache-2.0), PLIP (research-only), Meta LLaMA-3 (licenza propria) e XTuner (Apache-2.0).
L’HF repo conterrà solo `config.json` + remote code e punterà alle sorgenti ufficiali dei pesi.
Ti lascio una struttura di repo concreta + skeleton di codice immediatamente estendibile.

---

2. **Decision log (scelte chiave vs alternative)**

* **Code-only wrapper vs modello con pesi inclusi**

  * Scelta: **code-only** (niente `.bin` / `.safetensors` di terzi).
  * Perché:

    * Pathology-LLaVA è Apache-2.0 (puoi riusare il codice) ma i pesi possono essere scaricati direttamente da HF.
    * PLIP è posizionato esplicitamente come **research output**, con intended use di ricerca; non ha un chiaro via libera per redistribuzione per deploy.
    * LLaMA-3 è sotto Meta Llama 3 Community License: puoi usare e modificare, ma la redistribuzione richiede attribuzioni precise; duplicare i pesi è inutile e delicato.

* **Dove sta il codice? GitHub vs HF remote code**

  * Scelta: codice principale in **GitHub package** (`pathology_llava`), copiato/syncato come remote code nel repo HF.
  * Vantaggi:

    * Manutenzione e CI su GitHub.
    * HF repo minimale (config + mirror dei file chiave); `auto_map` può puntare a quei file.

* **Come integrare i modelli?**

  * Scelta: wrapper che chiama **HF ufficiali**:

    * `vinid/plip` per la vision tower.
    * `meta-llama/Meta-Llama-3-8B-Instruct` o simili per l’LLM.
    * `OpenFace-CQUPT/Pathology-LLaVA` per projector/adapter.
  * Alternativa (da evitare ora): reimplementare tutto in stile LLaVA-HF con gigantesco porting; troppo lavoro upfront.

---

3. **Piano gerarchico**

**Macro-obiettivo**
Repo pubblico “pathology-llava” + HF wrapper che espongono PA-LLaVA come modello HF multimodale (`AutoModelForCausalLM`, `AutoProcessor`), caricando pesi dai repo ufficiali.

**Obiettivi → sotto-obiettivi**

1. **Design repo & packaging**

   * Definire albero sorgenti (`src/pathology_llava/...`).
   * Configurare `pyproject.toml`, LICENSE, NOTICE, README.

2. **Implementare core wrapper HF**

   * `PathologyLLaVAConfig` (config HF).
   * `PathologyLLaVAForConditionalGeneration` (modello multimodale).
   * `PathologyLLaVAProcessor` (immagine + testo).
   * Utility per caricare PLIP, LLaMA, projector Pathology-LLaVA.

3. **HF model repo**

   * `config.json` con `model_type: "pathology-llava"` e `auto_map`.
   * Model card con sezioni su licenze e uso previsto.

4. **Esempi e test**

   * `examples/demo_inference.py` (patch-level QA).
   * `tests/test_config_and_loading.py` (istanza + forward dummy).

5. **Licensing & docs**

   * Testo NOTICE coerente.
   * README con sezioni “Licensing & Third-Party Models” e “Intended Use”.

Procedo ora con gli **artefatti**, step by step.

---

4. **Artefatti**

### 4.1. Struttura concreta del repo GitHub

Propongo layout PEP 621 con `src/`:

```text
pathology-llava/
  LICENSE
  NOTICE
  README.md
  pyproject.toml
  .gitignore

  src/
    pathology_llava/
      __init__.py
      configuration_pathology_llava.py
      modeling_pathology_llava.py
      processing_pathology_llava.py
      constants.py

      utils/
        __init__.py
        downloading.py
        vision.py
        multimodal.py

  examples/
    demo_inference.py

  tests/
    __init__.py
    test_config_and_loading.py
    test_dummy_forward.py
```

HF model repo (separato, es. `stefano-roy/Pathology-LLaVA-hf`):

```text
config.json
README.md   # model card
# (opzionale) copia dei file:
configuration_pathology_llava.py
modeling_pathology_llava.py
processing_pathology_llava.py
```

---

### 4.2. Core package: `__init__`, config, constants

#### `src/pathology_llava/__init__.py`

```python
from .configuration_pathology_llava import PathologyLLaVAConfig
from .modeling_pathology_llava import PathologyLLaVAForConditionalGeneration
from .processing_pathology_llava import PathologyLLaVAProcessor

__all__ = [
    "PathologyLLaVAConfig",
    "PathologyLLaVAForConditionalGeneration",
    "PathologyLLaVAProcessor",
]
```

#### `src/pathology_llava/constants.py`

```python
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
```

#### `src/pathology_llava/configuration_pathology_llava.py`

```python
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
```

---

### 4.3. Utils: downloading + vision + multimodal glue

#### `src/pathology_llava/utils/__init__.py`

```python
from .downloading import (
    resolve_vision_tower_path,
    resolve_llm_path,
    resolve_projector_path,
)
from .vision import encode_plip_for_projector
from .multimodal import (
    build_multimodal_inputs,
)

__all__ = [
    "resolve_vision_tower_path",
    "resolve_llm_path",
    "resolve_projector_path",
    "encode_plip_for_projector",
    "build_multimodal_inputs",
]
```

#### `src/pathology_llava/utils/downloading.py`

```python
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
```

#### `src/pathology_llava/utils/vision.py`

```python
from typing import Tuple

import torch
from torch import Tensor
from transformers import CLIPModel


def encode_plip_for_projector(
    vision_model: CLIPModel,
    pixel_values: Tensor,
    select_layer: int = -2,
) -> Tuple[Tensor, Tensor]:
    """
    Encode images with PLIP and extract features suitable for the Q-Former projector.

    This is a simplified approximation of the feature selection logic used in PA-LLaVA:
    - We take hidden states from an intermediate vision layer.
    - We treat the first token as 'global' query, the rest as patch tokens.

    Parameters
    ----------
    vision_model:
        CLIPModel instance loaded from the PLIP checkpoint.
    pixel_values:
        Tensor of shape (batch, 3, H, W).
    select_layer:
        Index of the hidden state layer to use (e.g. -2 for penultimate).

    Returns
    -------
    ori_pixel_embeds: Tensor
        Global token embeddings, shape (batch, 1, hidden_dim).
    patch_pixel_embeds: Tensor
        Patch token embeddings, shape (batch, num_patches, hidden_dim).
    """
    outputs = vision_model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[select_layer]  # (B, seq_len, hidden)
    # Typically [CLS] + patches; use that convention.
    ori_pixel_embeds = hidden_states[:, :1, :]          # (B, 1, D)
    patch_pixel_embeds = hidden_states[:, 1:, :]        # (B, P, D)
    return ori_pixel_embeds, patch_pixel_embeds
```

#### `src/pathology_llava/utils/multimodal.py`

```python
from typing import Dict, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel


def build_multimodal_inputs(
    llm: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    image_embeds: Tensor,
) -> Dict[str, Tensor]:
    """
    Build multimodal inputs for the LLM by concatenating image embeddings
    as prefix tokens in the hidden space.

    This does NOT reproduce the exact PA-LLaVA training recipe, but provides
    a simple, working multimodal path:

      - Embed text tokens via the LLM token embedding matrix.
      - Concatenate image_embeds (B, I, H) in front of the text embeddings.
      - Build an extended attention mask.

    Parameters
    ----------
    llm:
        LLaMA model (AutoModelForCausalLM) with get_input_embeddings().
    input_ids:
        (B, T) token ids.
    attention_mask:
        (B, T) attention mask or None.
    image_embeds:
        (B, I, H) image embeddings projected into LLM hidden space.

    Returns
    -------
    inputs:
        Dict ready to feed into llm(**inputs).
    """
    text_embeds = llm.get_input_embeddings()(input_ids)  # (B, T, H)
    if text_embeds.dtype != image_embeds.dtype:
        image_embeds = image_embeds.to(text_embeds.dtype)
    if text_embeds.device != image_embeds.device:
        image_embeds = image_embeds.to(text_embeds.device)

    multimodal_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (B, I+T, H)
    bsz, seq_len = input_ids.shape
    img_len = image_embeds.shape[1]

    if attention_mask is None:
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=input_ids.device)

    img_mask = torch.ones((bsz, img_len), dtype=attention_mask.dtype, device=attention_mask.device)
    extended_mask = torch.cat([img_mask, attention_mask], dim=1)  # (B, I+T)

    return {
        "inputs_embeds": multimodal_embeds,
        "attention_mask": extended_mask,
    }
```

---

### 4.4. Core HF model class

#### `src/pathology_llava/modeling_pathology_llava.py`

```python
from typing import Optional, List, Union, Any, Dict

import torch
from torch import nn, Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    CLIPModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_pathology_llava import PathologyLLaVAConfig
from .utils import (
    resolve_vision_tower_path,
    resolve_llm_path,
    resolve_projector_path,
    encode_plip_for_projector,
    build_multimodal_inputs,
)


class PathologyLLaVAForConditionalGeneration(PreTrainedModel):
    """
    Hugging Face-compatible multimodal wrapper for PA-LLaVA.

    This class:
      - loads PLIP as vision tower,
      - loads a LLaMA-based AutoModelForCausalLM,
      - loads the projector / Q-Former model from the official Pathology-LLaVA
        distribution,
      - exposes a `generate` method via the underlying language model.

    No third-party weights are distributed with this package.
    """

    config_class = PathologyLLaVAConfig

    def __init__(self, config: PathologyLLaVAConfig):
        super().__init__(config)

        # Vision tower (PLIP / CLIP-style).
        vision_id_or_path = resolve_vision_tower_path(config.vision_tower_name_or_path)
        self.vision_model = CLIPModel.from_pretrained(
            vision_id_or_path,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Language model (LLaMA).
        llm_id_or_path = resolve_llm_path(config.llm_name_or_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_id_or_path,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Projector / Q-Former from Pathology-LLaVA official weights.
        projector_dir = resolve_projector_path(
            pathology_llava_root=config.pathology_llava_root,
            projector_subfolder=config.projector_subfolder,
            pathology_llava_repo_id=config.pathology_llava_repo_id,
        )
        # We rely on the projector being exported as an AutoModel with custom code.
        self.projector = AutoModel.from_pretrained(
            projector_dir,
            trust_remote_code=True,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Ensure LLM does not use cache by default (better for training / gradients).
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False

        # Tie weights if needed.
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "PathologyLLaVAForConditionalGeneration":
        """
        Standard HF loading path. This method is mainly kept to be explicit.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _encode_images(self, pixel_values: Tensor) -> Tensor:
        """
        Run PLIP + projector to obtain image embeddings in LLM hidden space.

        Parameters
        ----------
        pixel_values:
            Tensor of shape (B, 3, H, W).

        Returns
        -------
        image_embeds: Tensor
            (B, I, H_lm) image embeddings aligned to the LLM hidden size.
        """
        ori_pixel_embeds, patch_pixel_embeds = encode_plip_for_projector(
            self.vision_model,
            pixel_values.to(self.vision_model.dtype),
        )

        # Simple attention mask: consider all patches as valid tokens.
        device = patch_pixel_embeds.device
        image_atts = torch.ones(
            patch_pixel_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )

        # The projector is expected to follow the PA-LLaVA Q-Former API:
        # forward(ori_pixel, patch_pixel, image_atts=None)
        projector_out = self.projector(
            ori_pixel=ori_pixel_embeds,
            patch_pixel=patch_pixel_embeds,
            image_atts=image_atts,
        )
        # We accept both dict-like and tensor outputs.
        if isinstance(projector_out, dict):
            image_embeds = projector_out.get("last_hidden_state", None)
            if image_embeds is None:
                # Fallback: try attribute.
                image_embeds = getattr(projector_out, "last_hidden_state", None)
        else:
            image_embeds = getattr(projector_out, "last_hidden_state", None)

        if image_embeds is None:
            raise RuntimeError(
                "Projector did not return `last_hidden_state`. "
                "Check the exported Pathology-LLaVA projector."
            )

        # Ensure dtype/device compatibility with the LLM.
        image_embeds = image_embeds.to(self.language_model.dtype).to(
            self.language_model.device
        )
        return image_embeds

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass that optionally incorporates image features.

        If `pixel_values` is provided, images are encoded and injected as
        prefix tokens in the LLM hidden space.
        """
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pixel_values are given.")
            if input_ids.device != self.language_model.device:
                input_ids = input_ids.to(self.language_model.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.language_model.device)

            image_embeds = self._encode_images(pixel_values)
            mm_inputs = build_multimodal_inputs(
                self.language_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=image_embeds,
            )

            if labels is not None:
                labels = labels.to(self.language_model.device)
                mm_inputs["labels"] = labels

            outputs = self.language_model(**mm_inputs, **kwargs)
        else:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # Wrap result as CausalLMOutputWithPast for HF compatibility.
        if isinstance(outputs, CausalLMOutputWithPast):
            return outputs

        # Minimal adapter in case underlying model returns plain logits.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = getattr(outputs, "loss", None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
```

---

### 4.5. Processor multimodale

#### `src/pathology_llava/processing_pathology_llava.py`

```python
from typing import List, Union, Optional, Dict, Any

from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    ProcessorMixin,
)

from .configuration_pathology_llava import PathologyLLaVAConfig
from .constants import DEFAULT_VISION_TOWER, DEFAULT_LLM_NAME


class PathologyLLaVAProcessor(ProcessorMixin):
    """
    HF Processor for Pathology-LLaVA wrapper.

    It combines:
      - CLIP-style image processor from PLIP,
      - tokenizer from the underlying LLaMA model.

    Typical usage:
        processor = PathologyLLaVAProcessor.from_pretrained(...)
        inputs = processor(images=image, text="Describe the patch", return_tensors="pt")
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: CLIPImageProcessor,
        tokenizer,
        **kwargs,
    ):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.current_processor = self.image_processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "PathologyLLaVAProcessor":
        """
        For simplicity, this loads:
          - image_processor from PLIP,
          - tokenizer from the LLaMA model used by the config.
        """
        config = PathologyLLaVAConfig.from_pretrained(pretrained_model_name_or_path)

        vision_id = config.vision_tower_name_or_path or DEFAULT_VISION_TOWER
        llm_id = config.llm_name_or_path or DEFAULT_LLM_NAME

        image_processor = CLIPImageProcessor.from_pretrained(vision_id)
        tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            use_fast=True,
            padding_side="right",
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    def __call__(
        self,
        images: Union[Image.Image, torch.Tensor, List[Image.Image], List[torch.Tensor]],
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Normalize to list for both images and text.
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        if isinstance(text, str):
            text = [text]

        image_outputs = self.image_processor(
            images=images,
            return_tensors="pt",
        )

        text_outputs = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        data = {
            "pixel_values": image_outputs["pixel_values"],
            "input_ids": text_outputs["input_ids"],
            "attention_mask": text_outputs["attention_mask"],
        }

        if return_tensors is None or return_tensors == "pt":
            return data

        raise ValueError(f"Unsupported return_tensors={return_tensors!r} for this processor.")
```

---

### 4.6. `pyproject.toml` minimale

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pathology-llava"
version = "0.1.0"
description = "Hugging Face-compatible wrapper for PA-LLaVA / Pathology-LLaVA using PLIP and LLaMA-3."
authors = [
  { name = "Stefano Roy Bisignano" }
]
license = { text = "Apache-2.0" }
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "torch>=2.1",
  "transformers>=4.40.0",
  "huggingface_hub>=0.23.0",
  "Pillow>=10.0",
]

[project.urls]
Homepage = "https://github.com/<your-user>/pathology-llava"
Source = "https://github.com/<your-user>/pathology-llava"
Issues = "https://github.com/<your-user>/pathology-llava/issues"

[tool.setuptools.packages.find]
where = ["src"]
```

---

### 4.7. README (estratti rilevanti)

#### `README.md` — sezione iniziale

```markdown
# pathology-llava

Hugging Face–compatible wrapper around PA-LLaVA / Pathology-LLaVA.

This package exposes a multimodal model
`PathologyLLaVAForConditionalGeneration` and a
`PathologyLLaVAProcessor` that combine:

- PLIP as vision tower (`vinid/plip`),
- a LLaMA-3 language model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`),
- projector / Q-Former and adapters from `OpenFace-CQUPT/Pathology-LLaVA`.

No third-party weights are distributed with this package. All models are
loaded from their official sources at runtime.
```

#### README — “Licensing & Third-Party Models”

```markdown
## Licensing & Third-Party Models

- **Wrapper code (`pathology-llava`)**

  - Licensed under the Apache License 2.0 (see `LICENSE` in this repo).

- **Pathology-LLaVA / PA-LLaVA**

  - Code and weights are distributed by OpenFace-CQUPT under Apache-2.0.
  - This project uses their public code and weights but does not
    redistribute them.
  - See: `OpenFace-CQUPT/Pathology-LLaVA` on Hugging Face.

- **PLIP**

  - `vinid/plip` is a CLIP-style vision model released as a research
    output for research communities, with intended use limited to
    research exploration.
  - This wrapper only references that model and expects users to obtain
    it from the original authors.

- **Meta LLaMA-3**

  - The language model is loaded from Meta’s LLaMA 3 distribution
    (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) and is governed by the
    Meta Llama 3 Community License.
  - This project does not redistribute LLaMA weights.

- **XTuner**

  - PA-LLaVA is implemented using XTuner, which is released under the
    Apache License 2.0.
```

#### README — “Intended Use & Limitations”

```markdown
## Intended Use & Limitations

This repository is intended **for research and experimentation only**.

- It is not designed or validated for clinical decision making.
- It should not be used as a stand-alone medical device or diagnostic tool.
- Any deployment in clinical or commercial settings must:

  - comply with the licenses of PLIP, Pathology-LLaVA, and LLaMA-3,
  - undergo appropriate validation, regulatory review, and legal review.

The authors of this wrapper do not provide any warranty on the behaviour
of the underlying models.
```

#### README — “Quickstart”

````markdown
## Quickstart

Install:

```bash
pip install pathology-llava
````

Run a simple inference example (assuming you have access to LLaMA-3):

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_id = "stefano-roy/Pathology-LLaVA-hf"  # HF wrapper repo

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
)

image = Image.open("example_patch.png").convert("RGB")
question = "Describe the main histologic pattern in this patch."

inputs = processor(images=image, text=question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
    )

print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
```

````

---

### 4.8. NOTICE (con crediti espliciti)

#### `NOTICE`

Testo conciso, originale (non copia-incolla di licenze):

```text
pathology-llava
Copyright (c) 2025 Stefano Roy Bisignano

This project provides wrapper code that integrates third-party models:

- Pathology-LLaVA / PA-LLaVA
  (code and weights licensed under Apache-2.0 by the original authors,
   available at https://huggingface.co/OpenFace-CQUPT/Pathology-LLaVA).

- PLIP (vinid/plip)
  (a CLIP-style vision model released as a research output for research
   communities; the model card describes its intended research-only use).

- Meta LLaMA 3
  (language model released by Meta under the Meta Llama 3 Community
   License; see the LICENSE file in the corresponding Meta repositories).

- XTuner
  (toolkit for efficient fine-tuning of LLMs and VLMs, Apache-2.0).

This repository does not redistribute any third-party weights. All such
weights must be obtained from their original authors and are governed by
their respective licenses.

Meta Llama 3 is licensed under the Meta Llama 3 Community License,
Copyright (c) Meta Platforms, Inc. All Rights Reserved.
````

(La riga finale è la formula di attribuzione richiesta dalla licenza LLaMA-3.)

Per `LICENSE`, scrivi “Apache License 2.0” e copia il testo ufficiale dal sito Apache (non lo riproduco qui integralmente per limiti di citazione).

---

### 4.9. HF wrapper: `config.json` + model card

#### HF `config.json` (es. repo `stefano-roy/Pathology-LLaVA-hf`)

```json
{
  "model_type": "pathology-llava",
  "architectures": [
    "PathologyLLaVAForConditionalGeneration"
  ],
  "torch_dtype": "float16",
  "vision_tower_name_or_path": "vinid/plip",
  "llm_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
  "pathology_llava_repo_id": "OpenFace-CQUPT/Pathology-LLaVA",
  "pathology_llava_root": null,
  "projector_subfolder": "instruction_tuning_weight_ft/projector",
  "auto_map": {
    "AutoConfig": "configuration_pathology_llava.PathologyLLaVAConfig",
    "AutoModelForCausalLM": "modeling_pathology_llava.PathologyLLaVAForConditionalGeneration",
    "AutoProcessor": "processing_pathology_llava.PathologyLLaVAProcessor"
  }
}
```

Questo presuppone che nel repo HF tu abbia copiato i tre file Python.

#### HF model card (estratti chiave di `README.md` del repo HF)

````markdown
# Pathology-LLaVA HF Wrapper (code-only)

This repository provides a Hugging Face–compatible wrapper around
PA-LLaVA / Pathology-LLaVA, combining PLIP, a LLaMA-3 language model,
and the official projector/adapters from OpenFace-CQUPT.

No third-party weights are included in this repository. All models are
loaded from their original sources at runtime.

## License

- Wrapper code in this repository: Apache License 2.0.
- Pathology-LLaVA code and weights: Apache-2.0, distributed by the
  original authors on Hugging Face.
- PLIP: research output `vinid/plip`, intended for research use only.
- LLaMA-3: Meta Llama 3 Community License (see Meta repositories).
- XTuner: Apache-2.0.

Users are responsible for complying with the licenses of all
third-party models.

## Intended use

This wrapper is intended for **research and experimentation**, e.g.:

- patch-level question-answering on pathology images,
- concept probing, qualitative analysis, and ablation studies.

It is **not intended** for clinical deployment or decision-making.

## Usage

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

model_id = "stefano-roy/Pathology-LLaVA-hf"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
)

image = Image.open("example_patch.png").convert("RGB")
question = "What is the dominant histologic pattern in this patch?"

inputs = processor(images=image, text=question, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    )

print(processor.tokenizer.decode(output_ids[0], skip_special_tokens=True))
````

If you want to keep all weights local and avoid automatic downloads, set:

* `PATHOLOGY_LLAVA_VISION_NAME_OR_PATH` to your local PLIP path,
* `PATHOLOGY_LLAVA_LLM_NAME_OR_PATH` to your local LLaMA-3 path,
* `PATHOLOGY_LLAVA_ROOT` to your extracted Pathology-LLaVA directory.

````

---

### 4.10. Esempio: `examples/demo_inference.py`

```python
#!/usr/bin/env python

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="stefano-roy/Pathology-LLaVA-hf",
        help="HF repo id for the wrapper.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to a pathology patch image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the main histologic pattern in this patch.",
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto",
    )

    image = Image.open(args.image_path).convert("RGB")
    inputs = processor(images=image, text=args.question, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
        )

    text = processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )
    print("QUESTION:", args.question)
    print("ANSWER:", text)


if __name__ == "__main__":
    main()
````

---

### 4.11. Test minimi

#### `tests/test_config_and_loading.py`

```python
import pytest

from pathology_llava import (
    PathologyLLaVAConfig,
    PathologyLLaVAForConditionalGeneration,
    PathologyLLaVAProcessor,
)


@pytest.mark.parametrize("model_type", ["pathology-llava"])
def test_config_roundtrip(model_type: str) -> None:
    cfg = PathologyLLaVAConfig()
    assert cfg.model_type == model_type

    # Save / load from dict
    cfg_dict = cfg.to_dict()
    cfg2 = PathologyLLaVAConfig(**cfg_dict)
    assert cfg2.model_type == model_type


def test_processor_instantiation(tmp_path) -> None:
    # This only checks Processor class instantiation, not HF downloads.
    # We skip if internet is not available.
    pytest.importorskip("transformers")

    # In real CI you would mock HF calls; here we just check the class import.
    assert hasattr(PathologyLLaVAProcessor, "__call__")


@pytest.mark.slow
def test_dummy_forward(monkeypatch) -> None:
    """
    Minimal smoke test. This would normally mock PLIP/LLM/projector
    to avoid downloading heavy models.
    """
    import torch

    cfg = PathologyLLaVAConfig(
        vision_tower_name_or_path="vinid/plip",
        llm_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    # In real tests you should monkeypatch the underlying HF loading
    # to use tiny random models. Here we only verify that the class
    # can be instantiated.
    monkeypatch.setattr(
        "pathology_llava.modeling_pathology_llava.CLIPModel.from_pretrained",
        lambda *a, **k: DummyVisionModel(),
    )
    monkeypatch.setattr(
        "pathology_llava.modeling_pathology_llava.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyLLM(),
    )
    monkeypatch.setattr(
        "pathology_llava.modeling_pathology_llava.AutoModel.from_pretrained",
        lambda *a, **k: DummyProjector(),
    )

    model = PathologyLLaVAForConditionalGeneration(cfg)

    pixel_values = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 100, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    assert out.logits.shape[0] == 2


class DummyVisionModel:
    dtype = torch.float16

    def __init__(self):
        self.vision_model = self

    def __call__(self, pixel_values, output_hidden_states):
        bsz = pixel_values.shape[0]
        seq_len = 197
        hidden_dim = 768
        hidden = torch.randn(bsz, seq_len, hidden_dim, dtype=pixel_values.dtype)
        return type("Out", (), {"hidden_states": [hidden] * 5})


class DummyLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"use_cache": False})
        self.embed = torch.nn.Embedding(100, 4096)
        self.proj = torch.nn.Linear(4096, 100)

    def get_input_embeddings(self):
        return self.embed

    @property
    def dtype(self):
        return self.embed.weight.dtype

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self.proj(inputs_embeds)
        return type("Out", (), {"logits": logits, "loss": None})


class DummyProjector(torch.nn.Module):
    def forward(self, ori_pixel, patch_pixel, image_atts=None):
        # Return something that looks like last_hidden_state.
        bsz = ori_pixel.shape[0]
        seq_len = 16
        hidden_dim = 4096
        last_hidden_state = torch.randn(bsz, seq_len, hidden_dim, dtype=ori_pixel.dtype)
        return type("Out", (), {"last_hidden_state": last_hidden_state})
```

---

5. **Checklist di fattibilità**

* **Dipendenze tecniche**

  * Python ≥ 3.9, PyTorch ≥ 2.1, transformers ≥ 4.40.
  * Accesso a HF per scaricare `vinid/plip`, `OpenFace-CQUPT/Pathology-LLaVA`, `meta-llama/Meta-Llama-3-8B-*`.

* **Rischi / punti da rifinire**

  * La logica `_encode_images` + `build_multimodal_inputs` è **approssimativa** rispetto alla pipeline originale PA-LLaVA (che usa Q-Former e template specifici). Dovrai riallinearla leggendo il loro `model.py`.
  * Per test seri serve mocking sistematico dei caricamenti HF per evitare download giganti in CI.
  * Questo wrapper non replica necessariamente performance identiche al paper; è un ponte pratico, non un porting 1:1 certificato.

* **Licenze**

  * Il codice tuo è pulito sotto Apache-2.0.
  * L’uso di PLIP rimane nel perimetro “research output”.
  * LLaMA-3 richiede di includere l’avviso di attribuzione in NOTICE.

---

6. **Prossime azioni operative**

7. Creare il repo GitHub `pathology-llava` con la struttura indicata e copiare i file skeleton.

8. Copiare il testo completo di Apache-2.0 in `LICENSE` dal sito ufficiale.

9. Implementare uno script opzionale `pathology_llava_download_weights` per scaricare e decomprimere Pathology-LLaVA in una root locale.

10. Allineare `_encode_images` e la chiamata al `projector` leggendo con calma il `model.py` di PA-LLaVA per avvicinarti al comportamento reale.

11. Creare il repo HF `stefano-roy/Pathology-LLaVA-hf`, aggiungere `config.json` + i tre file Python, e verificare che `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` funzioni end-to-end.

---

7. **Bibliografia essenziale**

* OpenFace-CQUPT – Pathology-LLaVA: code + weights (Apache-2.0).
* PLIP (`vinid/plip`) – CLIP-style vision model, research output.
* Meta LLaMA 3 – Community License Agreement.
* XTuner – Apache-2.0 toolkit used by PA-LLaVA.

Se vuoi, il prossimo passo può essere: rifiniamo `_encode_images` per aderire esattamente alla forma dei tensori che hai già ispezionato a HPC.
