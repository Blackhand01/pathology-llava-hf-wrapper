# Repository Code Trace

This file contains all Python code and special files from the repository, traced in style 'file.ext code << >>'.

# Makefile

<<
# Makefile per pathology-llava-hf-wrapper

# Usa pyenv (es. pyenv shell 3.12.7) per scegliere il Python giusto prima di chiamare make.
PYTHON      ?= python
VENV_DIR    ?= .venv
PYTHON_VENV := $(VENV_DIR)/bin/python
PIP_VENV    := $(VENV_DIR)/bin/pip
PKG         := pathology_llava
PYPROJECT   := pyproject.toml

# ========================
# Help
# ========================

.PHONY: help
help:
	@echo "Targets disponibili:"
	@echo "  make venv                 - crea il virtualenv locale (.venv)"
	@echo "  make env-info             - mostra come attivare il venv"
	@echo "  make install              - installa il wrapper in editable mode nel venv"
	@echo "  make install-xtuner       - installa XTuner (necessario per prepare-projector)"
	@echo "  make deps-update-pyproject- aggiorna le dependencies in pyproject.toml dal venv"
	@echo "  make download             - scarica snapshot Pathology-LLaVA da Hugging Face"
	@echo "  make prepare-projector    - converte i pesi con xtuner e prepara il projector"
	@echo "  make build-vlm            - costruisce solo il VLM (PLIP + projector + LLaMA)"
	@echo "  make smoke IMAGE=...      - test end-to-end su una singola immagine"
	@echo "  make test-patches HISTO_IMAGE=... - confronto patch bianca/nera vs patch istologica"
	@echo "  make test                 - esegue la test suite (pytest)"
	@echo "  make bootstrap            - esegue tutta la catena per avere il wrapper pronto all'uso"

# ========================
# Virtualenv
# ========================

$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)

.PHONY: venv
venv: $(VENV_DIR)
	@echo "Virtualenv creato in $(VENV_DIR)"

.PHONY: env-info
env-info:
	@echo "Per attivare il virtualenv nella shell corrente:"
	@echo ""
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "Python nel venv: $(PYTHON_VENV)"
	@echo "Pip nel venv:    $(PIP_VENV)"

# ========================
# Install / deps
# ========================

.PHONY: install
install: venv
	$(PYTHON_VENV) -m pip install -U pip
	$(PYTHON_VENV) -m pip install -e .

.PHONY: install-xtuner
install-xtuner: venv
	# XTuner è richiesto solo per prepare-projector.
	# Docs: https://xtuner.readthedocs.io/en/stable/get_started/installation.html
	$(PYTHON_VENV) -m pip install -U xtuner

# Aggiorna automaticamente le dependencies in pyproject.toml
# ATTENZIONE: prende il set di pacchetti dal venv (pip freeze) e li scrive in [project].dependencies
# in forma "pkg==version". È intenzionalmente brutale: perfetto per avere un env replicabile,
# meno elegante come packaging. Modificalo se vuoi strategie più fini.
.PHONY: deps-update-pyproject
deps-update-pyproject: venv
	$(PYTHON_VENV) -m pip install tomli_w >/dev/null 2>&1 || true
	$(PYTHON_VENV) scripts/update_pyproject_deps.py $(PYTHON_VENV) $(PYPROJECT)

# ========================
# Pathology-LLaVA snapshot & projector
# ========================

.PHONY: download
download: venv
	$(PYTHON_VENV) -m $(PKG).cli download

.PHONY: prepare-projector
prepare-projector: venv install-xtuner
	$(PYTHON_VENV) -m $(PKG).cli prepare-projector

.PHONY: build-vlm
build-vlm: venv
	$(PYTHON_VENV) -m $(PKG).cli build-vlm

# ========================
# Inference helpers
# ========================

# Esempio:
#   make smoke IMAGE=src/img/HP20.2506_13_2626_10284.png PROMPT="Describe the histologic features."
.PHONY: smoke
smoke: venv
ifndef IMAGE
	$(error Devi passare IMAGE=/path/to/patch.png)
endif
	$(PYTHON_VENV) -m $(PKG).cli smoke \
		--image "$(IMAGE)" \
		--prompt "$(or $(PROMPT),Describe the histologic features in this patch.)"

# Esempio:
#   make test-patches HISTO_IMAGE=src/img/HP20.2506_13_2626_10284.png
.PHONY: test-patches
test-patches: venv
ifndef HISTO_IMAGE
	$(error Devi passare HISTO_IMAGE=/path/to/histo_patch.png)
endif
	$(PYTHON_VENV) -m $(PKG).cli test-patches \
		--histo-image "$(HISTO_IMAGE)" \
		--prompt "$(or $(PROMPT),Describe the histologic features in this patch.)"

# ========================
# Dev / test
# ========================

.PHONY: test
test: venv
	$(PYTHON_VENV) -m pip install -U pytest >/dev/null 2>&1 || true
	$(PYTHON_VENV) -m pytest -q

# ========================
# Bootstrap completo
# ========================

# Esegue la catena completa:
#   1) venv
#   2) install (wrapper)
#   3) install-xtuner
#   4) download snapshot HF
#   5) prepare-projector
#   6) build-vlm
#
# Dopo questo, il VLM è pronto per essere usato e il repo è praticamente "ready to run" per l'open source.
.PHONY: bootstrap
bootstrap: venv install install-xtuner download prepare-projector build-vlm
	@echo ""
	@echo "[bootstrap] Setup completato."
	@echo " - Virtualenv: $(VENV_DIR)"
	@echo " - Modello Pathology-LLaVA scaricato e projector preparato."
	@echo " - VLM costruito e pronto all'uso."
	@echo ""
	@echo "Per attivare il venv manualmente:"
	@echo "  source $(VENV_DIR)/bin/activate"
>>

# scripts/update_pyproject_deps.py

<<
#!/usr/bin/env python3
"""
Script to update pyproject.toml dependencies from a virtual environment.

Usage: python scripts/update_pyproject_deps.py <python_venv_path> <pyproject_path>

This script reads the current pyproject.toml, gets the installed packages from
the specified virtual environment using pip freeze, and updates the
[project].dependencies section with the frozen versions.

WARNING: This is intentionally brute-force. It replaces the entire dependencies
list with what's currently installed in the venv. Use with caution.
"""

import subprocess
import pathlib
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/update_pyproject_deps.py <python_venv_path> <pyproject_path>")
        sys.exit(1)

    python_venv = sys.argv[1]
    pyproject_path = pathlib.Path(sys.argv[2])

    if not pyproject_path.exists():
        print(f"pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    # Install tomli_w if not available
    try:
        import tomli_w
    except ImportError:
        subprocess.check_call([python_venv, "-m", "pip", "install", "tomli_w"])
        import tomli_w

    try:
        import tomllib
    except ImportError:
        # Python < 3.11
        subprocess.check_call([python_venv, "-m", "pip", "install", "tomli"])
        import tomli as tomllib

    # Read pyproject.toml
    data = tomllib.loads(pyproject_path.read_text())

    # Get requirements from venv
    out = subprocess.check_output([python_venv, "-m", "pip", "freeze"], text=True)
    deps = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            continue
        # Skip self-dependency (adjust package name if needed)
        if line.lower().startswith("pathology-llava"):
            continue
        deps.append(line)

    # Update project.dependencies
    project = data.setdefault("project", {})
    project["dependencies"] = sorted(deps)

    # Write back
    pyproject_path.write_text(tomli_w.dumps(data))
    print(f"Updated [project].dependencies in {pyproject_path}")

if __name__ == "__main__":
    main()
>>

# src/pathology_llava/__init__.py

<<
from .configuration_pathology_llava import PathologyLLaVAConfig
from .modeling_pathology_llava import PathologyLLaVAForConditionalGeneration
from .processing_pathology_llava import PathologyLLaVAProcessor

__all__ = [
    "PathologyLLaVAConfig",
    "PathologyLLaVAForConditionalGeneration",
    "PathologyLLaVAProcessor",
]
>>

# src/pathology_llava/cli.py

<<
#!/usr/bin/env python3
"""
CLI utilities for the Pathology-LLaVA HF wrapper.

Sub-commands:

  - download
      Scarica dal repo ufficiale HF (OpenFace-CQUPT/Pathology-LLaVA) e
      scompatta gli archivi .tar.gz in una cache locale dedicata.

  - prepare-projector
      Usa xtuner per eseguire:

        xtuner convert pth_to_hf pallava_instruction_tuning.py \
          ./instruction_tuning_weight.pth ./instruction_tuning_weight_ft

      e crea quindi la directory
        instruction_tuning_weight_ft/projector
      che il wrapper si aspetta.

  - smoke
      Pipeline end-to-end minimale:
        (1) download + prepare-projector (se necessario)
        (2) carica il wrapper PathologyLLaVAForConditionalGeneration
        (3) esegue una generazione immagine+testo su una singola immagine.

  - build-vlm
      Verifica l’installazione end-to-end: si assicura che il projector
      esista (eseguendo download/convert se necessario) e istanzia
      modello + processor senza fare inferenza. Utile come smoke test
      “solo costruzione VLM”.

  - test-patches
      Confronto minimale su tre patch:
        (1) patch bianca 224x224
        (2) patch nera 224x224
        (3) patch istopatologica reale passata dall’utente

      Usa lo stesso prompt per tutte e stampa le tre risposte una
      sotto l’altra, così puoi vedere subito se il modello reagisce
      in modo diverso a “vuoto” vs “contenuto istopatologico”.

Tutta la cache è sotto:
    $HF_HOME/.cache/huggingface/pathology-llava
  oppure:
    ~/.cache/huggingface/pathology-llava
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from .configuration_pathology_llava import PathologyLLaVAConfig
from .constants import (
    DEFAULT_INSTRUCTION_TUNING_SUBFOLDER,
    DEFAULT_PATHOLOGY_LLAVA_REPO,
    DEFAULT_PROJECTOR_SUBFOLDER,
)
from .modeling_pathology_llava import PathologyLLaVAForConditionalGeneration
from .processing_pathology_llava import PathologyLLaVAProcessor
from .utils.downloading import _default_cache_root


def download_raw(
    cache_root: Optional[Path] = None,
    repo_id: str = DEFAULT_PATHOLOGY_LLAVA_REPO,
) -> Path:
    """
    Scarica il repo ufficiale HF in una cache dedicata e scompatta:
      - code.tar.gz  -> <snapshot>/code/
      - instruction_tuning_weight.pth.tar.gz -> <snapshot>/instruction_tuning_weight.pth

    Restituisce il path della snapshot (root) da usare come pathology_llava_root.
    """
    cache_root = cache_root or _default_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root),
        )
    )

    # Estrai il codice degli autori (repo PA-LLaVA) in <snapshot>/code
    code_archive = snapshot_path / "code.tar.gz"
    code_dir = snapshot_path / "code"
    if code_archive.exists() and not code_dir.exists():
        import tarfile

        code_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(code_archive, "r:gz") as tf:
            tf.extractall(code_dir)

    # Estrai i pesi instruction_tuning_weight.pth
    inst_archive = snapshot_path / "instruction_tuning_weight.pth.tar.gz"
    inst_file = snapshot_path / "instruction_tuning_weight.pth"
    if inst_archive.exists() and not inst_file.exists():
        import tarfile

        with tarfile.open(inst_archive, "r:gz") as tf:
            tf.extractall(snapshot_path)

    return snapshot_path


def prepare_projector(
    root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
    repo_id: str = DEFAULT_PATHOLOGY_LLAVA_REPO,
) -> Path:
    """
    Prepara il projector di Pathology-LLaVA usando xtuner, se non esiste già.

    Passi:
      1) download_raw(...) se root è None
      2) trova pallava_instruction_tuning.py sotto <root>/code
      3) esegue:
           xtuner convert pth_to_hf pallava_instruction_tuning.py \
             <root>/instruction_tuning_weight.pth \
             <root>/instruction_tuning_weight_ft
      4) controlla che esista <root>/instruction_tuning_weight_ft/projector
    """
    root = Path(root) if root is not None else download_raw(cache_root=cache_root, repo_id=repo_id)

    projector_dir = root / DEFAULT_PROJECTOR_SUBFOLDER
    if projector_dir.exists():
        print(f"[pathology-llava] Projector già presente in: {projector_dir}")
        return projector_dir

    if shutil.which("xtuner") is None:
        raise RuntimeError(
            "xtuner non trovato nel PATH. "
            "Installa xtuner (e le sue dipendenze) per preparare il projector di Pathology-LLaVA."
        )

    weights_pth = root / "instruction_tuning_weight.pth"
    if not weights_pth.exists():
        raise FileNotFoundError(
            f"File pesi instruction_tuning_weight.pth non trovato in {root}. "
            "Esegui prima `download` oppure verifica il contenuto della cache."
        )

    code_dir = root / "code"
    if not code_dir.exists():
        raise FileNotFoundError(
            f"Directory 'code' non trovata in {root}. "
            "Esegui prima `download` per estrarre code.tar.gz."
        )

    candidates = list(code_dir.rglob("pallava_instruction_tuning.py"))
    if not candidates:
        raise FileNotFoundError(
            f"Impossibile trovare pallava_instruction_tuning.py sotto {code_dir}.\n"
            "Controlla la struttura estratta da code.tar.gz."
        )
    config_path = candidates[0]

    ft_dir = root / DEFAULT_INSTRUCTION_TUNING_SUBFOLDER

    cmd = [
        "xtuner",
        "convert",
        "pth_to_hf",
        str(config_path),
        str(weights_pth),
        str(ft_dir),
    ]
    print("[pathology-llava] Eseguo:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(code_dir))

    projector_dir = root / DEFAULT_PROJECTOR_SUBFOLDER
    if not projector_dir.exists():
        raise FileNotFoundError(
            f"xtuner è terminato ma non esiste {projector_dir}. "
            "Controlla i log di xtuner e la cartella instruction_tuning_weight_ft."
        )

    print(f"[pathology-llava] Projector pronto in: {projector_dir}")
    print(
        "[pathology-llava] Suggerimento: esporta PATHOLOGY_LLAVA_ROOT per riusare questa cache, ad es.\n"
        f'  export PATHOLOGY_LLAVA_ROOT="{root}"'
    )
    return projector_dir


def _load_model_and_processor(root: Path) -> Tuple[PathologyLLaVAForConditionalGeneration, PathologyLLaVAProcessor]:
    """
    Costruisce config, modello e processor a partire da una root locale
    che contiene il projector (instruction_tuning_weight_ft/projector).
    """
    cfg = PathologyLLaVAConfig(pathology_llava_root=str(root))
    model = PathologyLLaVAForConditionalGeneration(cfg)

    image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_tower_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm_name_or_path,
        use_fast=True,
        padding_side="right",
    )
    processor = PathologyLLaVAProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return model, processor


def _run_inference(
    model: PathologyLLaVAForConditionalGeneration,
    processor: PathologyLLaVAProcessor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """
    Helper riutilizzabile per una singola chiamata immagine+testo.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
        )

    text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    return text.strip()


def _cmd_download(args: argparse.Namespace) -> None:
    root = download_raw()
    print(f"[pathology-llava] Snapshot scaricata/aggiornata in: {root}")


def _cmd_prepare(args: argparse.Namespace) -> None:
    prepare_projector()


def _cmd_build_vlm(args: argparse.Namespace) -> None:
    """
    Comando “solo costruzione VLM”: prepara il projector se serve
    e istanzia modello + processor. Non fa inferenza.
    """
    projector_dir = prepare_projector()
    root = projector_dir.parent.parent  # <root>/instruction_tuning_weight_ft/projector -> <root>

    model, processor = _load_model_and_processor(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("[pathology-llava] VLM istanziato correttamente.")
    print(f"[pathology-llava]   vision tower: {model.config.vision_tower_name_or_path}")
    print(f"[pathology-llava]   LLM:          {model.config.llm_name_or_path}")
    print(f"[pathology-llava]   root pesi:    {root}")
    _ = processor  # solo per esplicitare che il processor è costruito


def _cmd_smoke(args: argparse.Namespace) -> None:
    # 1) Download + projector
    projector_dir = prepare_projector()
    root = projector_dir.parent.parent  # <root>/instruction_tuning_weight_ft/projector -> <root>

    # 2) Modello + processor
    model, processor = _load_model_and_processor(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    text = _run_inference(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Pathology-LLaVA output ===")
    print(text)


def _cmd_test_patches(args: argparse.Namespace) -> None:
    """
    Confronto rapido tra:
      - patch bianca 224x224,
      - patch nera 224x224,
      - patch istopatologica reale.
    """
    # 1) Download + projector
    projector_dir = prepare_projector()
    root = projector_dir.parent.parent

    # 2) Modello + processor
    model, processor = _load_model_and_processor(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    size = args.size
    prompt = args.prompt

    # Patch sintetiche vuote
    white_patch = Image.new("RGB", (size, size), color=(255, 255, 255))
    black_patch = Image.new("RGB", (size, size), color=(0, 0, 0))

    # Patch istopatologica reale
    histo = Image.open(args.histo_image).convert("RGB")
    if histo.size != (size, size) and not args.no_resize:
        histo = histo.resize((size, size))

    print("=== White patch (vuota) ===")
    out_white = _run_inference(
        model=model,
        processor=processor,
        image=white_patch,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(out_white)

    print("\n=== Black patch (vuota) ===")
    out_black = _run_inference(
        model=model,
        processor=processor,
        image=black_patch,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(out_black)

    print("\n=== Patch istopatologica ===")
    out_histo = _run_inference(
        model=model,
        processor=processor,
        image=histo,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(out_histo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pathology-llava-cli", description="Utility CLI per Pathology-LLaVA HF wrapper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Scarica il repo HF ufficiale e scompatta gli archivi.")
    p_download.set_defaults(func=_cmd_download)

    p_prepare = subparsers.add_parser(
        "prepare-projector",
        help="Prepara il projector di Pathology-LLaVA usando xtuner (instruction_tuning_weight_ft/projector).",
    )
    p_prepare.set_defaults(func=_cmd_prepare)

    p_smoke = subparsers.add_parser(
        "smoke",
        help="Esegue un test end-to-end immagine+testo usando il wrapper Pathology-LLaVA.",
    )
    p_smoke.add_argument("--image", required=True, help="Path locale di una patch/immagine di test.")
    p_smoke.add_argument(
        "--prompt",
        default="Describe the pathology in this patch.",
        help="Prompt testuale da usare per la generazione.",
    )
    p_smoke.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Numero massimo di token generati.",
    )
    p_smoke.set_defaults(func=_cmd_smoke)

    p_build = subparsers.add_parser(
        "build-vlm",
        help="Prepara il projector (se serve) e istanzia il VLM senza eseguire inferenza.",
    )
    p_build.set_defaults(func=_cmd_build_vlm)

    p_test = subparsers.add_parser(
        "test-patches",
        help=(
            "Confronta le risposte del modello su patch vuote (bianca/nera) "
            "e su una patch istopatologica reale."
        ),
    )
    p_test.add_argument(
        "--histo-image",
        required=True,
        help="Path locale di una patch istopatologica (idealmente 224x224).",
    )
    p_test.add_argument(
        "--prompt",
        default="Describe the pathology in this patch.",
        help="Prompt testuale da usare per tutte le patch.",
    )
    p_test.add_argument(
        "--size",
        type=int,
        default=224,
        help="Lato in pixel delle patch sintetiche bianca/nera (default: 224).",
    )
    p_test.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Numero massimo di token generati.",
    )
    p_test.add_argument(
        "--no-resize",
        action="store_true",
        help="Non ridimensionare la patch istopatologica alla dimensione specificata.",
    )
    p_test.set_defaults(func=_cmd_test_patches)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
>>

# src/pathology_llava/configuration_pathology_llava.py

<<
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
>>

# src/pathology_llava/constants.py

<<
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
>>

# src/pathology_llava/modeling_pathology_llava.py

<<
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
>>

# src/pathology_llava/processing_pathology_llava.py

<<
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
>>

# src/pathology_llava/utils/__init__.py

<<
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
>>

# src/pathology_llava/utils/downloading.py

<<
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
>>

# src/pathology_llava/utils/multimodal.py

<<
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
>>

# src/pathology_llava/utils/vision.py

<<
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
>>

# tests/__init__.py

<<
>>

# tests/test_config_and_loading.py

<<
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
    # Questo test verifica solo che la classe Processor sia importabile.
    # In CI reale andrebbe mockato l'accesso a HF.
    pytest.importorskip("transformers")

    assert hasattr(PathologyLLaVAProcessor, "__call__")


@pytest.mark.slow
def test_dummy_forward(monkeypatch) -> None:
    """
    Minimal smoke test. Mocka PLIP/LLM/projector per evitare download pesanti.
    """
    import torch

    cfg = PathologyLLaVAConfig(
        vision_tower_name_or_path="vinid/plip",
        llm_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    )

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
        bsz = ori_pixel.shape[0]
        seq_len = 16
        hidden_dim = 4096
        last_hidden_state = torch.randn(bsz, seq_len, hidden_dim, dtype=ori_pixel.dtype)
        return type("Out", (), {"last_hidden_state": last_hidden_state})
>>

# tests/test_dummy_forward.py

<<
import pytest
import torch

from pathology_llava import PathologyLLaVAConfig, PathologyLLaVAForConditionalGeneration


@pytest.mark.slow
def test_dummy_forward(monkeypatch) -> None:
    """
    Minimal smoke test. This would normally mock PLIP/LLM/projector
    to avoid downloading heavy models.
    """
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
>>

# trace_code.py

<<
#!/usr/bin/env python3
"""
Script to generate an MD file with all Python code and special files from the repository,
formatted as "file.ext code << >>".
"""

import os
import pathlib

EXTRA_FILES = {'Makefile'}  # Add special files here, e.g. {'Makefile'}

def main():
    # Output MD file
    output_file = "repository_code.md"

    # Root directory (current working directory)
    root_dir = pathlib.Path(".")

    # Collect all Python files
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ and other irrelevant dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                py_files.append(pathlib.Path(root) / file)

    # Add extra files
    extra_files = [pathlib.Path(f) for f in EXTRA_FILES if pathlib.Path(f).exists()]
    all_files = py_files + extra_files

    # Sort files for consistent order
    all_files.sort()

    # Write to MD file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Repository Code Trace\n\n")
        f.write("This file contains all Python code and special files from the repository, traced in style 'file.ext code << >>'.\n\n")

        for file_path in all_files:
            # Get relative path
            rel_path = file_path.relative_to(root_dir)
            f.write(f"# {rel_path}\n\n")
            f.write("<<\n")

            # Read and write code
            try:
                with open(file_path, 'r', encoding='utf-8') as code_f:
                    code = code_f.read()
                    f.write(code)
            except Exception as e:
                f.write(f"# Error reading file: {e}\n")

            f.write(">>\n\n")

    print(f"Generated {output_file} with {len(all_files)} files.")


if __name__ == "__main__":
    main()
>>

