#!/usr/bin/env python3
"""
Bootstrap per il progetto pathology-llava-hf-wrapper.

FASI:
  1. Rileva root del repo e contesto.
  2. Valida struttura progetto (pyproject.toml, src/).
  3. Crea o riusa virtualenv locale (.pa-llava-venv) e aggiorna pip.
  4. Installa il wrapper in editable mode (pip install -e .).
  5. Autodiagnosi (python -V, pip list).
  6. Clona i repo esterni PA-LLaVA e PLIP.
  7. Scarica pesi/asset da Hugging Face (OpenFace-CQUPT/Pathology-LLaVA),
     prova a estrarre PLIP e a individuare i pesi di domain alignment,
     e salva tutti i path in models/assets_paths.json.

Esecuzione tipica (dalla cartella src/ o root del repo):

  python bootstrap_pathology_llava.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------
# Utils di stampa
# ---------------------------------------------------------------------


def print_header(title: str) -> None:
    bar = "=" * 68
    print()
    print(bar)
    print(f"[FASE] {title}")
    print(bar)


def print_sub(msg: str) -> None:
    print(f"  - {msg}")


# ---------------------------------------------------------------------
# Helper per subprocess con autolog
# ---------------------------------------------------------------------


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def run(
    cmd: Iterable[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
) -> CmdResult:
    cwd_str = str(cwd) if cwd is not None else None
    cmd_list = list(cmd)

    print()
    print(f">>> Eseguo (cwd={cwd_str or os.getcwd()}):")
    print("    " + " ".join(cmd_list))

    completed = subprocess.run(
        cmd_list,
        cwd=cwd_str,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        # non filtriamo: se c'è rumore, lo vedi
        print(completed.stderr, end="", file=sys.stderr)

    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd_list, completed.stdout, completed.stderr
        )

    return CmdResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


# ---------------------------------------------------------------------
# FASE 1: rilevazione root repo e contesto
# ---------------------------------------------------------------------


@dataclass
class Context:
    script_path: Path
    repo_root: Path
    src_dir: Path
    pyproject: Path
    venv_dir: Path
    venv_python: Path


def detect_repo_root() -> Context:
    print_header("1) Rilevazione della root del repository e configurazione contesto")

    script_path = Path(__file__).resolve()
    start_dir = script_path.parent
    print_sub(f"Script: {script_path}")
    print_sub(f"Directory di partenza per la ricerca della root: {start_dir}")

    # risali finché non trovi pyproject.toml o .git
    current = start_dir
    repo_root = None
    while True:
        pyproj = current / "pyproject.toml"
        git_dir = current / ".git"
        if pyproj.exists() or git_dir.exists():
            repo_root = current
            break
        if current.parent == current:
            break
        current = current.parent

    if repo_root is None:
        # fallback: assume che la root sia la dir dello script se non trova niente
        repo_root = start_dir
        print_sub(
            "ATTENZIONE: pyproject.toml/.git non trovati risalendo le directory; "
            "uso la directory dello script come root."
        )

    pyproject = repo_root / "pyproject.toml"
    src_dir = repo_root / "src"

    venv_dir = repo_root / ".pa-llava-venv"
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    print_sub(f"Root del repository rilevata: {repo_root}")
    print_sub(f"Virtualenv previsto: {venv_dir}")
    print_sub(f"Eseguibile Python nel venv: {venv_python}")

    return Context(
        script_path=script_path,
        repo_root=repo_root,
        src_dir=src_dir,
        pyproject=pyproject,
        venv_dir=venv_dir,
        venv_python=venv_python,
    )


# ---------------------------------------------------------------------
# FASE 2: validazione struttura progetto
# ---------------------------------------------------------------------


def validate_project(ctx: Context) -> None:
    print_header("2) Validazione struttura progetto (pyproject.toml, cartelle, ecc.)")

    if not ctx.pyproject.exists():
        print_sub(f"ERRORE: pyproject.toml non trovato in {ctx.repo_root}")
        raise SystemExit(1)
    print_sub(f"Trovato pyproject.toml: {ctx.pyproject}")

    # check sezioni base senza portarsi dietro librerie TOML
    txt = ctx.pyproject.read_text(encoding="utf-8")
    has_project = "[project]" in txt
    has_build = "[build-system]" in txt

    if not has_project or not has_build:
        print_sub(
            "ERRORE: pyproject.toml non contiene entrambe le sezioni [project] e [build-system]."
        )
        raise SystemExit(1)
    print_sub("pyproject.toml: sezioni [project] e [build-system] presenti.")

    if not ctx.src_dir.exists():
        print_sub(f"ERRORE: directory sorgenti non trovata: {ctx.src_dir}")
        raise SystemExit(1)
    print_sub(f"Directory sorgenti trovata: {ctx.src_dir}")


# ---------------------------------------------------------------------
# FASE 3: virtualenv
# ---------------------------------------------------------------------


def ensure_venv(ctx: Context) -> None:
    print_header("3) Creazione o riutilizzo del virtualenv locale (.pa-llava-venv)")

    if ctx.venv_dir.exists() and ctx.venv_python.exists():
        print_sub("Virtualenv già presente, lo riutilizzo.")
    else:
        print_sub("Virtualenv non trovato, creazione in corso...")
        run(
            [sys.executable, "-m", "venv", str(ctx.venv_dir)],
            cwd=ctx.repo_root,
            check=True,
        )
        print_sub("Virtualenv creato con successo.")

    if not ctx.venv_python.exists():
        print_sub(
            f"ERRORE: python nel venv non trovato in {ctx.venv_python}. "
            "Probabile virtualenv corrotto: cancella .pa-llava-venv e riprova."
        )
        raise SystemExit(1)

    print_sub(f"Python del venv: {ctx.venv_python}")
    print_sub("Aggiornamento pip nel virtualenv...")

    run(
        [str(ctx.venv_python), "-m", "pip", "install", "-U", "pip"],
        cwd=ctx.repo_root,
        check=True,
    )
    print_sub("pip aggiornato nel virtualenv.")


# ---------------------------------------------------------------------
# FASE 4: installazione wrapper
# ---------------------------------------------------------------------


def install_wrapper_editable(ctx: Context) -> None:
    print_header("4) Installazione del wrapper in modalità editabile (pip install -e .)")

    run(
        [str(ctx.venv_python), "-m", "pip", "install", "-e", "."],
        cwd=ctx.repo_root,
        check=True,
    )
    print_sub("Wrapper installato in modalità editabile.")


# ---------------------------------------------------------------------
# FASE 5: autodiagnosi
# ---------------------------------------------------------------------


def autodiagnose(ctx: Context) -> None:
    print_header("5) Autodiagnosi finale (versioni e sanity check)")

    print_sub("Versione di Python nel venv:")
    run([str(ctx.venv_python), "-V"], cwd=ctx.repo_root, check=True)

    print_sub("pip list (estratto):")
    run(
        [str(ctx.venv_python), "-m", "pip", "list"],
        cwd=ctx.repo_root,
        check=True,
    )


# ---------------------------------------------------------------------
# FASE 6: clone repo esterni (PA-LLaVA, PLIP)
# ---------------------------------------------------------------------


def clone_if_missing(target_dir: Path, git_url: str) -> None:
    if target_dir.exists():
        print_sub(f"Repo già presente, salto clone: {target_dir}")
        return

    print_sub(f"Clono {git_url} in {target_dir}")
    run(["git", "clone", git_url, str(target_dir)], cwd=target_dir.parent, check=True)
    print_sub("Clone completato.")


def clone_external_repos(ctx: Context) -> None:
    print_header("6) Download codice esterno: PA-LLaVA e PLIP (git clone)")

    external_dir = ctx.repo_root / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    print_sub(f"Cartella base per repo esterni: {external_dir}")

    pa_llava_dir = external_dir / "PA-LLaVA"
    plip_dir = external_dir / "plip"

    # PA-LLaVA
    clone_if_missing(
        pa_llava_dir, "https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA.git"
    )

    # PLIP
    clone_if_missing(
        plip_dir, "https://github.com/PathologyFoundation/plip.git"
    )

    print_sub("Codice PA-LLaVA e PLIP disponibile sotto 'external/'.")


# ---------------------------------------------------------------------
# FASE 7: download pesi / asset da Hugging Face
# ---------------------------------------------------------------------


def find_largest_file(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def extract_plip_archive(hf_dir: Path, models_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    # Cerca plip*.tar.gz nel dump HF
    archives = list(hf_dir.rglob("plip*.tar.gz"))
    if not archives:
        print_sub("WARN: nessun archivio 'plip*.tar.gz' trovato nel repo HF.")
        return None, None

    archive = archives[0]
    print_sub(f"Trovato archivio PLIP: {archive}")

    plip_target = models_dir / "plip"
    plip_target.mkdir(parents=True, exist_ok=True)

    # se la cartella non è vuota assumiamo già estratto
    if any(plip_target.iterdir()):
        print_sub(f"PLIP sembra già estratto in {plip_target}, non tocco nulla.")
        return archive, plip_target

    print_sub(f"Estraggo PLIP in: {plip_target}")
    try:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(plip_target)
    except Exception as e:
        print_sub(f"ERRORE durante l'estrazione di {archive}: {e}")
        return archive, None

    print_sub("Estrazione PLIP completata.")
    return archive, plip_target


def download_hf_assets(ctx: Context) -> None:
    print_header(
        "7) Download pesi da Hugging Face (OpenFace-CQUPT/Pathology-LLaVA) e configurazione path"
    )

    models_dir = ctx.repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print_sub(f"Cartella modelli: {models_dir}")

    hf_target = models_dir / "pathology-llava-hf"
    print_sub(f"Directory di destinazione per il repo HF: {hf_target}")

    # Usa il Python del venv per avere huggingface_hub già installato
    code = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "repo_id='OpenFace-CQUPT/Pathology-LLaVA', "
        f"local_dir=r'{hf_target}', "
        "local_dir_use_symlinks=False"
        ")"
    )

    try:
        run([str(ctx.venv_python), "-c", code], cwd=ctx.repo_root, check=True)
    except subprocess.CalledProcessError:
        print_sub(
            "ERRORE: download da Hugging Face fallito.\n"
            "Verifica:\n"
            "  * connessione di rete\n"
            "  * accesso pubblico al repo 'OpenFace-CQUPT/Pathology-LLaVA'\n"
            "  * che 'huggingface_hub' sia installato nel venv\n"
            "Puoi anche scaricare manualmente e posizionare i file in "
            f"{hf_target}, poi rilanciare lo script."
        )
        raise

    print_sub("Download da Hugging Face completato.")

    # -----------------------------------------------------------------
    # Heuristica: trova pesi PLIP e pesi di domain alignment
    # -----------------------------------------------------------------

    # 1. PLIP archivio
    plip_archive, plip_extracted = extract_plip_archive(hf_dir=hf_target, models_dir=models_dir)

    # 2. Domain alignment / pesi principali: prendi il file più grande tra pth/pt/bin/safetensors
    domain_ckpt = find_largest_file(
        hf_target, patterns=["*.pth", "*.pt", "*.bin", "*.safetensors"]
    )
    if domain_ckpt is None:
        print_sub(
            "WARN: nessun file .pth/.pt/.bin/.safetensors trovato nel dump HF. "
            "Dovrai indicare manualmente 'pretrained_pth' nelle config PA-LLaVA."
        )
    else:
        print_sub(f"Checkpoint di dimensione massima trovato (candidato domain alignment): {domain_ckpt}")

    # -----------------------------------------------------------------
    # Salva tutti i path in un JSON per l'uso nelle config
    # -----------------------------------------------------------------
    assets = {
        "hf_repo_dir": str(hf_target),
        "plip_archive": str(plip_archive) if plip_archive is not None else None,
        "plip_extracted_dir": str(plip_extracted) if plip_extracted is not None else None,
        "domain_alignment_checkpoint": str(domain_ckpt) if domain_ckpt is not None else None,
    }

    assets_json = models_dir / "assets_paths.json"
    assets_json.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    print_sub(f"File di configurazione pesi scritto in: {assets_json}")

    print()
    print("=== Riepilogo path da usare nelle config PA-LLaVA ===")
    print(f"  * HF repo locale:           {assets['hf_repo_dir']}")
    print(f"  * PLIP estratto (se c'è):   {assets['plip_extracted_dir']}")
    print(f"  * Domain alignment chkpt:   {assets['domain_alignment_checkpoint']}")
    print()
    print("Esempi per i file di config PA-LLaVA (da adattare):")
    print()
    print("  # pallava_domain_alignment.py / pallava_instruction_tuning.py")
    print("  visual_encoder = 'absolute path of plip'  ->  "
          "imposta a qualcosa tipo:")
    print(f"      visual_encoder = r\"{assets['plip_extracted_dir'] or '/path/alle/pesi/plip'}\"")
    print()
    print("  pretrained_pth = 'absolute path of domain alignment model weight'  ->")
    print(f"      pretrained_pth = r\"{assets['domain_alignment_checkpoint'] or '/path/al/checkpoint.pth'}\"")
    print()
    print("Se l'heuristica non intercetta il file giusto, apri il JSON e i file "
          "dentro 'models/pathology-llava-hf' e scegli tu quello corretto.")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> int:
    try:
        ctx = detect_repo_root()
        validate_project(ctx)
        ensure_venv(ctx)
        install_wrapper_editable(ctx)
        autodiagnose(ctx)
        clone_external_repos(ctx)
        download_hf_assets(ctx)

        print()
        print("Bootstrap COMPLETATO.")
        print("Per usare il wrapper:")
        print(f"  source {ctx.venv_dir}/bin/activate  # (o .\\Scripts\\activate su Windows)")
        print("  python -c \"import pathology_llava; print(pathology_llava.__file__)\"")
        print()
        print("Per allenare/riusare PA-LLaVA:")
        print("  * usa le config nel repo external/PA-LLaVA")
        print("  * sostituisci 'visual_encoder' e 'pretrained_pth' con i path "
              "stampati sopra / presenti in models/assets_paths.json")
        return 0

    except SystemExit as e:
        # già stampato un messaggio chiaro
        return int(e.code or 1)
    except Exception as e:
        print()
        print("ERRORE FATALE durante il bootstrap:")
        print(f"  {type(e).__name__}: {e}")
        print()
        print("Controlla l'output sopra per capire la FASE in cui si è rotto.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
