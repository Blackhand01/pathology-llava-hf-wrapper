#!/usr/bin/env python
import json
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any


def print_header(title: str) -> None:
    line = "=" * 68
    print()
    print(line)
    print(f"[FASE] {title}")
    print(line)


def load_assets_paths(models_dir: Path) -> Optional[Dict[str, Any]]:
    """Prova a leggere models/assets_paths.json se esiste."""
    config_path = models_dir / "assets_paths.json"
    if not config_path.exists():
        print(f"  - WARN: {config_path} non esiste, procedo comunque con path di default.")
        return None
    try:
        data = json.loads(config_path.read_text())
        print(f"  - Trovato assets_paths.json: {config_path}")
        return data
    except Exception as e:
        print(f"  - WARN: errore nel leggere {config_path}: {e!r}")
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_tar_gz(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  - WARN: archivio non trovato, salto: {src}")
        return
    ensure_dir(dst)
    print(f"  - Estraggo {src} -> {dst}")
    try:
        with tarfile.open(src, "r:gz") as tar:
            tar.extractall(path=dst)
    except Exception as e:
        print(f"  - ERRORE durante l'estrazione di {src.name}: {e!r}")


def main() -> int:
    # ==========================
    # FASE 1: Localizza root e models
    # ==========================
    print_header("1) Rilevazione root repo e directory modelli")

    script_path = Path(__file__).resolve()
    src_dir = script_path.parent
    repo_root = src_dir.parent
    models_dir = repo_root / "models"
    hf_dir = models_dir / "pathology-llava-hf"

    print(f"  - Script:            {script_path}")
    print(f"  - Directory src:     {src_dir}")
    print(f"  - Root repo:         {repo_root}")
    print(f"  - Directory models:  {models_dir}")
    print(f"  - HF dump dir:       {hf_dir}")

    if not hf_dir.exists():
        print("  - ERRORE: directory HF dump non trovata. Atteso:")
        print(f"      {hf_dir}")
        print("    Prima lancia il bootstrap per scaricare lo snapshot HF.")
        return 1

    assets_cfg = load_assets_paths(models_dir)

    # ==========================
    # FASE 2: Estrazione PLIP
    # ==========================
    print_header("2) Estrazione PLIP (plip.tar.gz -> models/plip)")

    plip_tar = hf_dir / "plip.tar.gz"
    plip_dst = models_dir / "plip"

    print(f"  - Archivio PLIP:       {plip_tar}")
    print(f"  - Destinazione PLIP:   {plip_dst}")

    if plip_dst.exists() and any(plip_dst.iterdir()):
        print("  - PLIP sembra già estratto (directory non vuota), non tocco nulla.")
    else:
        extract_tar_gz(plip_tar, plip_dst)

    # ==========================
    # FASE 3: Estrazione codice PA-LLaVA
    # ==========================
    print_header("3) Estrazione codice PA-LLaVA (code.tar.gz -> models/pathology-llava-hf/code)")

    code_tar = hf_dir / "code.tar.gz"
    code_dst = hf_dir / "code"

    print(f"  - Archivio codice:     {code_tar}")
    print(f"  - Destinazione codice: {code_dst}")

    if code_dst.exists() and any(code_dst.iterdir()):
        print("  - Codice PA-LLaVA sembra già estratto (directory non vuota), non tocco nulla.")
    else:
        extract_tar_gz(code_tar, code_dst)

    # ==========================
    # FASE 4: Estrazione pesi (domain_alignment / instruction_tuning)
    # ==========================
    print_header("4) Estrazione pesi (.pth.tar.gz -> models/pathology-llava-hf/weights)")

    weights_dir = hf_dir / "weights"
    ensure_dir(weights_dir)

    # Domain alignment
    da_tar = hf_dir / "domain_alignment_weight.pth.tar.gz"
    # Instruction tuning
    it_tar = hf_dir / "instruction_tuning_weight.pth.tar.gz"

    print(f"  - Archivio domain alignment:  {da_tar}")
    print(f"  - Archivio instr. tuning:    {it_tar}")
    print(f"  - Destinazione pesi:         {weights_dir}")

    # Se nella cartella weights c'è già qualcosa, non sovrascrivo.
    if any(weights_dir.iterdir()):
        print("  - Pesi già presenti in 'weights', non estraggo di nuovo.")
    else:
        extract_tar_gz(da_tar, weights_dir)
        extract_tar_gz(it_tar, weights_dir)

    # ==========================
    # FASE 5: Riepilogo e hint per config
    # ==========================
    print_header("5) Riepilogo path utili per le config PA-LLaVA")

    # Trova eventualmente qualche .pth come esempio
    domain_ckpt = None
    if weights_dir.exists():
        for p in sorted(weights_dir.rglob("*.pth")):
            domain_ckpt = p
            break

    print("  === Path principali ===")
    print(f"  * HF repo locale:       {hf_dir}")
    print(f"  * PLIP estratto:        {plip_dst}")
    print(f"  * Directory pesi:       {weights_dir}")
    print(f"  * Esempio checkpoint:   {domain_ckpt if domain_ckpt else 'Nessun .pth trovato in weights/'}")

    print()
    print("Esempi di come impostare le config PA-LLaVA:")
    print()
    print("  # pallava_domain_alignment.py / pallava_instruction_tuning.py")
    print(f"  visual_encoder = 'absolute path of plip'  ->")
    print(f"      visual_encoder = r\"{plip_dst}\"")
    print()
    print("  pretrained_pth = 'absolute path of domain alignment model weight'  ->")
    if domain_ckpt:
        print(f"      pretrained_pth = r\"{domain_ckpt}\"")
    else:
        print("      # scegli manualmente un file .pth dentro 'weights/'")

    # Aggiorna eventualmente assets_paths.json per salvare questi path
    try:
        cfg = assets_cfg or {}
        cfg.setdefault("hf_repo_local", str(hf_dir))
        cfg.setdefault("plip_dir", str(plip_dst))
        cfg.setdefault("weights_dir", str(weights_dir))
        if domain_ckpt:
            cfg.setdefault("domain_alignment_ckpt", str(domain_ckpt))

        config_path = models_dir / "assets_paths.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        print()
        print(f"  - assets_paths.json aggiornato: {config_path}")
    except Exception as e:
        print(f"  - WARN: impossibile aggiornare assets_paths.json: {e!r}")

    print()
    print("Post-processing COMPLETATO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
