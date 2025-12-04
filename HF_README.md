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
```

If you want to keep all weights local and avoid automatic downloads, set:

* `PATHOLOGY_LLAVA_VISION_NAME_OR_PATH` to your local PLIP path,
* `PATHOLOGY_LLAVA_LLM_NAME_OR_PATH` to your local LLaMA-3 path,
* `PATHOLOGY_LLAVA_ROOT` to your extracted Pathology-LLaVA directory.
