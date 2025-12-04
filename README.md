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

## Intended Use & Limitations

This repository is intended **for research and experimentation only**.

- It is not designed or validated for clinical decision making.
- It should not be used as a stand-alone medical device or diagnostic tool.
- Any deployment in clinical or commercial settings must:

  - comply with the licenses of PLIP, Pathology-LLaVA, and LLaMA-3,
  - undergo appropriate validation, regulatory review, and legal review.

The authors of this wrapper do not provide any warranty on the behaviour
of the underlying models.

## Quickstart

Install:

```bash
pip install pathology-llava
```

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
