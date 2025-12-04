Hugging Face–compatible wrapper around PA-LLaVA / Pathology-LLaVA.

This package exposes a multimodal model
`PathologyLLaVAForConditionalGeneration` and a
`PathologyLLaVAProcessor` that combine:

- PLIP as vision tower (`vinid/plip`),
- a LLaMA-3 language model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`),
- projector / Q-Former and adapters from `OpenFace-CQUPT/Pathology-LLaVA`.

No third-party weights are distributed with this package. All models are
loaded from their official sources at runtime.
