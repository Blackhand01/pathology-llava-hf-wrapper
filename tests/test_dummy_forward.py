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
