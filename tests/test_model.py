import torch

from simplellm import ModelConfig, ModernLLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        max_seq_len=16,
        multiple_of=16,
        dropout=0.0,
    )


def test_model_forward_shape():
    model = ModernLLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, 128)


def test_model_forward_with_cache():
    config = tiny_config()
    model = ModernLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    logits, caches = model(input_ids, use_cache=True)

    assert logits.shape == (2, 8, config.vocab_size)
    assert len(caches) == config.n_layers
    assert caches[0][0].shape == (2, 8, config.n_heads, config.dim // config.n_heads)
