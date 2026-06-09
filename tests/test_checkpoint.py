import torch

from simplellm import ModelConfig, ModernLLM, load_checkpoint, save_checkpoint


def tiny_config() -> ModelConfig:
    return ModelConfig(
        dim=32,
        n_layers=1,
        n_heads=4,
        vocab_size=64,
        max_seq_len=8,
        multiple_of=16,
        dropout=0.0,
    )


def test_structured_checkpoint_round_trip(tmp_path):
    config = tiny_config()
    model = ModernLLM(config)
    path = tmp_path / "structured.pt"

    save_checkpoint(path, model, config, tokenizer_name="tiny-tokenizer", step=7)

    loaded_model = ModernLLM(config)
    result = load_checkpoint(path, loaded_model)

    assert result.format == "structured"
    assert result.config == config
    assert result.tokenizer_name == "tiny-tokenizer"
    assert result.step == 7
    assert torch.equal(model.output.weight, loaded_model.output.weight)


def test_raw_state_dict_checkpoint_loads(tmp_path):
    config = tiny_config()
    model = ModernLLM(config)
    path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), path)

    loaded_model = ModernLLM(config)
    result = load_checkpoint(path, loaded_model)

    assert result.format == "state_dict"
    assert result.config is None
    assert torch.equal(model.output.weight, loaded_model.output.weight)
