import torch

from simplellm import ModelConfig, ModernLLM, generate


class TinyTokenizer:
    vocab_size = 32
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = eos_token

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [((ord(char) % (self.vocab_size - 1)) + 1) for char in text]
        if not ids:
            ids = [1]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids):
        return "".join(chr(97 + (int(token_id) % 26)) for token_id in token_ids)


def tiny_model() -> ModernLLM:
    config = ModelConfig(
        dim=32,
        n_layers=1,
        n_heads=4,
        vocab_size=32,
        max_seq_len=8,
        multiple_of=16,
        dropout=0.0,
    )
    return ModernLLM(config)


def test_generation_smoke():
    torch.manual_seed(0)
    output = generate(
        tiny_model(),
        TinyTokenizer(),
        "hi",
        max_new_tokens=2,
        temperature=1.0,
        top_p=0.9,
        device="cpu",
    )

    assert isinstance(output, str)
    assert output


def test_generation_clamps_long_prompt():
    torch.manual_seed(0)
    output = generate(
        tiny_model(),
        TinyTokenizer(),
        "this prompt is much longer than the tiny context",
        max_new_tokens=2,
        temperature=1.0,
        top_p=0.9,
        device="cpu",
    )

    assert isinstance(output, str)
    assert output
