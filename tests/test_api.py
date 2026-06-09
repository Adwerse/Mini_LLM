import torch
from fastapi.testclient import TestClient

from simplellm.api import create_app
from simplellm.config import ModelConfig


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class FakeInference:
    def __init__(self):
        self.device = torch.device("cpu")
        self.checkpoint_loaded = True
        self.checkpoint_path = "fake-checkpoint.pt"
        self.tokenizer_name = "fake-tokenizer"
        self.config = ModelConfig(
            dim=32,
            n_layers=1,
            n_heads=4,
            vocab_size=32,
            max_seq_len=12,
            multiple_of=16,
            dropout=0.0,
        )
        self.tokenizer = FakeTokenizer()
        self.model = torch.nn.Linear(2, 2)

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_p=0.9):
        return f"{prompt} ... generated"


def client_with_fake_model() -> TestClient:
    return TestClient(create_app(inference_service=FakeInference()))


def test_health():
    client = client_with_fake_model()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    client = client_with_fake_model()

    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "Mini Transformer Language Model"
    assert body["device"] == "cpu"
    assert body["checkpoint_loaded"] is True
    assert body["parameters"] == "6"
    assert body["max_seq_len"] == 12
    assert body["tokenizer"] == "fake-tokenizer"


def test_generate_success():
    client = client_with_fake_model()

    response = client.post(
        "/generate",
        json={
            "prompt": "Once",
            "max_new_tokens": 3,
            "temperature": 0.8,
            "top_p": 0.9,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generated_text"] == "Once ... generated"
    assert body["model_info"] == {
        "device": "cpu",
        "checkpoint": "fake-checkpoint.pt",
    }


def test_generate_rejects_empty_prompt():
    client = client_with_fake_model()

    response = client.post("/generate", json={"prompt": "   "})

    assert response.status_code == 422
    assert "prompt must not be empty" in response.text


def test_generate_rejects_invalid_sampling_values():
    client = client_with_fake_model()

    response = client.post(
        "/generate",
        json={"prompt": "Hello", "temperature": 0.0, "top_p": 1.2},
    )

    assert response.status_code == 422


def test_generate_rejects_too_many_tokens_for_model():
    client = client_with_fake_model()

    response = client.post(
        "/generate",
        json={"prompt": "Hello", "max_new_tokens": 13},
    )

    assert response.status_code == 422
    assert "max_new_tokens cannot exceed model max_seq_len" in response.text


def test_generate_rejects_context_length_exceeded():
    client = client_with_fake_model()

    response = client.post(
        "/generate",
        json={"prompt": "0123456789", "max_new_tokens": 3},
    )

    assert response.status_code == 422
    assert "context length exceeded" in response.text


def test_generate_requires_checkpoint():
    client = TestClient(create_app())

    response = client.post("/generate", json={"prompt": "Hello"})

    assert response.status_code == 503
    assert "Checkpoint is required" in response.text
