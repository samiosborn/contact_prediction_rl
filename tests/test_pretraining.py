# tests/test_pretraining.py

"""Tests for the fine-tuning benchmark, using the toy policy on the CPU (no pretrained weights needed)."""

import importlib.util
import json

import pytest
import torch

from contact_prediction_rl.pretraining.benchmark import BenchmarkConfig, run_benchmark, summarise
from contact_prediction_rl.pretraining.data import BatchSpec, SyntheticDemoDataset, make_loader
from contact_prediction_rl.pretraining.lora import apply_lora, count_parameters
from contact_prediction_rl.pretraining.models import LORA_TARGET_MODULES, ToyPolicy

pytestmark = pytest.mark.unit

SPEC = BatchSpec(image_keys=("observation.images.wrist",), image_shape=(3, 64, 64), chunk_size=4, language_length=6)


def make_toy() -> ToyPolicy:
    return ToyPolicy(SPEC.image_keys, SPEC.state_dim, SPEC.tactile_dim, SPEC.action_dim, SPEC.chunk_size, SPEC.vocab_size, width=32, depth=2, heads=4)


def test_synthetic_samples_have_the_spec_shapes():
    sample = SyntheticDemoDataset(SPEC, num_samples=4)[0]

    assert sample["observation.images.wrist"].shape == SPEC.image_shape
    assert sample["observation.state"].shape == (SPEC.state_dim,)
    assert sample["observation.tactile"].shape == (SPEC.tactile_dim,)
    assert sample["action"].shape == (SPEC.chunk_size, SPEC.action_dim)
    assert sample["observation.language.tokens"].max() < SPEC.vocab_size

    # samples are reproducible
    assert torch.equal(sample["action"], SyntheticDemoDataset(SPEC, num_samples=4)[0]["action"])


def test_the_toy_policy_trains_on_synthetic_batches():
    policy = make_toy()
    batch = next(iter(make_loader(SyntheticDemoDataset(SPEC, num_samples=8), batch_size=4, num_workers=0)))

    loss, info = policy(batch)
    loss.backward()

    assert loss.ndim == 0 and info["loss"] == pytest.approx(loss.item())
    assert policy.image_encoder[0].weight.grad is not None
    assert policy.action_head.weight.grad is not None


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="peft is not installed")
def test_lora_trains_only_the_adapters():
    policy = apply_lora(make_toy(), LORA_TARGET_MODULES["toy"], rank=4)
    trainable, total = count_parameters(policy)

    assert 0 < trainable < total
    assert all("lora_" in name for name, p in policy.named_parameters() if p.requires_grad)


def test_benchmark_writes_a_result_and_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    output = tmp_path / "benchmark.jsonl"
    cfg = BenchmarkConfig(strategy="none", batch_size=2, steps=2, warmup_steps=1, num_workers=0, bf16=False, num_samples=8, output=str(output))

    result = run_benchmark(cfg)

    assert result["world_size"] == 1 and result["samples_per_s"] > 0
    assert 0 <= result["data_time_fraction"] <= 1
    assert json.loads(output.read_text())["samples_per_s"] == pytest.approx(result["samples_per_s"])
    assert summarise(str(output))[0]["scaling_efficiency"] == pytest.approx(1.0)
