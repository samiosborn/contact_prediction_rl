# tests/test_full_reconstruction.py

"""Tests for the full reconstruction (FR) auxiliary loss."""

import pytest
import torch
import torch.nn.functional as F

from contact_prediction_rl.learning.models.auxiliary import FullReconstruction
from test_tactile_reconstruction import (
    ACTION_DIM,
    LATENT_DIM,
    OBSERVATION_DIM,
    PROPRIO_DIM,
    TACTILE_DIM,
    build,
    make_cfg,
    observe,
    run_iteration,
)


def make_objective(pos_weight: float = 10.0) -> FullReconstruction:
    return FullReconstruction(LATENT_DIM, OBSERVATION_DIM, TACTILE_DIM, hidden_dims=[16, 16], pos_weight=pos_weight)


def fr_cfg() -> dict:
    cfg = make_cfg()
    cfg["auxiliary"] = {**cfg["auxiliary"], "objective": "full_reconstruction"}
    return cfg


def test_the_decoder_rebuilds_the_whole_observation():
    objective = make_objective()

    assert objective.decoder(torch.randn(7, LATENT_DIM)).shape == (7, OBSERVATION_DIM)


def test_the_loss_is_the_tactile_cross_entropy_plus_the_proprioceptive_squared_error():
    objective = make_objective()

    # a decoder that outputs the same vector for every input, so the expected loss can be computed by hand
    last = objective.decoder[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.linspace(-1.0, 1.0, OBSERVATION_DIM))
    output = last.bias.detach()

    raw = torch.cat([torch.randn(10, PROPRIO_DIM) * 50.0, (torch.rand(10, TACTILE_DIM) > 0.5).float()], dim=1)
    normalised = torch.randn(10, OBSERVATION_DIM)  # a different tensor: only its proprioceptive part is used
    latent = torch.randn(10, LATENT_DIM)

    loss, metrics = objective.loss(raw, latent, normalised)

    expected_tactile = F.binary_cross_entropy_with_logits(
        output[PROPRIO_DIM:].expand(10, -1), raw[:, PROPRIO_DIM:], pos_weight=torch.full((TACTILE_DIM,), 10.0)
    )
    expected_proprio = F.mse_loss(output[:PROPRIO_DIM].expand(10, -1), normalised[:, :PROPRIO_DIM])

    assert metrics["aux_tactile_bce"] == pytest.approx(expected_tactile.item(), rel=1e-5)
    assert metrics["aux_proprio_mse"] == pytest.approx(expected_proprio.item(), rel=1e-5)
    assert loss.item() == pytest.approx((expected_tactile + expected_proprio).item(), rel=1e-5)


def test_the_tactile_target_is_raw_and_the_proprioceptive_target_is_normalised():
    objective = make_objective()
    latent = torch.randn(10, LATENT_DIM)
    raw = torch.cat([torch.randn(10, PROPRIO_DIM), (torch.rand(10, TACTILE_DIM) > 0.5).float()], dim=1)
    normalised = torch.randn(10, OBSERVATION_DIM)

    base = objective.loss(raw, latent, normalised)[1]

    # changing the raw proprioception or the normalised tactile part changes nothing
    changed_raw = raw.clone()
    changed_raw[:, :PROPRIO_DIM] += 100.0
    changed_normalised = normalised.clone()
    changed_normalised[:, PROPRIO_DIM:] += 100.0
    same = objective.loss(changed_raw, latent, changed_normalised)[1]
    assert same["aux_tactile_bce"] == pytest.approx(base["aux_tactile_bce"])
    assert same["aux_proprio_mse"] == pytest.approx(base["aux_proprio_mse"])

    # changing the normalised proprioception does change the squared error
    shifted = normalised.clone()
    shifted[:, :PROPRIO_DIM] += 3.0
    assert objective.loss(raw, latent, shifted)[1]["aux_proprio_mse"] != pytest.approx(base["aux_proprio_mse"])


def test_the_normalised_observations_are_required():
    with pytest.raises(ValueError, match="normalised"):
        make_objective().loss(torch.zeros(4, OBSERVATION_DIM), torch.zeros(4, LATENT_DIM))


def test_the_tactile_part_must_be_part_of_the_observation():
    for bad in (0, OBSERVATION_DIM):
        with pytest.raises(ValueError):
            FullReconstruction(LATENT_DIM, OBSERVATION_DIM, bad, hidden_dims=[8])


def test_ppo_trains_the_full_reconstruction_decoder_with_its_own_optimiser():
    torch.manual_seed(0)
    alg = build(fr_cfg())

    assert isinstance(alg.auxiliary, FullReconstruction)
    assert alg.auxiliary.decoder[-1].out_features == OBSERVATION_DIM

    ppo_ids = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    decoder_ids = {id(p) for p in alg.auxiliary.parameters()}
    clipped_ids = {id(p) for group in alg._clip_groups for p in group}
    assert not (decoder_ids & ppo_ids) and not (decoder_ids & clipped_ids)

    obs = observe()
    before = [p.detach().clone() for p in alg.auxiliary.parameters()]
    losses = []
    for _ in range(50):
        result, obs = run_iteration(alg, obs)
        losses.append(result["aux_loss"])

    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.auxiliary.parameters()))
    assert sum(losses[-5:]) / 5 < 0.9 * sum(losses[:5]) / 5
    assert {"aux_loss", "aux_tactile_bce", "aux_proprio_mse", "aux_f1"} <= set(result)


def test_the_fr_agent_configuration():
    from contact_prediction_rl.tasks.lift.config.franka_hx5d20.agents.ppo_cfg import (
        TACTILE_DIM as CONFIG_TACTILE_DIM,
        FrankaHX5D20BlockLiftTactileFRPaperPPOCfg,
    )

    cfg = FrankaHX5D20BlockLiftTactileFRPaperPPOCfg().to_dict()

    assert cfg["auxiliary"]["objective"] == "full_reconstruction"
    assert cfg["auxiliary"]["tactile_dim"] == CONFIG_TACTILE_DIM
    assert cfg["auxiliary"]["pos_weight"] == 10.0
    assert cfg["architecture"] == "shared"
    assert cfg["experiment_name"].endswith("_tactile_fr")
