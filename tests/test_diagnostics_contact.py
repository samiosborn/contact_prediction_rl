"""Pure Torch tests for prevalence, mask diversity and trivial contact predictors."""

import pytest
import torch

from contact_prediction_rl.diagnostics.contact import ContactTracker, binary_contact_metrics


def test_trivial_fixed_contact_pattern_has_perfect_classification_and_zero_flips():
    tracker = ContactTracker(3, 5)
    labels = torch.tensor([[1, 1, 1, 0, 0]] * 3, dtype=torch.float32)
    for _ in range(4):
        tracker.observe(labels)
    result = tracker.summary(fixed_pattern=(1, 1, 1, 0, 0))
    assert result["per_contact_positive_rate"] == [1, 1, 1, 0, 0]
    assert result["distinct_contact_masks"] == 1
    assert result["contact_transition_rate"] == 0
    assert result["fixed_constant_f1"] == pytest.approx(1)
    assert result["majority_constant_f1_in_sample"] == pytest.approx(1)
    scores = binary_contact_metrics((labels * 2 - 1) * 9, labels)
    assert scores["contact_f1"] == pytest.approx(1)


def test_contact_flips_do_not_cross_episode_boundaries():
    tracker = ContactTracker(2, 2)
    tracker.observe(torch.tensor([[1, 0], [1, 0]]), dones=torch.tensor([True, False]))
    tracker.observe(torch.tensor([[0, 1], [0, 1]]))
    result = tracker.summary(fixed_pattern=(1, 0))
    assert result["per_contact_transition_rate"] == pytest.approx([1., 1.])
    assert result["contact_transition_rate"] == pytest.approx(1)
    assert result["distinct_contact_masks"] == 2
    assert result["fixed_constant_f1"] == pytest.approx(.5)


def test_bad_contact_shapes_and_labels():
    tracker = ContactTracker(2, 5)
    with pytest.raises(ValueError):
        tracker.observe(torch.ones(2, 4))
    with pytest.raises(ValueError):
        tracker.observe(torch.ones(2, 5) * 0.5)
    with pytest.raises(ValueError):
        binary_contact_metrics(torch.zeros(2, 5), torch.ones(2, 4))
    with pytest.raises(ValueError):
        tracker.summary(fixed_pattern=(0, 1))
