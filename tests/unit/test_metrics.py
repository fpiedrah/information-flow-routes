import pytest
import torch

from information_flow_routes.metrics import (
    compute_attention_contributions,
    compute_contributions,
    compute_contributions_with_residual,
    compute_decomposed_feed_forward_contributions,
    compute_feed_forward_contributions,
    threshold_and_renormalize_contributions,
)

EPSILON = 1e-5


def test_compute_contributions():
    component_output = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    post_component_residual = torch.tensor([1.0, 1.0])

    expected = torch.tensor([2.0 / (2.0 + EPSILON), EPSILON / (2.0 + EPSILON)])
    out = compute_contributions(
        component_output, post_component_residual, distance_norm=1
    )

    assert torch.allclose(out, expected, atol=EPSILON)


def test_compute_contributions_with_residual():
    component_output = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    pre_component_residual = torch.tensor([1.0, 1.0])
    post_component_residual = torch.tensor([1.0, 1.0])

    total = 4.0 + EPSILON
    exp_comp = torch.tensor([2.0 / total, EPSILON / total])
    exp_pre = 2.0 / total

    comp_contrib, pre_residual_contrib = compute_contributions_with_residual(
        component_output,
        pre_component_residual,
        post_component_residual,
        distance_norm=1,
    )

    assert torch.allclose(comp_contrib.flatten(), exp_comp, atol=EPSILON)
    assert torch.allclose(pre_residual_contrib, torch.tensor(exp_pre), atol=EPSILON)


def test_compute_attention_contributions():
    batches = 4
    num_tokens = 5
    num_attn_heads = 6
    hidden_dim = 10
    key_dim = num_tokens

    pre_attention_residual = torch.ones((batches, num_tokens, hidden_dim))
    post_attention_residual = torch.ones((batches, num_tokens, hidden_dim))
    decomposed_attention = torch.ones(
        (batches, num_tokens, key_dim, num_attn_heads, hidden_dim)
    )

    attn_contrib, resid_contrib = compute_attention_contributions(
        pre_attention_residual,
        post_attention_residual,
        decomposed_attention,
        distance_norm=1,
    )

    assert attn_contrib.shape == (batches, num_tokens, key_dim, num_attn_heads)
    assert resid_contrib.shape == (batches, num_tokens)

    expected_value = 1.0 / (key_dim * num_attn_heads + 1)
    sum_components = attn_contrib.sum(dim=(-2, -1))
    total = sum_components + resid_contrib

    assert torch.allclose(total, torch.full_like(total, 1.0), atol=EPSILON)
    assert torch.allclose(
        attn_contrib, torch.full_like(attn_contrib, expected_value), atol=EPSILON
    )


def test_compute_feed_forward_contributions():
    post_attention_residual = torch.tensor([[0.0, 0.0]])
    post_feed_forward_residual = torch.tensor([[1.0, 1.0]])
    feed_forward_output = torch.tensor([[1.0, 1.0]])

    expected_ffn = 2.0 / (2.0 + EPSILON)
    expected_res = EPSILON / (2.0 + EPSILON)

    ffn_contrib, resid_contrib = compute_feed_forward_contributions(
        post_attention_residual,
        post_feed_forward_residual,
        feed_forward_output,
        distance_norm=1,
    )

    expected_ffn_tensor = torch.tensor([[expected_ffn, expected_ffn]])
    expected_res_tensor = torch.tensor([[expected_res, expected_res]])

    assert torch.allclose(ffn_contrib, expected_ffn_tensor, atol=EPSILON)
    assert torch.allclose(resid_contrib, expected_res_tensor, atol=EPSILON)


def test_compute_decomposed_feed_forward_contributions():

    post_attention_residual = torch.tensor([1.0, 1.0])
    post_feed_forward_residual = torch.tensor([4.0, 4.0])

    decomposed_feed_forward = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    neuron_contrib, resid_contrib = compute_decomposed_feed_forward_contributions(
        post_attention_residual,
        post_feed_forward_residual,
        decomposed_feed_forward,
        distance_norm=1,
    )

    assert torch.allclose(neuron_contrib, torch.tensor([0.25, 0.50]), atol=EPSILON)
    assert resid_contrib == pytest.approx(0.25, abs=EPSILON)


def test_threshold_and_renormalize_contributions():
    component_contrib = torch.tensor([[0.05, 0.15], [0.05, 0.05]])
    residual_contrib = torch.tensor([0.8, 0.9])
    norm_comp, norm_res = threshold_and_renormalize_contributions(
        0.1, component_contrib, residual_contrib
    )

    expected_norm_comp = torch.tensor([[0.0, 0.157894], [0.0, 0.0]])
    expected_norm_res = torch.tensor([0.842105, 1.0])

    assert torch.allclose(norm_comp, expected_norm_comp, atol=EPSILON)
    assert torch.allclose(norm_res, expected_norm_res, atol=EPSILON)
