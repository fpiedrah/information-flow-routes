from types import SimpleNamespace

import torch

from information_flow_routes.model import decompose_attention


def test_decompose_attention_shape():
    num_tokens = 5
    num_attn_heads = 6
    hidden_dim = 10
    key_dim = num_tokens
    head_dim = 8

    attention_value = torch.rand(key_dim, num_attn_heads, head_dim)
    attention_scores = torch.rand(num_attn_heads, num_tokens, key_dim)
    attention_output_weights = torch.rand(num_attn_heads, head_dim, hidden_dim)

    cache = {
        "attention": {
            "projections": {"value": SimpleNamespace(value=attention_value)},
            "scores": SimpleNamespace(value=attention_scores),
            "output_weights": SimpleNamespace(value=attention_output_weights),
        }
    }

    result = decompose_attention(cache)
    expected_shape = (num_tokens, num_tokens, num_attn_heads, hidden_dim)
    assert result.shape == expected_shape
