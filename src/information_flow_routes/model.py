import typing

import nnsight
import torch
from beartype import beartype
from transformers import PreTrainedTokenizerBase


@beartype
def tokens_to_strings(
    tokenizer: PreTrainedTokenizerBase, tokens: torch.Tensor
) -> list[str]:
    # TODO: manage batch processing
    tokens = tokens[0]
    return [tokenizer.decode(token) for token in tokens.tolist()]


@beartype
def capture_inference_components(
    model: nnsight.LanguageModel, tokens: torch.Tensor
) -> list[dict[str, typing.Any]]:
    BATCH_SIZE = 1

    # TODO: manage batch processing
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    num_token = tokens.shape[1]

    cache = []
    with model.trace(tokens):
        for index, layer in enumerate(model.model.layers):
            next_layer_norm = (
                model.model.layers[index + 1].input_layernorm.input.save()
                if index < num_layers - 1
                else model.model.norm.input.save()
            )

            cache.append(
                {
                    "residuals": {
                        "pre_attention": layer.input_layernorm.input.save(),
                        "post_attention": layer.post_attention_layernorm.input.save(),
                        "post_feed_forward": next_layer_norm,
                    },
                    "attention": {
                        "projections": {
                            # TODO: manage batch processing
                            "value": layer.self_attn.v_proj.output.view(
                                BATCH_SIZE, num_token, num_key_value_heads, head_dim
                            )
                            .repeat_interleave(
                                num_attention_heads // num_key_value_heads, dim=-2
                            )
                            .save(),
                        },
                        "output_weights": layer.self_attn.o_proj.weight.view(
                            hidden_size, num_attention_heads, head_dim
                        )
                        .permute(1, 2, 0)
                        .save(),
                        "scores": layer.self_attn.output[1].squeeze(0).save(),
                    },
                    "feed_forward": {
                        "activations": layer.mlp.down_proj.input.save(),
                        "output": layer.mlp.down_proj.output.save(),
                    },
                }
            )

    return cache
