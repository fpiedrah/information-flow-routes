import einops
import torch
from beartype import beartype


@torch.inference_mode()
@beartype
def compute_contributions(
    component_output: torch.Tensor,
    post_component_residual: torch.Tensor,
    distance_norm: int = 1,
) -> torch.Tensor:
    EPSILON = 1e-5

    num_part_dims = len(component_output.shape) - len(post_component_residual.shape)

    if num_part_dims < 0:
        raise ValueError(
            "The 'component_output' tensor must have at least as many dimensions as the "
            "'post_component_residual' tensor. Got: "
            f"{len(component_output.shape)} vs {len(post_component_residual.shape)}."
        )

    if component_output.shape[num_part_dims:] != post_component_residual.shape:
        raise ValueError(
            "The trailing dimensions of 'component_output' must match the shape of "
            "'post_component_residual'. Got: "
            f"{component_output.shape[num_part_dims:]} vs {post_component_residual.shape}."
        )

    post_component_residual = post_component_residual.expand(component_output.shape)

    distance = torch.nn.functional.pairwise_distance(
        component_output, post_component_residual, p=distance_norm
    )
    residual_norm = torch.norm(post_component_residual, p=distance_norm, dim=-1)
    distance = (residual_norm - distance).clip(min=EPSILON)

    return distance / distance.sum(dim=tuple(range(num_part_dims)), keepdim=True)


@torch.inference_mode()
@beartype
def compute_contributions_with_residual(
    component_output: torch.Tensor,
    pre_component_residual: torch.Tensor,
    post_component_residual: torch.Tensor,
    distance_norm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pre_component_residual.shape != post_component_residual.shape:
        raise ValueError(
            "The shapes of 'pre_component_residual' and 'post_component_residual' must be identical. "
            f"Got: {pre_component_residual.shape} vs {post_component_residual.shape}."
        )

    num_part_dims = len(component_output.shape) - len(post_component_residual.shape)
    if num_part_dims < 0:
        raise ValueError(
            "The 'component_output' tensor must have at least as many dimensions as the "
            "'post_component_residual' tensor. Got: "
            f"{len(component_output.shape)} vs {len(post_component_residual.shape)}."
        )

    flatten_component_output = component_output.flatten(
        start_dim=0, end_dim=num_part_dims - 1
    )
    flatten_component_output = torch.cat(
        [flatten_component_output, pre_component_residual.unsqueeze(0)]
    )

    contributions = compute_contributions(
        flatten_component_output, post_component_residual, distance_norm
    )
    component_contributions, pre_component_residual_contributions = torch.split(
        contributions, flatten_component_output.shape[0] - 1
    )

    return (
        component_contributions.unflatten(0, component_output.shape[0:num_part_dims]),
        pre_component_residual_contributions[0],
    )


@torch.inference_mode()
@beartype
def compute_attention_contributions(
    pre_attention_residual: torch.Tensor,
    post_attention_residual: torch.Tensor,
    decomposed_attention: torch.Tensor,
    distance_norm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    decomposed_attention = einops.rearrange(
        decomposed_attention,
        "batch token_pos key_pos attn_head hidden_dim"
        " -> key_pos attn_head batch token_pos hidden_dim",
    )

    attention_contributions, residual_contributions = (
        compute_contributions_with_residual(
            decomposed_attention,
            pre_attention_residual,
            post_attention_residual,
            distance_norm,
        )
    )

    attention_contributions = einops.rearrange(
        attention_contributions,
        "key_pos attn_head batch token_pos -> batch token_pos key_pos attn_head",
    )

    return attention_contributions, residual_contributions


@torch.inference_mode()
@beartype
def compute_feed_forward_contributions(
    post_attention_residual: torch.Tensor,
    post_feed_forward_residual: torch.Tensor,
    feed_forward_output: torch.Tensor,
    distance_norm: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    contributions = compute_contributions(
        torch.stack((feed_forward_output, post_attention_residual)),
        post_feed_forward_residual,
        distance_norm,
    )

    return contributions[0], contributions[1]


@torch.inference_mode()
@beartype
def compute_decomposed_feed_forward_contributions(
    post_attention_residual: torch.Tensor,
    post_feed_forward_residual: torch.Tensor,
    decomposed_feed_forward: torch.Tensor,
    distance_norm: int = 1,
) -> tuple[torch.Tensor, float]:
    neuron_contributions, residual_contributions = compute_contributions_with_residual(
        decomposed_feed_forward,
        post_attention_residual,
        post_feed_forward_residual,
        distance_norm,
    )

    return neuron_contributions, residual_contributions.item()


@torch.inference_mode()
@beartype
def threshold_and_renormalize_contributions(
    threshold: float,
    component_contributions: torch.Tensor,
    residual_contributions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    component_dim = len(component_contributions.shape)
    residual_dim = len(residual_contributions.shape)

    bound_dim = component_dim - residual_dim
    if bound_dim < 0:
        raise ValueError(
            "The 'component_contributions' tensor must have at least as many dimensions as the "
            "'residual_contributions' tensor. "
            f"Got: {component_dim} vs {residual_dim}."
        )

    if component_contributions.shape[0:residual_dim] != residual_contributions.shape:
        raise ValueError(
            "The trailing dimensions of 'component_contributions' must match the shape of 'residual_contributions'. "
            f"Got: {component_contributions.shape[component_dim - residual_dim :]} vs {residual_contributions.shape}."
        )

    component_contributions = component_contributions * (
        component_contributions > threshold
    )
    residual_contributions = residual_contributions * (
        residual_contributions > threshold
    )

    denominator = residual_contributions + component_contributions.sum(
        dim=tuple(range(residual_dim, component_dim))
    )

    return (
        component_contributions
        / denominator.reshape(denominator.shape + (1,) * bound_dim),
        residual_contributions / denominator,
    )
