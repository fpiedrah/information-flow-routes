import enum
import itertools

import networkx
import nnsight
from beartype import beartype

from information_flow_routes.metrics import (
    compute_attention_contributions,
    compute_feed_forward_contributions,
    threshold_and_renormalize_contributions,
)
from information_flow_routes.model import (
    capture_inference_components,
    decompose_attention,
)


class Component(str, enum.Enum):
    TOKEN = "TOK"
    FEED_FORWARD = "FF"
    POST_ATTENTION_RESIDUAL = "RPA"
    POST_FEED_FORWARD_RESIDUAL = "RPFF"

    def __str__(self):
        return self.value


@beartype
class InformationFlowGraph(networkx.DiGraph):

    def __init__(self, num_layers: int | None = None, num_tokens: int | None = None):
        super().__init__(self)

        if num_layers and num_tokens:
            self._num_layers = num_layers
            self._num_tokens = num_tokens

            self._build()

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_tokens(self):
        return self._num_tokens

    def get_output_node(self, token_index: int) -> str:
        return self._feed_forward_residual_node_name(self.num_layers - 1, token_index)

    def _attention_residual_node_name(self, layer_index: int, token_index: int) -> str:
        return f"{Component.POST_ATTENTION_RESIDUAL}-{layer_index}-{token_index}"

    def _feed_forward_node_name(self, layer_index: int, token_index: int) -> str:
        return f"{Component.FEED_FORWARD}-{layer_index}-{token_index}"

    def _feed_forward_residual_node_name(
        self, layer_index: int, token_index: int
    ) -> str:
        return f"{Component.POST_FEED_FORWARD_RESIDUAL}-{layer_index}-{token_index}"

    def _token_node_name(self, token_index: int) -> str:
        return f"{Component.TOKEN}-{token_index}"

    def _build(self) -> None:
        for layer_index, token_index in itertools.product(
            range(self.num_layers), range(self.num_tokens)
        ):
            self.add_node(self._attention_residual_node_name(layer_index, token_index))
            self.add_node(self._feed_forward_node_name(layer_index, token_index))
            self.add_node(
                self._feed_forward_residual_node_name(layer_index, token_index)
            )

        for token_index in range(self.num_tokens):
            self.add_node(self._token_node_name(token_index))

    def _update_edge_weight(self, source: str, target: str, weight: float) -> None:
        if not self.has_edge(source, target):
            self.add_edge(source, target, weight=0.0)

        self[source][target]["weight"] += weight

    def _resolve_source_node_name(self, layer_index: int, token_index: int) -> str:
        match layer_index:
            case 0:
                return self._token_node_name(token_index)
            case _ if 1 <= layer_index <= self.num_layers:
                return self._feed_forward_residual_node_name(
                    layer_index - 1, token_index
                )
            case _:
                raise ValueError(
                    f"Invalid layer_index: {layer_index}. Expected 0 or a value "
                    f"between 1 and {self.num_layers}."
                )

    def update_attention_weight(
        self, layer_index: int, token_source: int, token_target: int, weight: float
    ) -> None:
        self._update_edge_weight(
            self._resolve_source_node_name(layer_index, token_source),
            self._attention_residual_node_name(layer_index, token_target),
            weight,
        )

    def update_residual_to_attention_weight(
        self, layer_index: int, token_index: int, weight: float
    ) -> None:
        self._update_edge_weight(
            self._resolve_source_node_name(layer_index, token_index),
            self._attention_residual_node_name(layer_index, token_index),
            weight,
        )

    def update_feed_forward_weight(
        self, layer_index: int, token_index: int, weight: float
    ) -> None:
        self._update_edge_weight(
            self._attention_residual_node_name(layer_index, token_index),
            self._feed_forward_node_name(layer_index, token_index),
            weight,
        )

        self._update_edge_weight(
            self._feed_forward_node_name(layer_index, token_index),
            self._feed_forward_residual_node_name(layer_index, token_index),
            weight,
        )

    def update_residual_to_feed_forward_weight(
        self, layer_index: int, token_index: int, weight: float
    ) -> None:
        self._update_edge_weight(
            self._attention_residual_node_name(layer_index, token_index),
            self._feed_forward_residual_node_name(layer_index, token_index),
            weight,
        )


def construct_information_flow_graph(
    model: nnsight.LanguageModel,
    prompt: str,
    normalization_threshold: float | None = None,
) -> InformationFlowGraph:
    BATCH_INDEX = 0

    num_layers = len(model.model.layers)

    tokens = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
    num_tokens = tokens.shape[1]

    information_flow_graph = InformationFlowGraph(num_layers, num_tokens)
    inference_components_cache = capture_inference_components(model, tokens)

    for layer_index in range(num_layers):
        cache = inference_components_cache[layer_index]

        attention_contributions, attention_residual_contributions = (
            compute_attention_contributions(
                cache["residuals"]["pre_attention"][BATCH_INDEX].unsqueeze(0),
                cache["residuals"]["post_attention"][BATCH_INDEX].unsqueeze(0),
                decompose_attention(cache).unsqueeze(0),
            )
        )

        if normalization_threshold:
            attention_contributions, attention_residual_contributions = (
                threshold_and_renormalize_contributions(
                    normalization_threshold,
                    attention_contributions,
                    attention_residual_contributions,
                )
            )

        for source_token in range(num_tokens):
            for target_token in range(num_tokens):
                contribution = (
                    attention_contributions[BATCH_INDEX, source_token, target_token]
                    .sum()
                    .item()
                )
                information_flow_graph.update_attention_weight(
                    layer_index, source_token, target_token, contribution
                )

        for token_index in range(num_tokens):
            information_flow_graph.update_residual_to_attention_weight(
                layer_index,
                token_index,
                attention_residual_contributions[BATCH_INDEX, token_index].item(),
            )

        feed_forward_contributions, feed_forward_residual_contributions = (
            compute_feed_forward_contributions(
                cache["residuals"]["post_attention"][BATCH_INDEX].unsqueeze(0),
                cache["residuals"]["post_feed_forward"][BATCH_INDEX].unsqueeze(0),
                cache["feed_forward"]["output"][BATCH_INDEX].unsqueeze(0),
            )
        )

        if normalization_threshold:
            feed_forward_contributions, feed_forward_residual_contributions = (
                threshold_and_renormalize_contributions(
                    normalization_threshold,
                    feed_forward_contributions,
                    feed_forward_residual_contributions,
                )
            )

        for token_index in range(num_tokens):
            information_flow_graph.update_feed_forward_weight(
                layer_index,
                token_index,
                feed_forward_contributions[0, token_index].item(),
            )
            information_flow_graph.update_residual_to_feed_forward_weight(
                layer_index,
                token_index,
                feed_forward_residual_contributions[0, token_index].item(),
            )

    return information_flow_graph


def find_prediction_paths(
    graph: InformationFlowGraph, root_token_index: int, threshold: float
) -> list[InformationFlowGraph]:
    num_layers = graph.num_layers
    num_tokens = graph.num_tokens

    new_graph = InformationFlowGraph(num_layers, num_tokens)

    reversed_graph = graph.reverse()
    graph_search = networkx.subgraph_view(
        reversed_graph,
        filter_edge=lambda source, target: reversed_graph[source][target]["weight"]
        > threshold,
    )

    if root_token_index > num_tokens:
        raise ValueError()

    if root_token_index < 0:
        raise ValueError()

    return [
        graph_search.edge_subgraph(
            networkx.edge_dfs(
                graph_search, source=new_graph.get_output_node(root_token_index)
            )
        )
    ]
