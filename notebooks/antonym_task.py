import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import nnsight

    from information_flow_routes.graph import (
        Component,
        Graph,
        construct_information_flow_graph,
        find_prediction_paths,
        paths_from_token_nodes,
        average_edge_weights,
    )
    from information_flow_routes.model import tokens_to_strings
    from information_flow_routes.utilities import find_token_substring_positions
    from information_flow_routes.visualization import Renderer
    return (
        Component,
        Graph,
        Renderer,
        average_edge_weights,
        construct_information_flow_graph,
        find_prediction_paths,
        find_token_substring_positions,
        nnsight,
        paths_from_token_nodes,
        tokens_to_strings,
    )


@app.cell
def _():
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

    THRESHOLD = 0.03
    return MODEL_NAME, THRESHOLD


@app.cell
def _(MODEL_NAME, nnsight):
    model = nnsight.LanguageModel(MODEL_NAME)
    model.config.output_attentions = True
    return (model,)


@app.cell
def _(
    Renderer,
    THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    model,
    tokens_to_strings,
):
    few_shot_prompt = "big: small; happy: sad; true: false; daily: nightly; valid:"

    few_shot_tokens = model.tokenizer(
        few_shot_prompt,
        return_tensors="pt",
    )["input_ids"]
    few_shot_string_tokens = tokens_to_strings(model.tokenizer, few_shot_tokens)

    Renderer(
        model.config.num_hidden_layers,
        few_shot_string_tokens,
        len(few_shot_string_tokens) - 1,
    ).plot(
        find_prediction_paths(
            construct_information_flow_graph(model, few_shot_prompt, THRESHOLD),
            len(few_shot_string_tokens) - 1,
            THRESHOLD,
        )
    )
    return few_shot_prompt, few_shot_string_tokens, few_shot_tokens


@app.cell
def _(
    Renderer,
    THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    model,
    tokens_to_strings,
):
    instructed_prompt = "The contrary of massive is"

    instructed_tokens = model.tokenizer(
        instructed_prompt,
        return_tensors="pt",
    )["input_ids"]
    instructed_string_tokens = tokens_to_strings(
        model.tokenizer, instructed_tokens
    )

    instructed_renderer = Renderer(
        model.config.num_hidden_layers,
        instructed_string_tokens,
        len(instructed_string_tokens) - 1,
    )

    instructed_renderer.plot(
        find_prediction_paths(
            construct_information_flow_graph(model, instructed_prompt, THRESHOLD),
            len(instructed_string_tokens) - 1,
            THRESHOLD,
        )
    )
    return (
        instructed_prompt,
        instructed_renderer,
        instructed_string_tokens,
        instructed_tokens,
    )


@app.cell
def _(
    THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    find_token_substring_positions,
    instructed_prompt,
    instructed_renderer,
    instructed_string_tokens,
    model,
    paths_from_token_nodes,
):
    instructed_antonym_substring = ["contrary", "massive"]
    instructed_antonym_root_indices = find_token_substring_positions(
        instructed_prompt,
        instructed_antonym_substring,
        model.tokenizer,
        prepend_space=True,
    )

    instructed_renderer.plot(
        paths_from_token_nodes(
            find_prediction_paths(
                construct_information_flow_graph(
                    model, instructed_prompt, THRESHOLD
                ),
                len(instructed_string_tokens) - 1,
                THRESHOLD,
            ),
            instructed_antonym_root_indices,
        )
    )
    return instructed_antonym_root_indices, instructed_antonym_substring


@app.cell
def _(
    THRESHOLD,
    average_edge_weights,
    construct_information_flow_graph,
    find_prediction_paths,
    instructed_renderer,
    instructed_string_tokens,
    model,
):
    instructed_renderer.plot(
        find_prediction_paths(
            average_edge_weights(
                [
                    construct_information_flow_graph(model, prompt, THRESHOLD)
                    for prompt in [
                        f"The contrary of {label} is"
                        for label in [
                            "true",
                            "daily",
                            "distribution",
                            "valid",
                            "expand",
                            "other",
                            "square",
                            "pretty",
                            "clinical",
                        ]
                    ]
                ],
                num_layers=model.config.num_hidden_layers,
                num_tokens=len(instructed_string_tokens),
            ),
            len(instructed_string_tokens) - 1,
            THRESHOLD,
        )
    )
    return


@app.cell
def _(
    Renderer,
    THRESHOLD,
    average_edge_weights,
    construct_information_flow_graph,
    find_prediction_paths,
    instructed_prompt,
    model,
    tokens_to_strings,
):
    counterfactual_prompt = "The synonym of massive is"

    counterfactual_tokens = model.tokenizer(
        instructed_prompt,
        return_tensors="pt",
    )["input_ids"]
    counterfactual_string_tokens = tokens_to_strings(
        model.tokenizer, counterfactual_tokens
    )

    counterfactual_renderer = Renderer(
        model.config.num_hidden_layers,
        counterfactual_string_tokens,
        len(counterfactual_string_tokens) - 1,
    )

    counterfactual_renderer.plot(
        find_prediction_paths(
            average_edge_weights(
                [
                    construct_information_flow_graph(model, prompt, THRESHOLD)
                    for prompt in [
                        f"The synonym of {label} is"
                        for label in [
                            "true",
                            "daily",
                            "distribution",
                            "valid",
                            "expand",
                            "other",
                            "square",
                            "pretty",
                            "clinical",
                        ]
                    ]
                ],
                num_layers=model.config.num_hidden_layers,
                num_tokens=len(counterfactual_string_tokens),
            ),
            len(counterfactual_string_tokens) - 1,
            THRESHOLD,
        )
    )
    return (
        counterfactual_prompt,
        counterfactual_renderer,
        counterfactual_string_tokens,
        counterfactual_tokens,
    )


app._unparsable_cell(
    r"""
    factual_flow_graphs = [
        construct_information_flow_graph(model, prompt, THRESHOLD)
        for prompt in [
            f\"The contrary of {label} is\"
            for label in [
                \"true\",
                \"daily\",
                \"distribution\",
                \"valid\",
                \"expand\",
                \"other\",
                \"square\",
                \"pretty\",
                \"clinical\",
            ]
        ]
    ]

    counterfactual_flow_graphs = [
        construct_information_flow_graph(model, prompt, THRESHOLD)
        for prompt in [
            f\"The synonym of {label} is\"
            for label in [
                \"true\",
                \"daily\",
                \"distribution\",
                \"valid\",
                \"expand\",
                \"other\",
                \"square\",
                \"pretty\",
                \"clinical\",
            ]
        ]
    ]

    import networkx as nx


    def diff_digraph(graph1, graph2):
        new_graph = nx.DiGraph()

        for node in set(graph1.nodes):  # | set(graph2.nodes):
            new_graph.add_node(node)

        for u, v in set(graph1.edges):  # | set(graph2.nodes):
            # for v in set(graph1.nodes):  # | set(graph2.nodes):
            weight1 = graph1[u][v][\"weight\"]  # if graph1.has_edge(u, v) else 0
            weight2 = graph2[u][v][\"weight\"]  # if graph2.has_edge(u, v) else 0
            new_graph.add_edge(u, v, weight=weight1 - weight2)

        return new_graph


    counterfactual_renderer.plot(
        find_prediction_paths(
            average_edge_weights(
                [
                    diff_digraph(factual_flow_graph, counterfactual_flow_graph)
                    for factual_flow_graph, counterfactual_flow_graph in list(
                        zip(factual_flow_graphs, counterfactual_flow_graphs)
                    )[:1]
                ],
                num_layers=model.config.num_hidden_layers,
                num_tokens=len(counterfactual_string_tokens),s
            ),
            len(counterfactual_string_tokens) - 1,
            THRESHOLD,
        )
    )
    """,
    name="_"
)


@app.cell
def _(counterfactual_flow_graphs, factual_flow_graphs):
    set(factual_flow_graphs[1].edges) ^ set(counterfactual_flow_graphs[1].edges)
    return


if __name__ == "__main__":
    app.run()
