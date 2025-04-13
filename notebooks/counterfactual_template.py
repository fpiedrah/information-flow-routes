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
        subgraph_from_counterfactual,
        subgraph_from_token_nodes,
    )
    from information_flow_routes.model import tokens_to_strings
    from information_flow_routes.utilities import find_token_substring_positions
    from information_flow_routes.visualization import Renderer
    return (
        Component,
        Graph,
        Renderer,
        construct_information_flow_graph,
        find_prediction_paths,
        find_token_substring_positions,
        nnsight,
        subgraph_from_counterfactual,
        subgraph_from_token_nodes,
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
    factual_prompt = (
        "protect: harm; ethical: unethical; maximum: minimum; few: many; danger:"
    )

    factual_tokens = model.tokenizer(
        factual_prompt,
        return_tensors="pt",
    )["input_ids"]
    factual_string_tokens = tokens_to_strings(model.tokenizer, factual_tokens)

    factual_graph = find_prediction_paths(
        construct_information_flow_graph(model, factual_prompt, THRESHOLD),
        len(factual_string_tokens) - 1,
        THRESHOLD,
    )

    Renderer(
        model.config.num_hidden_layers,
        factual_string_tokens,
        len(factual_string_tokens) - 1,
    ).plot(factual_graph)
    return factual_graph, factual_prompt, factual_string_tokens, factual_tokens


@app.cell
def _(
    Renderer,
    THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    model,
    tokens_to_strings,
):
    counterfactual_prompt = (
        "protect: original; ethical: major; maximum: affected; few: joy; danger:"
    )

    counterfactual_tokens = model.tokenizer(
        counterfactual_prompt,
        return_tensors="pt",
    )["input_ids"]
    counterfactual_string_tokens = tokens_to_strings(
        model.tokenizer, counterfactual_tokens
    )

    counterfactual_graph = find_prediction_paths(
        construct_information_flow_graph(model, counterfactual_prompt, THRESHOLD),
        len(counterfactual_string_tokens) - 1,
        THRESHOLD,
    )

    Renderer(
        model.config.num_hidden_layers,
        counterfactual_string_tokens,
        len(counterfactual_string_tokens) - 1,
    ).plot(counterfactual_graph)
    return (
        counterfactual_graph,
        counterfactual_prompt,
        counterfactual_string_tokens,
        counterfactual_tokens,
    )


@app.cell
def _(
    Renderer,
    counterfactual_graph,
    factual_graph,
    factual_string_tokens,
    model,
    subgraph_from_counterfactual,
):
    Renderer(
        model.config.num_hidden_layers,
        factual_string_tokens,
        len(factual_string_tokens) - 1,
    ).plot(subgraph_from_counterfactual(factual_graph, counterfactual_graph))
    return


if __name__ == "__main__":
    app.run()
