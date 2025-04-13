import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import nnsight

    from information_flow_routes.graph import (
        Component, Graph, construct_information_flow_graph,
        find_prediction_paths, subgraph_from_token_nodes)
    from information_flow_routes.model import tokens_to_strings
    from information_flow_routes.utilities import \
        find_token_substring_positions
    from information_flow_routes.visualization import Renderer

    return (
        Component,
        Graph,
        Renderer,
        construct_information_flow_graph,
        find_prediction_paths,
        find_token_substring_positions,
        nnsight,
        subgraph_from_token_nodes,
        tokens_to_strings,
    )


@app.cell
def _():
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

    ZERO_SHOT_THRESHOLD = 0.07
    FEW_SHOT_THRESHOLD = 0.03
    return FEW_SHOT_THRESHOLD, MODEL_NAME, ZERO_SHOT_THRESHOLD


@app.cell
def _(MODEL_NAME, nnsight):
    model = nnsight.LanguageModel(MODEL_NAME)
    model.config.output_attentions = True
    return (model,)


@app.cell
def _(
    FEW_SHOT_THRESHOLD,
    Renderer,
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
            construct_information_flow_graph(
                model, few_shot_prompt, FEW_SHOT_THRESHOLD
            ),
            len(few_shot_string_tokens) - 1,
            FEW_SHOT_THRESHOLD,
        )
    )
    return few_shot_prompt, few_shot_string_tokens, few_shot_tokens


@app.cell
def _(
    Renderer,
    ZERO_SHOT_THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    model,
    tokens_to_strings,
):
    instructed_prompt = "The antonym of massive is"

    instructed_tokens = model.tokenizer(
        instructed_prompt,
        return_tensors="pt",
    )["input_ids"]
    instructed_string_tokens = tokens_to_strings(model.tokenizer, instructed_tokens)

    instructed_renderer = Renderer(
        model.config.num_hidden_layers,
        instructed_string_tokens,
        len(instructed_string_tokens) - 1,
    )

    instructed_renderer.plot(
        find_prediction_paths(
            construct_information_flow_graph(
                model, instructed_prompt, ZERO_SHOT_THRESHOLD
            ),
            len(instructed_string_tokens) - 1,
            ZERO_SHOT_THRESHOLD,
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
    ZERO_SHOT_THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    find_token_substring_positions,
    instructed_prompt,
    instructed_renderer,
    instructed_string_tokens,
    model,
    subgraph_from_token_nodes,
):
    instructed_antonym_substring = ["antonym"]
    instructed_antonym_root_indices = find_token_substring_positions(
        instructed_prompt,
        instructed_antonym_substring,
        model.tokenizer,
        prepend_space=True,
    )

    instructed_renderer.plot(
        subgraph_from_token_nodes(
            find_prediction_paths(
                construct_information_flow_graph(
                    model, instructed_prompt, ZERO_SHOT_THRESHOLD
                ),
                len(instructed_string_tokens) - 1,
                ZERO_SHOT_THRESHOLD,
            ),
            instructed_antonym_root_indices,
            ZERO_SHOT_THRESHOLD,
        )
    )
    return instructed_antonym_root_indices, instructed_antonym_substring


@app.cell
def _(
    ZERO_SHOT_THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    find_token_substring_positions,
    instructed_prompt,
    instructed_renderer,
    instructed_string_tokens,
    model,
    subgraph_from_token_nodes,
):
    instructed_massive_substring = ["massive"]
    instructed_massive_root_indices = find_token_substring_positions(
        instructed_prompt,
        instructed_massive_substring,
        model.tokenizer,
        prepend_space=True,
    )

    instructed_renderer.plot(
        subgraph_from_token_nodes(
            find_prediction_paths(
                construct_information_flow_graph(
                    model, instructed_prompt, ZERO_SHOT_THRESHOLD
                ),
                len(instructed_string_tokens) - 1,
                ZERO_SHOT_THRESHOLD,
            ),
            instructed_massive_root_indices,
            ZERO_SHOT_THRESHOLD,
        )
    )
    return instructed_massive_root_indices, instructed_massive_substring


@app.cell
def _(
    ZERO_SHOT_THRESHOLD,
    construct_information_flow_graph,
    find_prediction_paths,
    find_token_substring_positions,
    instructed_prompt,
    instructed_renderer,
    instructed_string_tokens,
    model,
    subgraph_from_token_nodes,
):
    instructed_combined_substring = ["antonym", "massive"]
    instructed_combined_root_indices = find_token_substring_positions(
        instructed_prompt,
        instructed_combined_substring,
        model.tokenizer,
        prepend_space=True,
    )

    instructed_renderer.plot(
        subgraph_from_token_nodes(
            find_prediction_paths(
                construct_information_flow_graph(
                    model, instructed_prompt, ZERO_SHOT_THRESHOLD
                ),
                len(instructed_string_tokens) - 1,
                ZERO_SHOT_THRESHOLD,
            ),
            instructed_combined_root_indices,
            ZERO_SHOT_THRESHOLD,
        )
    )
    return instructed_combined_root_indices, instructed_combined_substring


if __name__ == "__main__":
    app.run()
