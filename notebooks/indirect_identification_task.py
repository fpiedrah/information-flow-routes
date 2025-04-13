import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import nnsight

    from information_flow_routes.graph import (
        construct_information_flow_graph, find_prediction_paths)
    from information_flow_routes.model import tokens_to_strings
    from information_flow_routes.utilities import \
        find_token_substring_positions
    from information_flow_routes.visualization import Renderer

    return (
        Renderer,
        construct_information_flow_graph,
        find_prediction_paths,
        find_token_substring_positions,
        nnsight,
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
    instructed_prompt = "When Mary and John went to the store, John gave a drink to"

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


if __name__ == "__main__":
    app.run()
