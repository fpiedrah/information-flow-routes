import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import collections
    import copy
    import json
    import os
    import random

    import marimo
    import nnsight

    from information_flow_routes.graph import (
        average_edge_weights,
        construct_information_flow_graph,
        find_prediction_paths,
        compute_weight_difference,
        extract_causal_components,
    )
    from information_flow_routes.model import tokens_to_strings
    from information_flow_routes.visualization import Renderer
    return (
        Renderer,
        average_edge_weights,
        collections,
        compute_weight_difference,
        construct_information_flow_graph,
        copy,
        extract_causal_components,
        find_prediction_paths,
        json,
        marimo,
        nnsight,
        os,
        random,
        tokens_to_strings,
    )


@app.cell
def _(marimo, os):
    # ARGUMENTS
    CLI_ARGUMENTS = marimo.cli_args()

    # GENERAL
    TASK_IDENTIFIER = CLI_ARGUMENTS.get("TASK_IDENTIFIER", "003-SENTENCE_CASE")

    # MODEL CONFIGURATION
    MODEL_NAME = CLI_ARGUMENTS.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")

    # DATA CONFIGURATION
    DATASET_BASE_PATH = CLI_ARGUMENTS.get("DATASET_BASE_PATH", "./datasets/")
    DATASET_FILE = CLI_ARGUMENTS.get("DATASET_FILE", "003-sentence_case.json")
    DATASET_PATH = os.path.join(DATASET_BASE_PATH, DATASET_FILE)

    EXPORT_PDF = CLI_ARGUMENTS.get("EXPORT_PDF", False)
    EXPORT_BASE_PATH = CLI_ARGUMENTS.get("EXPORT_BASE_PATH", "./assets")
    EXPORT_PATH = os.path.join(EXPORT_BASE_PATH, TASK_IDENTIFIER)

    if EXPORT_PDF:
        os.makedirs(EXPORT_PATH, exist_ok=True)

    # PROMPT CONFIGURATION
    INSTRUCTIONS = CLI_ARGUMENTS.get(
        "INSTRUCTIONS",
        "Return the uppercase letter of the first letter of the input string.",
    )
    QUERY_TEMPLATE = CLI_ARGUMENTS.get("QUERY_TEMPLATE", "\nQ: {query}\nA:")
    ZERO_SHOT_TEMPLATE = f"{INSTRUCTIONS}{QUERY_TEMPLATE}"

    # INFERENCE PARAMETERS
    THRESHOLD = CLI_ARGUMENTS.get("THRESHOLD", 0.03)
    NUM_EXAMPLES = CLI_ARGUMENTS.get("NUM_EXAMPLES", 3)
    MAX_NUM_PROMPTS = CLI_ARGUMENTS.get("MAX_NUM_PROMPTS", 20)
    return (
        CLI_ARGUMENTS,
        DATASET_BASE_PATH,
        DATASET_FILE,
        DATASET_PATH,
        EXPORT_BASE_PATH,
        EXPORT_PATH,
        EXPORT_PDF,
        INSTRUCTIONS,
        MAX_NUM_PROMPTS,
        MODEL_NAME,
        NUM_EXAMPLES,
        QUERY_TEMPLATE,
        TASK_IDENTIFIER,
        THRESHOLD,
        ZERO_SHOT_TEMPLATE,
    )


@app.cell
def _(MODEL_NAME, nnsight):
    model = nnsight.LanguageModel(MODEL_NAME)
    num_layers = model.config.num_hidden_layers

    model.config.output_attentions = True
    model.config.attn_implementation = "eager"
    return model, num_layers


@app.cell
def _(DATASET_PATH, json, model):
    dataset = json.load(open(DATASET_PATH))


    def is_single_token_pair(instance, tokenizer):
        input = instance.get("input", "")
        output = instance.get("output", "")

        input_tokens = tokenizer.encode(input, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)

        return len(input_tokens) == 1 and len(output_tokens) == 1


    dataset = list(
        filter(
            lambda instance: is_single_token_pair(instance, model.tokenizer),
            dataset,
        )
    )
    return dataset, is_single_token_pair


@app.cell
def _(
    INSTRUCTIONS,
    MAX_NUM_PROMPTS,
    ZERO_SHOT_TEMPLATE,
    collections,
    dataset,
    model,
    tokens_to_strings,
):
    def create_zero_shot_prompts(instructions, dataset):
        return [
            ZERO_SHOT_TEMPLATE.format(
                instructions=instructions,
                query=example["input"],
            )
            for example in dataset
        ]


    zero_shot_prompts = create_zero_shot_prompts(INSTRUCTIONS, dataset)

    zero_shot_tokens = [
        model.tokenizer(prompt, return_tensors="pt")["input_ids"]
        for prompt in zero_shot_prompts
    ]

    zero_shot_string_tokens = [
        tokens_to_strings(model.tokenizer, tokens) for tokens in zero_shot_tokens
    ]

    most_common_length = collections.Counter(
        [len(tokens) for tokens in zero_shot_string_tokens]
    ).most_common(1)[0][0]

    zero_shot_string_tokens, zero_shot_prompts = map(
        list,
        zip(
            *filter(
                lambda token_prompt_pair: len(token_prompt_pair[0])
                == most_common_length,
                zip(zero_shot_string_tokens, zero_shot_prompts),
            )
        ),
    )

    zero_shot_string_tokens, zero_shot_prompts = (
        zero_shot_string_tokens[:MAX_NUM_PROMPTS],
        zero_shot_prompts[:MAX_NUM_PROMPTS],
    )
    return (
        create_zero_shot_prompts,
        most_common_length,
        zero_shot_prompts,
        zero_shot_string_tokens,
        zero_shot_tokens,
    )


@app.cell
def _(
    EXPORT_PATH,
    EXPORT_PDF,
    Renderer,
    TASK_IDENTIFIER,
    THRESHOLD,
    average_edge_weights,
    construct_information_flow_graph,
    find_prediction_paths,
    marimo,
    model,
    num_layers,
    os,
    zero_shot_prompts,
    zero_shot_string_tokens,
):
    num_zero_shot_tokens = len(zero_shot_string_tokens[0])

    zero_shot_graphs = [
        construct_information_flow_graph(model, zero_shot_prompt, THRESHOLD)
        for zero_shot_prompt in marimo.status.progress_bar(zero_shot_prompts)
    ]

    zero_shot_graph = average_edge_weights(
        zero_shot_graphs, num_layers, num_zero_shot_tokens
    )

    Renderer(
        model.config.num_hidden_layers,
        zero_shot_string_tokens[0],
        len(zero_shot_string_tokens[0]) - 1,
    ).plot(
        find_prediction_paths(
            zero_shot_graph,
            len(zero_shot_string_tokens[0]) - 1,
            THRESHOLD,
        ),
        export_pdf=EXPORT_PDF,
        filename=os.path.join(EXPORT_PATH, TASK_IDENTIFIER, "zero_shot_graph.pdf"),
    )
    return num_zero_shot_tokens, zero_shot_graph, zero_shot_graphs


@app.cell
def _(
    MAX_NUM_PROMPTS,
    NUM_EXAMPLES,
    copy,
    dataset,
    model,
    random,
    tokens_to_strings,
):
    num_few_shot_tokens = (NUM_EXAMPLES * 4) + 3


    def shuffle_dataset_outputs(dataset):
        shuffled_dataset = copy.deepcopy(dataset)

        all_outputs = [example["output"] for example in shuffled_dataset]

        shuffled_outputs = all_outputs.copy()
        random.shuffle(shuffled_outputs)

        for index, example in enumerate(shuffled_dataset):
            example["output"] = shuffled_outputs[index]

        return shuffled_dataset


    def create_few_shot_prompts(dataset, num_examples):
        few_shot_prompts = []

        for index, query in enumerate(dataset):
            examples = dataset[max(0, index - num_examples) : index]

            if len(examples) < num_examples:
                examples = dataset[:index] + dataset[: num_examples - index]

            entries = [
                f"{example['input']}: {example['output']}" for example in examples
            ]
            entries += [f"{query['input']}:"]

            few_shot_prompts.append("; ".join(entries))

        return few_shot_prompts


    few_shot_prompts = create_few_shot_prompts(dataset, NUM_EXAMPLES)

    few_shot_tokens = [
        model.tokenizer(prompt, return_tensors="pt")["input_ids"]
        for prompt in few_shot_prompts
    ]

    few_shot_string_tokens = [
        tokens_to_strings(model.tokenizer, tokens) for tokens in few_shot_tokens
    ]

    few_shot_counterfactual_prompts = create_few_shot_prompts(
        shuffle_dataset_outputs(dataset), NUM_EXAMPLES
    )

    few_shot_counterfactual_tokens = [
        model.tokenizer(prompt, return_tensors="pt")["input_ids"]
        for prompt in few_shot_counterfactual_prompts
    ]

    few_shot_counterfactual_string_tokens = [
        tokens_to_strings(model.tokenizer, tokens)
        for tokens in few_shot_counterfactual_tokens
    ]

    paired_few_shot_prompts = list(
        zip(
            few_shot_string_tokens,
            few_shot_prompts,
            few_shot_counterfactual_string_tokens,
            few_shot_counterfactual_prompts,
        )
    )

    (
        few_shot_string_tokens,
        few_shot_prompts,
        few_shot_counterfactual_string_tokens,
        few_shot_counterfactual_prompts,
    ) = zip(
        *list(
            filter(
                lambda pair: len(pair[0]) == num_few_shot_tokens
                and len(pair[2]) == num_few_shot_tokens,
                paired_few_shot_prompts,
            )
        )
    )

    (
        few_shot_string_tokens,
        few_shot_prompts,
        few_shot_counterfactual_string_tokens,
        few_shot_counterfactual_prompts,
    ) = (
        few_shot_string_tokens[:MAX_NUM_PROMPTS],
        few_shot_prompts[:MAX_NUM_PROMPTS],
        few_shot_counterfactual_string_tokens[:MAX_NUM_PROMPTS],
        few_shot_counterfactual_prompts[:MAX_NUM_PROMPTS],
    )
    return (
        create_few_shot_prompts,
        few_shot_counterfactual_prompts,
        few_shot_counterfactual_string_tokens,
        few_shot_counterfactual_tokens,
        few_shot_prompts,
        few_shot_string_tokens,
        few_shot_tokens,
        num_few_shot_tokens,
        paired_few_shot_prompts,
        shuffle_dataset_outputs,
    )


@app.cell
def _(
    EXPORT_PATH,
    EXPORT_PDF,
    Renderer,
    TASK_IDENTIFIER,
    THRESHOLD,
    average_edge_weights,
    construct_information_flow_graph,
    few_shot_prompts,
    few_shot_string_tokens,
    find_prediction_paths,
    marimo,
    model,
    num_few_shot_tokens,
    num_layers,
    os,
):
    few_shot_graphs = [
        construct_information_flow_graph(model, prompt, THRESHOLD)
        for prompt in marimo.status.progress_bar(few_shot_prompts)
    ]

    few_shot_graph = average_edge_weights(
        few_shot_graphs, num_layers, num_few_shot_tokens
    )

    renderer = Renderer(
        model.config.num_hidden_layers,
        few_shot_string_tokens[0],
        len(few_shot_string_tokens[0]) - 1,
    )

    renderer.plot(
        find_prediction_paths(
            few_shot_graph,
            len(few_shot_string_tokens[0]) - 1,
            THRESHOLD,
        ),
        export_pdf=EXPORT_PDF,
        filename=os.path.join(EXPORT_PATH, TASK_IDENTIFIER, "few_shot_graph.pdf"),
    )
    return few_shot_graph, few_shot_graphs, renderer


@app.cell
def _(
    EXPORT_PATH,
    EXPORT_PDF,
    TASK_IDENTIFIER,
    THRESHOLD,
    average_edge_weights,
    construct_information_flow_graph,
    few_shot_counterfactual_prompts,
    few_shot_string_tokens,
    find_prediction_paths,
    marimo,
    model,
    num_few_shot_tokens,
    num_layers,
    os,
    renderer,
):
    few_shot_counterfactual_graphs = [
        construct_information_flow_graph(model, prompt, THRESHOLD)
        for prompt in marimo.status.progress_bar(few_shot_counterfactual_prompts)
    ]

    few_shot_counterfactual_graph = average_edge_weights(
        few_shot_counterfactual_graphs, num_layers, num_few_shot_tokens
    )

    renderer.plot(
        find_prediction_paths(
            few_shot_counterfactual_graph,
            len(few_shot_string_tokens[0]) - 1,
            THRESHOLD,
        ),
        export_pdf=EXPORT_PDF,
        filename=os.path.join(
            EXPORT_PATH, TASK_IDENTIFIER, "few_shot_counterfactual_graph.pdf"
        ),
    )
    return few_shot_counterfactual_graph, few_shot_counterfactual_graphs


@app.cell
def _(
    EXPORT_PATH,
    EXPORT_PDF,
    TASK_IDENTIFIER,
    THRESHOLD,
    average_edge_weights,
    compute_weight_difference,
    extract_causal_components,
    few_shot_counterfactual_graphs,
    few_shot_graph,
    few_shot_graphs,
    few_shot_string_tokens,
    find_prediction_paths,
    num_few_shot_tokens,
    num_layers,
    os,
    renderer,
):
    few_shot_differences = [
        compute_weight_difference(factual, counterfactual)
        for factual, counterfactual in zip(
            few_shot_graphs, few_shot_counterfactual_graphs
        )
    ]

    few_shot_average_difference = average_edge_weights(
        few_shot_differences, num_layers, num_few_shot_tokens
    )

    renderer.plot(
        extract_causal_components(
            few_shot_average_difference,
            THRESHOLD,
            reference_graph=find_prediction_paths(
                few_shot_graph,
                len(few_shot_string_tokens[0]) - 1,
                THRESHOLD,
            ),
        ),
        export_pdf=EXPORT_PDF,
        filename=os.path.join(
            EXPORT_PATH, TASK_IDENTIFIER, "few_shot_causal_component.pdf"
        ),
    )
    return few_shot_average_difference, few_shot_differences


if __name__ == "__main__":
    app.run()
