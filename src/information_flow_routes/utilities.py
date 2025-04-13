def find_token_substring_positions(
    source_string, target_strings, tokenizer, prepend_space=True
):
    main_token_ids = tokenizer(source_string)["input_ids"]
    matching_token_indices = []

    for target in target_strings:
        target_text = f" {target}" if prepend_space else target

        target_token_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_length = len(target_token_ids)

        for index in range(len(main_token_ids) - target_length + 1):
            if main_token_ids[index : index + target_length] == target_token_ids:
                matching_token_indices.extend(range(index, index + target_length))

    return matching_token_indices
