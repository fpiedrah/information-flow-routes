#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --tasks-per-node=1
#SBATCH --job-name train_model
#SBATCH --partition gpu-he --gres=gpu:1
#SBATCH --time 2:00:00
#SBATCH --array=0-7

INSTRUCTIONS_ARRAY=(
        "Output the antonym of the word in the question."
        "Return the word in the input with the first letter capitalized."
        "Return the uppercase letter of the first letter of the input string."
        "What is the capital of the country?"
        "Return the first letter of the word in lower case."
        "Write the word in its past tense form."
        "What is the plural of the following word?"
        "What is the synonym?"
)

INSTRUCTIONS_ARRAY=(
        "002-ANTONYM"
        "003-SENTENCE_CASE"
        "004-CAPITALIZE_FIRST_LETTER"
        "008-COUNTRY_CAPITAL_MAPPING"
        "014-LOWERCASE_FIRST_LETTER"
        "023-VERB_TO_PAST_TENSE"
        "027-NOUN_TO_PLURAL"
        "028-SYNONYM"
)

DATASETS_ARRAY=(
        "002-antonym.json"
        "003-sentence_case.json"
        "004-capitalize_first_letter.json"
        "008-country_capital_mapping.json"
        "014-lowercase_first_letter.json"
        "023-verb_to_past_tense.json"
        "027-noun_to_plural.json"
        "028-synonym.json"
)

# SLURM
INDEX=$SLURM_ARRAY_TASK_ID

# GENERAL
TASK_IDENTIFIER=${TASKS_ARRAY[$INDEX]}
DATASET_BASE_PATH="../datasets/"
DATASET_FILE=${DATASETS_ARRAY[$INDEX]}

# PROMPT CONFIGURATION
INSTRUCTIONS=${INSTRUCTIONS_ARRAY[$INDEX]}

python -u counterfactual_template.py -- \
    --TASK_IDENTIFIER "$TASK_IDENTIFIER" \
    --DATASET_BASE_PATH "$DATASET_BASE_PATH" \
    --INSTRUCTIONS "$INSTRUCTIONS"
