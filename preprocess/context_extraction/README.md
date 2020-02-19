# Creating Reverse Substitution Acronym Expansion Dataset from MIMIC-III Clinical Notes

## Introduction

In the context of acronym expansion, Reverse Substitution (RS) is the unsupervised process of extracting acronym expansions from text and then replacing them with their known corresponding acronyms.

> Original Text: He went to physical therapy
> Modified Text: He went to PT
> Target Expansion (LF): physical therapy
> Acronym (SF): PT

This repository uses the same SF-LF sense inventory from the Minnesota CASI dataset to generate an RS dataset in MIMIC-III.

## Steps

Run the scripts in the following order:

1. `casi_lfs.py` - this just cleans the CASI acronym expansions to make them easier to locate in clinical text.
2. `mimic_contexts.py` - this scans the MIMIC-III NOTEEVENTS.csv dataset and extracts all LFs stored from step 1.
3. `mimic_contexts.py -collect` - step 2 uses multi-processing and stores the output in a temporary directory so the `collect` flag simply concatenates the chunks together.
4. `mimic_contexts.py -add_counts` - this stores associated counts for each LF and SF.  Necessary for the following step to determine which SFs we can keep (have more than one LF associated with it in text.)
5. *(Optional)* `mimic_contexts.py -render_stats` - this will show you statistics from steps 1-4.
6. `generate_dataset.py` - Filters SFs with a single target expansion and downsamples frequent expansions to create a more balanced test set.
7. `preprocess_dataset.py` - Tokenize RS dataset and readies for evaluation.
8.  `compute_metadata_marginals.py` - Computes empirical probabilities of p(LF|metadata) based on LF contexts extracted from MIMIC. These empirical counts are consumed by the LMC model when computing token marginals over metadata.

## Using the Dataset

The final dataset will be located in `./preprocess/context_extraction/data/mimic_rs_dataset_preprocessed_window_{window size}.csv`

The dataset can be used for evaluation in the `acronyms` module.  In the main `evaluate.py` script, passing the flag `--dataset mimic` will result in full evaluations on this dataset.  For other instructions, please refer to the main README.
