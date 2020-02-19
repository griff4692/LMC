# Clinical Acronym Expansion

## Evaluation Instruction

The main training script `evaluate.py` requires having access to a pre-trained BSG or LMC model.

1. Ensure serialized language model (LMC or BSG) state is visible in the top-level directory `~/LMC/weights/{lmc,bsg}/{experiment_name}`
2. Then, run `python evaluate.py --lm_experiment {experiment_name} --experiment my-evaluation --dataset {casi, mimic} --lm_type {bsg, lmc} --epochs 0 --train_frac 0.0`.
- `--lm_experiment` - Must match the experiment name from `~/LMC/weights/{lmc,bsg}/{experiment_name}`
- `--experiment` - Specify directory name within `~/LMC/weights/acronyms` where you want evaluation results and metrics stored.
- `--lm_type` - Specify whether to load a BSG model (from `~/LMC/weights/bsg`) or an LMC model (from `~/LMC/weights/lmc`))
- `--epochs 0 --train_frac 0.0` means that the entire dataset is used as the test set and there is no fine-tuning.  Please adjust these hyper-parameter settings if you would like to fine-tune the language model on a portion of the dataset before evaluating on a smaller test set.

## Reading Output

Based on the `--experiment` and `--dataset` flag passed to `evaluate.py`, the evaluation output on the test set will exist in a directory located at `~/LMC/weights/acronyms/{experiment_name}_{dataset}`.  It will include the following files inside `results` subdirectory:

1. `confusion/{SF}.html` - HTML confusion matrices for each acronym in test set.
2. `correct.txt` - Rendered output of correctly predicted examples
2. `errors.txt` - Rendered output of incorrectly predicted examples
3. `error_tracker.json` - Example ids for correctly and incorrectly predicted examples.  Consumed by the optional script `error_comparison`, which is useful for comparing the output of two different models.
4. `reports.txt` - per class and aggregate F-1 statistics for each SF in the test set.
5. `summary.csv` - Per SF breakdown of the results in csv format.  Helpful for generating visualizations based on model outputs.