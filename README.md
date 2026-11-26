# Swedish Grammar Scorer

A Swedish grammatical correctness scoring model developed for
the AI Sweden master's thesis selection assignment.

## Model and Dataset

- Model: https://huggingface.co/godu24/swedish-grammar-scorer
- Dataset: https://huggingface.co/datasets/godu24/swedish-grammar-training-data
- Architecture: Swedish BERT
- Parameters: 124,692,482
- Maximum sequence length: 128 tokens
- Output: Score between 0 and 1

## Performance

Public test evaluation using 120 examples:

| Metric | Baseline | Proposed Model |
|---|---:|---:|
| Spearman | 0.528 | 0.821 |
| ROC-AUC | 0.783 | 0.963 |
| Execution time | 10.1 s | 13.9 s |
| Samples/minute | 714 | 517 |

Hardware: NVIDIA Tesla T4 (Google Colab).

The cached evaluation includes model loading.
The first run including model download took 45.3 seconds.
Speed comparisons are limited by differences in hardware.

## Installation

    pip install -r requirements.txt

Recorded environment:
- Python 3.13.16
- PyTorch 2.14.1+cu130

A compatible CUDA environment may be required.

## Prediction

    python predict.py --input test.jsonl --output scores.jsonl

Input example:

    {"id": 1, "text": "Jag har köpt en ny bil."}

Output example:

    {"id": 1, "score": 0.9172777533531189}

The prediction script automatically downloads the published model.

## Training

Two-stage training approach:

1. Binary classification using 35,452 cleaned DaLAJ examples.
2. Regression fine-tuning using 382 training examples and
   96 validation examples from AI Sweden starter data.

The BERT encoder is frozen during the second stage.
Only the classification head is trained using MSE loss.

    python train.py --output-dir trained_models

Random seed: 42.

See training_config.json and stage2_split.json.

## Evaluation

Obtain the original starter repository:

    git clone https://github.com/nicher92/swedish_correctness_classifier.git starter_source
    git -C starter_source checkout a3f4f81b16694473f528975f20939e4ca586c88a

Run the official evaluation script:

    python starter_source/evaluate.py --repo . --input starter_source/data/test.jsonl --labels starter_source/data/test_labels.jsonl

## Reproducibility

- Fixed random seed: 42.
- Pinned model and dataset revisions.
- All 12 pinned dependencies match the evaluation environment.
- Dependency resolution verified using pip dry-run.
- Official PyTorch CUDA 13.0 package index configured.
- Dataset integrity verified using SHA-256.
- Group-aware training and validation split.
- Zero group overlap.
- Automatic retrieval of the original starter dataset.
- Successful evaluation from a fresh GitHub clone.
- Successful reconstruction of the training and validation split.

Known original training versions:
- Transformers: 4.57.3
- Scikit-learn: 1.7.2

Current evaluation versions:
- Transformers: 5.22.1
- Scikit-learn: 1.6.1

Full end-to-end retraining and actual clean dependency
installation remain to be verified.

## Limitations

- Inputs longer than 128 tokens are truncated.
- Context-dependent grammar may be difficult to evaluate.
- Scores are estimates, not definitive linguistic judgments.
- Hidden test performance is unknown.
- Evaluation speed depends on hardware and caching.

## Base Model

https://huggingface.co/KBLab/bert-base-swedish-cased

## Original Data Sources

- https://spraakbanken.gu.se/en/resources/dalaj-ged-superlim
- https://github.com/nicher92/swedish_correctness_classifier

## Licensing

Refer to the original model and dataset licenses.

The AI Sweden starter dataset is retrieved from its original
source rather than redistributed in the current repository version.
Redistribution permission remains to be confirmed, including
previously published copies in Git history.

No project-wide license is asserted.
