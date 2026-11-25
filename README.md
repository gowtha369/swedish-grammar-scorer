
# Swedish Grammar Scorer

A Swedish grammatical correctness scoring model developed
for the AI Sweden master's thesis selection assignment.

## Model

Hugging Face:
https://huggingface.co/godu24/swedish-grammar-scorer

Dataset:
https://huggingface.co/datasets/godu24/swedish-grammar-training-data

Architecture: Swedish BERT
Parameters: 124,692,482
Maximum sequence length: 128
Output: Score between 0 and 1

## Performance

Public test evaluation (120 examples):

| Metric | Result |
|---|---:|
| Spearman | 0.821 |
| ROC-AUC | 0.963 |
| Execution time | 15.6 seconds |
| Samples/minute | 460 |

Hardware: NVIDIA Tesla T4.

## Installation

pip install -r requirements.txt

## Prediction

python predict.py --input test.jsonl --output scores.jsonl

Input example:

{"id": 1, "text": "Jag har köpt en ny bil."}

Output example:

{"id": 1, "score": 0.9172777533531189}

## Training

Two-stage training:

1. Binary classification using 35,452 DaLAJ examples.
2. Regression fine-tuning using 382 training examples
   and 96 validation examples from AI Sweden starter data.

Run:

python train.py --output-dir trained_models

Random seed: 42.

See training_config.json and stage2_split.json.

## Reproducibility

- Model and dataset revisions recorded.
- Random seed documented.
- Training configuration documented.
- Dataset checksum recorded.
- Dependency versions recorded.
- Public evaluation results saved.

Full end-to-end reproduction remains to be verified.

## Limitations

- Inputs longer than 128 tokens are truncated.
- Context-dependent grammar may be difficult to evaluate.
- Scores are estimates, not definitive linguistic judgments.
- Hidden test performance is unknown.

## Base Model

https://huggingface.co/KBLab/bert-base-swedish-cased

## Licensing

See the original model and dataset licenses.
Redistribution conditions for all training sources must
be checked before assigning a project-wide license.

