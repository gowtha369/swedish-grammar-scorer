
import argparse
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_ID = "godu24/swedish-grammar-scorer"
REVISION = "a82985adbf612173e18b084002359f7a5a134c72"
MAX_LENGTH = 128
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser(
        description="Score Swedish grammatical correctness."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=REVISION
    ).to(device)

    model.eval()

    with open(args.input, encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    with open(args.output, "w", encoding="utf-8") as f:
        with torch.inference_mode():

            for start in range(0, len(records), BATCH_SIZE):

                batch = records[start:start + BATCH_SIZE]
                texts = [row["text"] for row in batch]

                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)

                scores = torch.softmax(
                    outputs.logits.float(),
                    dim=-1
                )[:, 1]

                scores = scores.clamp(0, 1).cpu().tolist()

                for row, score in zip(batch, scores):
                    result = {
                        "id": row["id"],
                        "score": float(score)
                    }

                    f.write(
                        json.dumps(
                            result,
                            ensure_ascii=False
                        ) + "\n"
                    )


if __name__ == "__main__":
    main()
