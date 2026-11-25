
import argparse
import copy
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


ROOT = Path(__file__).resolve().parent

with open(ROOT / "training_config.json") as f:
    CONFIG = json.load(f)

with open(ROOT / "stage2_split.json") as f:
    SPLIT = json.load(f)

SEED = CONFIG["random_seed"]
MAX_LENGTH = CONFIG["max_length"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class GrammarDataset(Dataset):

    def __init__(self, rows, tokenizer, regression=False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.regression = regression

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        row = self.rows[idx]

        item = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in item.items()
        }

        if self.regression:
            item["score"] = torch.tensor(
                row["score"], dtype=torch.float32
            )
        else:
            item["labels"] = torch.tensor(
                row["label"], dtype=torch.long
            )

        return item


def make_loader(rows, tokenizer, batch_size,
                shuffle=False, regression=False):

    generator = torch.Generator().manual_seed(SEED)

    return DataLoader(
        GrammarDataset(rows, tokenizer, regression),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator
    )


def train_stage1(output_dir):

    set_seed()

    cfg = CONFIG["stage1"]

    data_path = hf_hub_download(
        repo_id=cfg["dataset"],
        filename="dalaj_train_final.jsonl",
        repo_type="dataset",
        revision=cfg["dataset_revision"]
    )

    with open(data_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == cfg["training_examples"]

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        revision=CONFIG["base_model_revision"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"],
        revision=CONFIG["base_model_revision"],
        num_labels=CONFIG["num_labels"]
    ).to(DEVICE)

    loader = make_loader(
        rows, tokenizer, cfg["batch_size"], shuffle=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg["epochs"]):

        model.train()
        total_loss = 0

        for batch in loader:

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp
            ):
                loss = model(**batch).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["gradient_clipping"]
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(
            f"Stage 1 Epoch {epoch + 1}: "
            f"Loss={total_loss / len(loader):.4f}"
        )

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("Stage 1 completed:", output_dir)


def load_stage2_data():

    repo = ROOT / "starter_source"

    if not repo.exists():
        subprocess.run(
            ["git", "clone", SPLIT["repository"], str(repo)],
            check=True
        )

    subprocess.run(
        ["git", "-C", str(repo), "checkout", SPLIT["revision"]],
        check=True
    )

    source = repo / SPLIT["source_file"]

    checksum = hashlib.sha256(source.read_bytes()).hexdigest()

    assert checksum == SPLIT["source_sha256"]

    df = pd.read_json(source, lines=True)

    indexed = df.set_index("id", drop=False)

    train_df = indexed.loc[SPLIT["training_ids"]]
    val_df = indexed.loc[SPLIT["validation_ids"]]

    assert len(train_df) == 382
    assert len(val_df) == 96

    assert not (
        set(train_df["group"]) & set(val_df["group"])
    )

    return (
        train_df.to_dict("records"),
        val_df.to_dict("records")
    )


def train_stage2(stage1_dir, output_dir):

    set_seed()

    cfg = CONFIG["stage2"]

    tokenizer = AutoTokenizer.from_pretrained(stage1_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        stage1_dir
    ).to(DEVICE)

    for parameter in model.bert.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    train_rows, val_rows = load_stage2_data()

    train_loader = make_loader(
        train_rows,
        tokenizer,
        cfg["batch_size"],
        shuffle=True,
        regression=True
    )

    val_loader = make_loader(
        val_rows,
        tokenizer,
        cfg["validation_batch_size"],
        regression=True
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    criterion = torch.nn.MSELoss()

    best_mae = float("inf")
    best_state = None

    for epoch in range(cfg["epochs"]):

        model.train()
        total_loss = 0

        for batch in train_loader:

            targets = batch.pop("score").to(DEVICE)

            inputs = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            logits = model(**inputs).logits

            scores = torch.softmax(
                logits.float(), dim=-1
            )[:, 1]

            loss = criterion(scores, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        predictions = []
        actual = []

        with torch.inference_mode():

            for batch in val_loader:

                targets = batch.pop("score")

                inputs = {
                    k: v.to(DEVICE)
                    for k, v in batch.items()
                }

                scores = torch.softmax(
                    model(**inputs).logits.float(),
                    dim=-1
                )[:, 1]

                predictions.extend(scores.cpu().tolist())
                actual.extend(targets.tolist())

        mae = float(
            np.mean(
                np.abs(
                    np.array(predictions) - np.array(actual)
                )
            )
        )

        print(
            f"Stage 2 Epoch {epoch + 1}: "
            f"Loss={total_loss / len(train_loader):.4f}, "
            f"Validation MAE={mae:.4f}"
        )

        if mae < best_mae:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print("Stage 2 completed:", output_dir)
    print("Best validation MAE:", best_mae)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="trained_models"
    )

    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stage1_dir = output / "swedish_bert_stage1"
    final_dir = output / "swedish_bert_regression"

    print("Training device:", DEVICE)

    train_stage1(stage1_dir)
    train_stage2(stage1_dir, final_dir)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
