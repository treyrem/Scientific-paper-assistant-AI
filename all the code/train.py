import argparse
import os
import sys
import logging
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import accelerate

from datasets import Dataset

# Import metrics
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:
    print(
        "Error: scikit-learn is not installed.\n"
        "Install it with: pip install scikit-learn"
    )
    sys.exit(1)


def get_examples(data_folder: str):
    """
    Read section sentence files and return list of {'text', 'label'}.
    Expects: abstracts.txt, methods.txt, results.txt, conclusions.txt
    """
    label_map = {"abstract": 0, "methods": 1, "results": 2, "conclusion": 3}
    file_map = {
        "abstract": "abstracts.txt",
        "methods": "methods.txt",
        "results": "results.txt",
        "conclusion": "conclusions.txt",
    }
    examples = []
    for section, label in label_map.items():
        file_path = os.path.join(data_folder, file_map[section])
        if not os.path.isfile(file_path):
            logging.warning(f"File not found: {file_path}, skipping.")
            continue
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    examples.append({"text": text, "label": label})
    return examples


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 macro
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SciDeBERTa with validation metrics"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Directory with section .txt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model and tokenizer",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Warmup fraction of total steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Loading examples from {args.data_folder}...")

    examples = get_examples(args.data_folder)
    if not examples:
        logger.error("No examples found. Check your data_folder path and files.")
        sys.exit(1)
    ds = Dataset.from_list(examples)

    # Split into train and validation
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("KISTI-AI/scideberta-cs")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=128
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Rename and set format
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "KISTI-AI/scideberta-cs", num_labels=4
    )

    # Freeze all but last two layers
    num_layers = model.config.num_hidden_layers
    for name, param in model.deberta.named_parameters():
        if "layer" in name:
            layer_id = int(name.split("layer.")[1].split(".")[0])
            param.requires_grad = layer_id >= (num_layers - 2)
        else:
            param.requires_grad = True

    # Training arguments (basic)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Evaluate on the validation set
    logger.info("Evaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset=val_ds)
    # Log metrics both via logger and print
    logger.info(f"Validation metrics: {metrics}")
    print("Validation metrics:", metrics)

    logger.info(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
