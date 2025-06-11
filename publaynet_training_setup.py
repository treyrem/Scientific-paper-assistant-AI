#!/usr/bin/env python3
"""
PublayNet Training Script for Google Colab
- Downloads Parquet shards locally before loading
- Uses `datasets.Dataset.from_parquet` with local paths and concatenates shards
- Resolves import conflicts by streamlining dependencies and monkey-patching accelerate if needed
- Adds estimation of total training steps and duration
- Designed to run in a Colab notebook environment
"""

import pytesseract
import os
import logging
import torch
import time
import math
import urllib.request
from datasets import Dataset, concatenate_datasets
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from PIL import Image
import io
from tqdm.auto import tqdm
import re

try:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

except ImportError:
    pytesseract = None
    print("Warning: pytesseract not installed. Will use fallback synthetic OCR.")
# Monkey-patch accelerate.utils.memory.clear_device_cache if missing
try:
    import accelerate.utils.memory as _acc_mem

    if not hasattr(_acc_mem, "clear_device_cache"):
        _acc_mem.clear_device_cache = lambda *args, **kwargs: None
except ImportError:
    pass

# ----------------------------------------
# 2. Logging Setup
# ----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# 3. Parquet Shard URLs (remote)
# ----------------------------------------
train_files_urls = [
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/train-00000-of-00208-3f1d0dff7cee414a.parquet",
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/train-00001-of-00208-09600884a028020b.parquet",
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/train-00002-of-00208-71e7fc4394ba9f89.parquet",
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/train-00003-of-00208-78b4a785746f31f4.parquet",
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/train-00004-of-00208-7e2fec441fcf671d.parquet",
]
val_files_urls = [
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/validation-00000-of-00008-a39835b959a7216c.parquet",
    "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/data/validation-00001-of-00008-55d49c1129bdb061.parquet",
]

# ----------------------------------------
# 4. Download Parquet Shards Locally
# ----------------------------------------
shard_dir = "publaynet_shards"
os.makedirs(shard_dir, exist_ok=True)


def download_shards(urls, save_dir=shard_dir):
    local_paths = []
    for url in urls:
        filename = os.path.join(save_dir, os.path.basename(url))
        if not os.path.exists(filename):
            logger.info(f"Downloading {url} to {filename}...")
            urllib.request.urlretrieve(url, filename)
            logger.info(f"Downloaded {filename}")
        else:
            logger.info(f"File {filename} already exists, skipping download.")
        local_paths.append(filename)
    return local_paths


logger.info("Downloading train shards...")
train_files = download_shards(train_files_urls)
logger.info("Downloading validation shards...")
val_files = download_shards(val_files_urls)

# ----------------------------------------
# 5. Load Dataset by Concatenating Shards
# ----------------------------------------
logger.info("Loading and concatenating train shards...")
train_dsets = [Dataset.from_parquet(fp) for fp in train_files]
train_dataset = concatenate_datasets(train_dsets)
logger.info(f"Train dataset has {train_dataset.num_rows} rows.")

logger.info("Loading and concatenating validation shards...")
val_dsets = [Dataset.from_parquet(fp) for fp in val_files]
val_dataset = concatenate_datasets(val_dsets)
logger.info(f"Validation dataset has {val_dataset.num_rows} rows.")

# ----------------------------------------
# 6. Define Label Mappings and Helper Functions BEFORE they're used
# ----------------------------------------
id2label = {0: "text", 1: "title", 2: "list", 3: "figure", 4: "table"}
label2id = {v: k for k, v in id2label.items()}
NUM_LABELS = len(id2label)


def normalize_bbox(bbox, img_size):
    """Normalize bbox coordinates to 0-1000 scale for LayoutLMv3"""
    img_width, img_height = img_size
    x_min, y_min, x_max, y_max = bbox

    # Normalize to 0-1000 scale
    norm_x_min = int((x_min / img_width) * 1000)
    norm_y_min = int((y_min / img_height) * 1000)
    norm_x_max = int((x_max / img_width) * 1000)
    norm_y_max = int((y_max / img_height) * 1000)

    # Clamp values to valid range
    norm_x_min = max(0, min(1000, norm_x_min))
    norm_y_min = max(0, min(1000, norm_y_min))
    norm_x_max = max(0, min(1000, norm_x_max))
    norm_y_max = max(0, min(1000, norm_y_max))

    # Ensure max >= min
    if norm_x_max <= norm_x_min:
        norm_x_max = norm_x_min + 1
    if norm_y_max <= norm_y_min:
        norm_y_max = norm_y_min + 1

    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]


def extract_text_with_ocr(image: Image.Image, bboxes: list, categories: list):
    """Extract actual text from bounding box regions using OCR"""

    words = []
    boxes = []
    word_labels = []

    img_width, img_height = image.size

    for bbox, category in zip(bboxes, categories):
        x, y, w, h = bbox

        # Ensure coordinates are within image bounds
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(img_width - x, int(w))
        h = min(img_height - y, int(h))

        # Skip invalid bboxes
        if w <= 0 or h <= 0:
            continue

        # Crop the region
        try:
            region = image.crop((x, y, x + w, y + h))

            # Extract text using OCR with word-level bounding boxes
            ocr_data = pytesseract.image_to_data(
                region,
                output_type=pytesseract.Output.DICT,
                config="--psm 6",  # Uniform block of text
            )

            # Process OCR results
            region_words_found = False
            for i in range(len(ocr_data["text"])):
                word = ocr_data["text"][i].strip()
                confidence = float(ocr_data["conf"][i])

                # Filter out low-confidence or empty words
                if word and confidence > 30:  # Minimum confidence threshold
                    # Clean the word
                    word = re.sub(r"[^\w\s\-\.]", "", word)
                    if len(word) > 0:
                        region_words_found = True
                        # Calculate absolute word position in original image
                        word_x = x + ocr_data["left"][i]
                        word_y = y + ocr_data["top"][i]
                        word_w = ocr_data["width"][i]
                        word_h = ocr_data["height"][i]

                        # Create normalized bbox for the word
                        word_bbox = normalize_bbox(
                            [word_x, word_y, word_x + word_w, word_y + word_h],
                            (img_width, img_height),
                        )

                        words.append(word)
                        boxes.append(word_bbox)
                        word_labels.append(category)

            # If no words found, create a placeholder
            if not region_words_found:
                placeholder_bbox = normalize_bbox(
                    [x, y, x + w, y + h], (img_width, img_height)
                )
                if category == 3:  # figure
                    words.append("[FIGURE]")
                elif category == 4:  # table
                    words.append("[TABLE]")
                else:
                    words.append("[EMPTY]")
                boxes.append(placeholder_bbox)
                word_labels.append(category)

        except Exception as e:
            print(f"OCR failed for region {bbox}: {e}")
            # Fallback: create placeholder
            placeholder_bbox = normalize_bbox(
                [x, y, x + w, y + h], (img_width, img_height)
            )
            words.append(f"[REGION_{len(words)}]")
            boxes.append(placeholder_bbox)
            word_labels.append(category)

    return words, boxes, word_labels


def fallback_synthetic_ocr(image: Image.Image, bboxes: list, categories: list):
    """Fallback function when OCR is not available"""
    words = []
    boxes = []
    word_labels = []

    img_width, img_height = image.size

    for i, (bbox, category) in enumerate(zip(bboxes, categories)):
        x, y, w, h = bbox

        # Create a synthetic word for this region
        if category == 0:  # text
            word = f"text_region_{i}"
        elif category == 1:  # title
            word = f"title_{i}"
        elif category == 2:  # list
            word = f"list_item_{i}"
        elif category == 3:  # figure
            word = f"[FIGURE_{i}]"
        elif category == 4:  # table
            word = f"[TABLE_{i}]"
        else:
            word = f"region_{i}"

        # Normalize bbox
        normalized_bbox = normalize_bbox([x, y, x + w, y + h], (img_width, img_height))

        words.append(word)
        boxes.append(normalized_bbox)
        word_labels.append(category)

    return words, boxes, word_labels


def extract_text_from_regions(image: Image.Image, bboxes: list, categories: list):
    """Main function to extract text with OCR fallback"""
    # Validate categories before processing
    valid_categories = []
    valid_bboxes = []

    for bbox, category in zip(bboxes, categories):
        if 0 <= category < NUM_LABELS:  # Only keep valid categories
            valid_categories.append(category)
            valid_bboxes.append(bbox)
        else:
            print(
                f"Warning: Invalid category {category} found, skipping this annotation"
            )

    if not valid_categories:
        print("Warning: No valid categories found after filtering")
        return [], [], []

    try:
        return extract_text_with_ocr(image, valid_bboxes, valid_categories)
    except ImportError:
        print("Warning: pytesseract not installed. Using synthetic OCR.")
        return fallback_synthetic_ocr(image, valid_bboxes, valid_categories)
    except Exception as e:
        print(f"Warning: OCR failed ({e}). Using synthetic OCR.")
        return fallback_synthetic_ocr(image, valid_bboxes, valid_categories)


# ----------------------------------------
# 7. Estimate Training Duration
# ----------------------------------------

num_train_samples = train_dataset.num_rows
num_epochs = 3
batch_size = 2
steps_per_epoch = math.ceil(num_train_samples / batch_size)
total_steps = steps_per_epoch * num_epochs
logger.info(f"Number of training samples: {num_train_samples}")
logger.info(f"Batch size: {batch_size}, Epochs: {num_epochs}")
logger.info(f"Estimated total training steps: {total_steps}")

# Quick benchmark for time per step
use_benchmark = True
approx_time_per_step = None
if use_benchmark:
    logger.info("Running quick benchmark to estimate time per step...")
    # Disable OCR so we can supply bounding boxes
    processor_bench = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    dummy_model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=5
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    example = train_dataset[0]

    def transform_example_once(ex):
        # Extract raw image bytes from the ImageFeature dict
        image_bytes = ex["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        annotations = ex.get("annotations", [])
        bboxes = [ann["bbox"] for ann in annotations] if annotations else []
        categories = [ann["category_id"] for ann in annotations] if annotations else []
        words, boxes, word_labels = extract_text_from_regions(image, bboxes, categories)

        # Ensure we have at least one word for benchmarking
        if not words:
            words = ["[EMPTY]"]
            boxes = [[0, 0, 100, 100]]
            word_labels = [0]

        encoding = processor_bench(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            key: val.squeeze().to(dummy_model.device) for key, val in encoding.items()
        }

    bench_batch = transform_example_once(example)
    batch = {k: v.unsqueeze(0) for k, v in bench_batch.items()}
    dummy_model.train()
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-5)
    device = dummy_model.device
    batch = {k: v.to(device) for k, v in batch.items()}
    start_time = time.time()
    outputs = dummy_model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    approx_time_per_step = time.time() - start_time
    logger.info(f"Approximate time per step: {approx_time_per_step:.4f} seconds")
    estimated_total_time = approx_time_per_step * total_steps
    logger.info(f"Estimated total training time: {estimated_total_time/3600:.2f} hours")

# ----------------------------------------
# 8. Initialize Processor and Model
# ----------------------------------------
# Disable OCR so we can supply bounding boxes in the main pipeline
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------
# 9. Preprocessing Function for Each Example
# ----------------------------------------
def transform_example(examples):
    # Check if this is a batch or single example
    is_batched = isinstance(examples["image"], list)

    if is_batched:
        # Process batch
        batch_encoding = {
            "input_ids": [],
            "attention_mask": [],
            "bbox": [],
            "labels": [],
            "pixel_values": [],
        }

        for i in range(len(examples["image"])):
            # Extract raw image bytes from ImageFeature dict
            image_bytes = examples["image"][i]["bytes"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            annotations = (
                examples["annotations"][i] if "annotations" in examples else []
            )

            if not annotations:
                # Skip this example
                continue

            bboxes = [ann["bbox"] for ann in annotations]
            categories = [ann["category_id"] for ann in annotations]

            # Use proper OCR text extraction
            words, boxes, word_labels = extract_text_from_regions(
                image, bboxes, categories
            )

            # Skip if no words extracted
            if not words:
                continue

            # Final validation: ensure all word_labels are in valid range
            word_labels = [max(0, min(NUM_LABELS - 1, label)) for label in word_labels]

            encoding = processor(
                image,
                words,
                boxes=boxes,
                word_labels=word_labels,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            for key in encoding:
                if key in batch_encoding:
                    batch_encoding[key].append(encoding[key].squeeze())

        # Stack all tensors in the batch
        return {
            key: torch.stack(val) if val else torch.tensor([])
            for key, val in batch_encoding.items()
        }
    else:
        # Process single example
        image_bytes = examples["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        annotations = examples.get("annotations", [])

        # This should never happen after filtering, but safety check
        if not annotations:
            raise ValueError(
                "Found example without annotations after filtering - this shouldn't happen!"
            )

        bboxes = [ann["bbox"] for ann in annotations]
        categories = [ann["category_id"] for ann in annotations]

        # Use proper OCR text extraction
        words, boxes, word_labels = extract_text_from_regions(image, bboxes, categories)

        # Ensure we have at least one word
        if not words:
            words = ["[EMPTY]"]
            boxes = [[0, 0, 100, 100]]  # Default small box
            word_labels = [0]  # Default to text category

        # Final validation: ensure all word_labels are in valid range
        word_labels = [max(0, min(NUM_LABELS - 1, label)) for label in word_labels]

        encoding = processor(
            image,
            words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {key: val.squeeze() for key, val in encoding.items()}


# ----------------------------------------
# 10. Filter and Apply Transform to Datasets
# ----------------------------------------


# First, filter out examples without objects and with invalid labels
def has_valid_objects(example):
    annotations = example.get("annotations")
    if annotations is None or len(annotations) == 0:
        return False

    # Check if all category_ids are valid (0-4 for PublayNet)
    for ann in annotations:
        category_id = ann.get("category_id")
        if category_id is None or category_id < 0 or category_id >= NUM_LABELS:
            return False

    return True


logger.info(
    "Filtering datasets to remove examples without objects and invalid labels..."
)
original_train_len = len(train_dataset)
original_val_len = len(val_dataset)

# Debug: Check some examples before filtering
logger.info("Checking first few examples for label ranges...")
for i in range(min(5, len(train_dataset))):
    example = train_dataset[i]
    annotations = example.get("annotations", [])
    if annotations:
        categories = [ann["category_id"] for ann in annotations]
        logger.info(f"Example {i} categories: {categories}")

train_dataset = train_dataset.filter(has_valid_objects)
val_dataset = val_dataset.filter(has_valid_objects)

logger.info(f"Filtered train dataset: {original_train_len} -> {len(train_dataset)}")
logger.info(f"Filtered val dataset: {original_val_len} -> {len(val_dataset)}")

# Update training estimates with filtered dataset sizes
num_train_samples = len(train_dataset)
steps_per_epoch = math.ceil(num_train_samples / batch_size)
total_steps = steps_per_epoch * num_epochs
logger.info(f"Updated training estimates after filtering:")
logger.info(f"Number of training samples: {num_train_samples}")
logger.info(f"Estimated total training steps: {total_steps}")
if approx_time_per_step:
    estimated_total_time = approx_time_per_step * total_steps
    logger.info(
        f"Updated estimated total training time: {estimated_total_time/3600:.2f} hours"
    )

# Now apply transforms to the filtered datasets
logger.info("Transforming train split...")
train_dataset = train_dataset.with_transform(transform_example)
logger.info("Transforming validation split...")
val_dataset = val_dataset.with_transform(transform_example)

# ----------------------------------------
# 11. Data Collator
# ----------------------------------------
data_collator = DataCollatorForTokenClassification(processor.tokenizer)

# ----------------------------------------
# 12. Training Arguments
# ----------------------------------------
training_args = TrainingArguments(
    output_dir="./publaynet_output",
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=2,
    eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    remove_unused_columns=False,
    report_to=[],
)

# ----------------------------------------
# 13. Initialize Trainer
# ----------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,  # Changed from tokenizer to processing_class
)

# ----------------------------------------
# 14. Start Training
# ----------------------------------------
logger.info("Starting training...")

# Debug: Check if datasets are actually populated
print(f"Final train dataset length: {len(train_dataset)}")
print(f"Final val dataset length: {len(val_dataset)}")

trainer.train()
logger.info("Training complete!")
logger.info("Saving model to ./publaynet_output/final_model")
trainer.save_model("./publaynet_output/final_model")
