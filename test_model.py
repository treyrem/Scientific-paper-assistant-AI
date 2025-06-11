#!/usr/bin/env python3
"""
Quick test of the trained model
"""
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datasets import load_from_disk
from PIL import Image
import io

# Load model and processor
model = LayoutLMv3ForTokenClassification.from_pretrained("./quick_model")
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)

# Load test data
dataset = load_from_disk("./fixed_publaynet_train")
test_example = dataset[0]  # Get first example

# Prepare input
image_bytes = test_example["image"]["bytes"]
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
annotations = test_example["annotations"]

# Simple dummy processing for testing
words = [f"word_{i}" for i in range(len(annotations))]
boxes = [[0, 0, 100, 100] for _ in annotations]

# Process
encoding = processor(
    image,
    words,
    boxes=boxes,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=-1)

print("🧪 Model Test Results:")
print(f"Input shape: {encoding['input_ids'].shape}")
print(f"Output shape: {outputs.logits.shape}")
print(f"Predictions shape: {predictions.shape}")

# Show some predictions
pred_labels = predictions[0].cpu().numpy()
valid_preds = pred_labels[pred_labels != -100]  # Remove padding tokens

id2label = {0: "text", 1: "title", 2: "list", 3: "figure", 4: "table"}
print(f"Predicted labels: {[id2label.get(p, 'unknown') for p in valid_preds[:10]]}")
print("✅ Model is working and making predictions!")
