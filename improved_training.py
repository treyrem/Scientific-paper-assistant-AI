#!/usr/bin/env python3
"""
Improved training with class balancing to fix the "everything is text" problem
"""
import torch
import os
from datasets import load_from_disk
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from PIL import Image
import io
import random
from collections import Counter
import numpy as np

# Load the dataset
dataset = load_from_disk("./fixed_publaynet_train")
print(f"Loaded dataset: {len(dataset)} examples")

# Analyze class distribution
def analyze_class_distribution(dataset):
    """Analyze how many examples have each class"""
    class_counts = Counter()
    examples_by_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for i, example in enumerate(dataset):
        annotations = example.get("annotations", [])
        if annotations:
            categories = [ann["category_id"] for ann in annotations]
            # Check which classes are present in this example
            unique_classes = set(categories)
            for cls in unique_classes:
                class_counts[cls] += 1
                examples_by_class[cls].append(i)
    
    return class_counts, examples_by_class

print("Analyzing class distribution...")
class_counts, examples_by_class = analyze_class_distribution(dataset)

print(f"Examples containing each class:")
id2label = {0: "text", 1: "title", 2: "list", 3: "figure", 4: "table"}
for cls in range(5):
    label = id2label[cls]
    count = class_counts[cls]
    print(f"  {label}: {count} examples ({count/len(dataset)*100:.1f}%)")

# CREATE BALANCED DATASET
def create_balanced_dataset(dataset, examples_by_class, samples_per_class=300):
    """Create a more balanced dataset"""
    selected_indices = []
    
    for cls in range(5):
        available_examples = examples_by_class[cls]
        if len(available_examples) >= samples_per_class:
            # Randomly sample if we have enough
            selected = random.sample(available_examples, samples_per_class)
        else:
            # Use all available examples if we don't have enough
            selected = available_examples
            print(f"  ⚠️ Only {len(selected)} examples available for class {id2label[cls]}")
        
        selected_indices.extend(selected)
    
    # Remove duplicates while preserving order
    unique_indices = []
    seen = set()
    for idx in selected_indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)
    
    return dataset.select(unique_indices)

print(f"\nCreating balanced dataset with ~300 examples per class...")
balanced_dataset = create_balanced_dataset(dataset, examples_by_class, samples_per_class=300)
print(f"Balanced dataset size: {len(balanced_dataset)} examples")

# Verify the balance
bal_class_counts, _ = analyze_class_distribution(balanced_dataset)
print(f"Balanced distribution:")
for cls in range(5):
    label = id2label[cls]
    count = bal_class_counts[cls]
    print(f"  {label}: {count} examples ({count/len(balanced_dataset)*100:.1f}%)")

# Split dataset
train_size = int(0.8 * len(balanced_dataset))
train_dataset = balanced_dataset.select(range(train_size))
val_dataset = balanced_dataset.select(range(train_size, len(balanced_dataset)))

print(f"\nDataset split: {len(train_dataset)} train, {len(val_dataset)} validation")

# Initialize model with class weights to handle remaining imbalance
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Calculate class weights
total_samples = sum(bal_class_counts.values())
class_weights = {}
for cls in range(5):
    if bal_class_counts[cls] > 0:
        weight = total_samples / (5 * bal_class_counts[cls])  # Inverse frequency
        class_weights[cls] = weight
    else:
        class_weights[cls] = 1.0

print(f"Class weights: {class_weights}")

class WeightedLayoutLMv3(LayoutLMv3ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # Create class weights tensor
        weights = [class_weights.get(i, 1.0) for i in range(5)]
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, **inputs):
        outputs = super().forward(**inputs)
        if outputs.loss is not None and hasattr(self, 'class_weights'):
            # Apply class weights to loss
            labels = inputs.get('labels')
            if labels is not None:
                # Move class weights to same device as labels
                class_weights = self.class_weights.to(labels.device)
                
                # Create loss function with weights
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                
                # Reshape for loss calculation
                active_logits = outputs.logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                
                # Calculate weighted loss
                weighted_loss = loss_fct(active_logits, active_labels)
                outputs.loss = weighted_loss
        
        return outputs

# Initialize weighted model
model = WeightedLayoutLMv3.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=5,
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()}
)
model = model.to(device)

# Your existing transform function (simplified)
def normalize_bbox(bbox, img_size):
    img_width, img_height = img_size
    x, y, w, h = bbox
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h
    
    norm_x_min = max(0, min(1000, int((x_min / img_width) * 1000)))
    norm_y_min = max(0, min(1000, int((y_min / img_height) * 1000)))
    norm_x_max = max(norm_x_min + 1, min(1000, int((x_max / img_width) * 1000)))
    norm_y_max = max(norm_y_min + 1, min(1000, int((y_max / img_height) * 1000)))
    
    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]

def simple_transform(example):
    try:
        image_bytes = example["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        annotations = example.get("annotations", [])
        
        if not annotations:
            return None
        
        bboxes = [ann["bbox"] for ann in annotations]
        categories = [ann["category_id"] for ann in annotations]
        
        # Simple word generation
        words = []
        boxes = []
        word_labels = []
        
        img_width, img_height = image.size
        
        for i, (bbox, category) in enumerate(zip(bboxes, categories)):
            if category == 0:  # text
                word = f"content_{i}"
            elif category == 1:  # title  
                word = f"TITLE_{i}"  # Make titles more distinctive
            elif category == 2:  # list
                word = f"item_{i}"
            elif category == 3:  # figure
                word = f"[FIGURE_{i}]"
            elif category == 4:  # table
                word = f"[TABLE_{i}]"
            else:
                word = f"element_{i}"
            
            words.append(word)
            boxes.append(normalize_bbox(bbox, (img_width, img_height)))
            word_labels.append(category)
        
        if not words:
            return None
        
        encoding = processor(
            image, words, boxes=boxes, word_labels=word_labels,
            padding="max_length", truncation=True, max_length=256,
            return_tensors="pt"
        )
        
        return {k: v.squeeze() for k, v in encoding.items()}
    
    except Exception as e:
        print(f"Transform error: {e}")
        return None

# Apply transforms
print("Transforming datasets...")
train_dataset = train_dataset.map(simple_transform, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(simple_transform, remove_columns=val_dataset.column_names)

# Filter None values
train_dataset = train_dataset.filter(lambda x: x is not None)
val_dataset = val_dataset.filter(lambda x: x is not None)

print(f"After processing: {len(train_dataset)} train, {len(val_dataset)} val examples")

# Training arguments
training_args = TrainingArguments(
    output_dir="./balanced_publaynet_model",
    overwrite_output_dir=True,
    num_train_epochs=3,  # More epochs for better learning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    remove_unused_columns=False,
    report_to=[],
    learning_rate=2e-5,
    warmup_steps=50,
    fp16=torch.cuda.is_available(),
)

# Data collator
data_collator = DataCollatorForTokenClassification(processor.tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor,
)

print(f"🚀 Starting balanced training on {len(train_dataset)} examples...")
print("This should give much better class predictions!")

trainer.train()

print("✅ Balanced training complete!")
trainer.save_model("./balanced_publaynet_final")
print("✅ Balanced model saved!")

print("\n📊 The balanced model should now predict all classes more accurately!")
print("🧪 Test it with the same test script to see the improvement.")
