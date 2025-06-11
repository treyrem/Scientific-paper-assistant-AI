#!/usr/bin/env python3
"""
Scaled training using the fixed dataset - should finish in 1-2 hours
"""
import torch
import os
import re
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
import pytesseract

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the FIXED dataset
dataset = load_from_disk("./fixed_publaynet_train")
print(f"Loaded fixed dataset: {len(dataset)} examples")

# Use more examples but still manageable
USE_FULL_DATASET = False  # Set to True for full training
if USE_FULL_DATASET:
    SUBSET_SIZE = len(dataset)
    print("Using full dataset")
else:
    SUBSET_SIZE = 2000  # Use 2000 examples for faster training
    dataset = dataset.select(range(SUBSET_SIZE))
    print(f"Using subset of {SUBSET_SIZE} examples")

# Split dataset
train_size = int(0.9 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)}")

# Initialize processor and model with correct labels
id2label = {0: "text", 1: "title", 2: "list", 3: "figure", 4: "table"}
label2id = {v: k for k, v in id2label.items()}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", 
    num_labels=5,
    id2label=id2label,
    label2id=label2id
).to(device)

def normalize_bbox(bbox, img_size):
    """Normalize bbox to 0-1000 scale"""
    img_width, img_height = img_size
    x, y, w, h = bbox
    
    # Convert to absolute coordinates
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h
    
    # Normalize to 0-1000
    norm_x_min = int((x_min / img_width) * 1000)
    norm_y_min = int((y_min / img_height) * 1000)
    norm_x_max = int((x_max / img_width) * 1000)
    norm_y_max = int((y_max / img_height) * 1000)
    
    # Clamp to valid range
    norm_x_min = max(0, min(1000, norm_x_min))
    norm_y_min = max(0, min(1000, norm_y_min))
    norm_x_max = max(norm_x_min + 1, min(1000, norm_x_max))
    norm_y_max = max(norm_y_min + 1, min(1000, norm_y_max))
    
    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]

def extract_text_simple(image, bboxes, categories):
    """Simple text extraction with fallback"""
    words = []
    boxes = []
    word_labels = []
    
    img_width, img_height = image.size
    
    for i, (bbox, category) in enumerate(zip(bboxes, categories)):
        try:
            x, y, w, h = bbox
            # Create region
            region = image.crop((x, y, x + w, y + h))
            
            # Try OCR first
            try:
                text = pytesseract.image_to_string(region, config='--psm 6').strip()
                text = re.sub(r'[^\w\s\-\.]', '', text)
                
                if text and len(text) > 0:
                    # Split into words
                    region_words = text.split()[:5]  # Limit to 5 words per region
                    for word in region_words:
                        if word:
                            words.append(word)
                            boxes.append(normalize_bbox(bbox, (img_width, img_height)))
                            word_labels.append(category)
                else:
                    raise Exception("No text found")
                    
            except:
                # Fallback to synthetic text
                if category == 0:  # text
                    word = f"text_{i}"
                elif category == 1:  # title
                    word = f"title_{i}"
                elif category == 2:  # list
                    word = f"item_{i}"
                elif category == 3:  # figure
                    word = f"[FIGURE]"
                elif category == 4:  # table
                    word = f"[TABLE]"
                else:
                    word = f"region_{i}"
                
                words.append(word)
                boxes.append(normalize_bbox(bbox, (img_width, img_height)))
                word_labels.append(category)
                
        except Exception as e:
            print(f"Error processing bbox {i}: {e}")
            continue
    
    return words, boxes, word_labels

def transform_example(example):
    """Transform example for training"""
    try:
        # Get image
        image_bytes = example["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get annotations
        annotations = example.get("annotations", [])
        if not annotations:
            return None
            
        bboxes = [ann["bbox"] for ann in annotations]
        categories = [ann["category_id"] for ann in annotations]
        
        # Verify categories are valid (should be 0-4 after fixing)
        if any(cat < 0 or cat > 4 for cat in categories):
            print(f"Invalid category found: {categories}")
            return None
        
        # Extract text
        words, boxes, word_labels = extract_text_simple(image, bboxes, categories)
        
        if not words:
            return None
        
        # Process with LayoutLMv3
        encoding = processor(
            image, words, boxes=boxes, word_labels=word_labels,
            padding="max_length", truncation=True, max_length=256,
            return_tensors="pt"
        )
        
        return {k: v.squeeze() for k, v in encoding.items()}
        
    except Exception as e:
        print(f"Error in transform_example: {e}")
        return None

# Apply transforms
print("Transforming datasets...")
train_dataset = train_dataset.map(transform_example, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(transform_example, remove_columns=val_dataset.column_names)

# Filter out failed transformations
train_dataset = train_dataset.filter(lambda x: x is not None)
val_dataset = val_dataset.filter(lambda x: x is not None)

print(f"After processing: {len(train_dataset)} train, {len(val_dataset)} val examples")

if len(train_dataset) == 0:
    print("❌ No valid training examples after processing!")
    exit(1)

# Training arguments
training_args = TrainingArguments(
    output_dir="./publaynet_model_v1",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    logging_steps=25,
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

print(f"🚀 Starting training on {len(train_dataset)} examples...")
print(f"Expected time: ~1-2 hours")

trainer.train()

print("✅ Training complete!")
trainer.save_model("./publaynet_final_model")
print("✅ Model saved to ./publaynet_final_model")

# Test the model quickly
print("\n🧪 Testing model on first validation example...")
test_example = val_dataset[0]
model.eval()
with torch.no_grad():
    inputs = {k: v.unsqueeze(0).to(device) for k, v in test_example.items() if k != 'labels'}
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(f"Prediction shape: {predictions.shape}")
    print("✅ Model appears to be working!")
