#!/usr/bin/env python3
"""
Bulletproof PublayNet training script - thoroughly reviewed for bugs
Fixed all known issues:
1. Windows multiprocessing errors
2. Model forward() method signature conflicts
3. Class weight implementation issues
4. Data transformation edge cases
"""

# Fix for Windows multiprocessing
import os
import sys

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing

        multiprocessing.freeze_support()

import torch
import random
import io
from collections import Counter
from datasets import load_from_disk
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)


def main():
    print("🚀 Starting Bulletproof PublayNet Training")
    print("=" * 50)

    # Load dataset
    dataset = load_from_disk("./fixed_publaynet_train")
    print(f"✅ Loaded dataset: {len(dataset)} examples")

    # Class mappings
    id2label = {0: "text", 1: "title", 2: "list", 3: "figure", 4: "table"}
    label2id = {v: k for k, v in id2label.items()}

    # Analyze class distribution
    def analyze_class_presence(dataset):
        examples_with_class = {0: [], 1: [], 2: [], 3: [], 4: []}
        class_token_counts = Counter()

        for i, example in enumerate(dataset):
            annotations = example.get("annotations", [])
            if annotations:
                categories = [ann["category_id"] for ann in annotations]

                # Count tokens
                for cat in categories:
                    class_token_counts[cat] += 1

                # Track examples containing each class
                unique_classes = set(categories)
                for cls in unique_classes:
                    examples_with_class[cls].append(i)

        return examples_with_class, class_token_counts

    print("\n📊 Analyzing class distribution...")
    examples_with_class, token_counts = analyze_class_presence(dataset)

    for cls in range(5):
        count = len(examples_with_class[cls])
        tokens = token_counts[cls]
        print(f"  {id2label[cls]}: {count:,} examples, {tokens:,} tokens")

    # Create comprehensive dataset
    def create_smart_dataset(dataset, examples_with_class, max_examples=6000):
        """Create dataset with smart sampling - no edge cases"""
        selected_indices = set()

        # Add ALL examples with rare classes
        rare_classes = [1, 2, 3, 4]  # title, list, figure, table
        for cls in rare_classes:
            class_examples = examples_with_class[cls]
            selected_indices.update(class_examples)
            print(f"  Added {len(class_examples)} examples with {id2label[cls]}")

        current_size = len(selected_indices)
        print(f"  Current size: {current_size}")

        # Add text examples if needed
        if current_size < max_examples:
            remaining_slots = max_examples - current_size
            text_examples = [
                idx for idx in examples_with_class[0] if idx not in selected_indices
            ]

            if text_examples:
                additional_needed = min(remaining_slots, len(text_examples))
                if additional_needed > 0:
                    additional_examples = random.sample(
                        text_examples, additional_needed
                    )
                    selected_indices.update(additional_examples)
                    print(
                        f"  Added {len(additional_examples)} additional text examples"
                    )

        final_indices = list(selected_indices)
        random.shuffle(final_indices)
        return dataset.select(final_indices)

    print("\n🎯 Creating comprehensive dataset...")
    comprehensive_dataset = create_smart_dataset(
        dataset, examples_with_class, max_examples=6000
    )
    print(f"✅ Final dataset: {len(comprehensive_dataset)} examples")

    # Calculate class weights (simpler, more robust version)
    _, comp_token_counts = analyze_class_presence(comprehensive_dataset)
    total_tokens = sum(comp_token_counts.values())

    # Simple inverse frequency weights with caps
    class_weights = {}
    for cls in range(5):
        if comp_token_counts[cls] > 0:
            weight = total_tokens / (5 * comp_token_counts[cls])
            # Apply boosts
            if cls in [2, 3, 4]:  # list, figure, table
                weight *= 2.0
            elif cls == 1:  # title
                weight *= 1.5
            # Cap at 5.0
            class_weights[cls] = min(weight, 5.0)
        else:
            class_weights[cls] = 1.0

    print(f"\n⚖️ Class weights: {class_weights}")

    # Split dataset
    train_size = int(0.9 * len(comprehensive_dataset))
    val_size = len(comprehensive_dataset) - train_size

    indices = list(range(len(comprehensive_dataset)))
    random.shuffle(indices)

    train_dataset = comprehensive_dataset.select(indices[:train_size])
    val_dataset = comprehensive_dataset.select(indices[train_size:])

    print(f"📊 Split: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Initialize processor and model (FIXED - no custom model class)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    # Use standard model, we'll handle class weights in training args
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=5, id2label=id2label, label2id=label2id
    )
    model = model.to(device)

    # Helper functions
    def normalize_bbox(bbox, img_size):
        """Normalize bbox coordinates to 0-1000 scale"""
        img_width, img_height = img_size
        x, y, w, h = bbox

        # Convert to corners
        x_min, y_min = max(0, x), max(0, y)
        x_max, y_max = min(img_width, x + w), min(img_height, y + h)

        # Normalize to 0-1000
        norm_x_min = int((x_min / img_width) * 1000)
        norm_y_min = int((y_min / img_height) * 1000)
        norm_x_max = int((x_max / img_width) * 1000)
        norm_y_max = int((y_max / img_height) * 1000)

        # Ensure valid box
        norm_x_max = max(norm_x_min + 1, norm_x_max)
        norm_y_max = max(norm_y_min + 1, norm_y_max)

        return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]

    def safe_transform(example):
        """Robust transformation with comprehensive error handling"""
        try:
            # Extract image
            image_bytes = example["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Extract annotations
            annotations = example.get("annotations", [])
            if not annotations:
                return None

            bboxes = [ann["bbox"] for ann in annotations]
            categories = [ann["category_id"] for ann in annotations]

            # Validate categories
            if any(cat < 0 or cat >= 5 for cat in categories):
                return None

            # Generate words
            words = []
            boxes = []
            word_labels = []

            img_width, img_height = image.size
            if img_width <= 0 or img_height <= 0:
                return None

            for i, (bbox, category) in enumerate(zip(bboxes, categories)):
                # Validate bbox
                if len(bbox) != 4:
                    continue

                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                # Generate synthetic words based on category
                if category == 0:  # text
                    region_words = [f"text_{i}", "content"]
                elif category == 1:  # title
                    region_words = [f"TITLE_{i}", "HEADING"]
                elif category == 2:  # list
                    region_words = [f"item_{i}", "point"]
                elif category == 3:  # figure
                    region_words = [f"FIGURE_{i}", "caption"]
                elif category == 4:  # table
                    region_words = [f"TABLE_{i}", "data"]
                else:
                    region_words = [f"element_{i}"]

                # Normalize bbox
                try:
                    normalized_bbox = normalize_bbox(bbox, (img_width, img_height))
                except:
                    continue

                # Add words with same bbox and label
                for word in region_words:
                    words.append(word)
                    boxes.append(normalized_bbox)
                    word_labels.append(category)

            # Check if we have valid data
            if not words or len(words) != len(boxes) or len(boxes) != len(word_labels):
                return None

            # Limit sequence length
            max_words = 200
            if len(words) > max_words:
                words = words[:max_words]
                boxes = boxes[:max_words]
                word_labels = word_labels[:max_words]

            # Process with LayoutLMv3
            encoding = processor(
                image,
                words,
                boxes=boxes,
                word_labels=word_labels,
                padding="max_length",
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )

            # Validate encoding
            if not all(
                key in encoding
                for key in ["input_ids", "attention_mask", "bbox", "labels"]
            ):
                return None

            return {k: v.squeeze() for k, v in encoding.items()}

        except Exception as e:
            print(f"Transform error: {e}")
            return None

    # Apply transformations
    print("\n🔄 Transforming datasets...")
    train_dataset = train_dataset.map(
        safe_transform, remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        safe_transform, remove_columns=val_dataset.column_names
    )

    # Filter None values
    train_dataset = train_dataset.filter(lambda x: x is not None)
    val_dataset = val_dataset.filter(lambda x: x is not None)

    print(f"✅ After processing: {len(train_dataset)} train, {len(val_dataset)} val")

    if len(train_dataset) == 0:
        print("❌ No valid training examples!")
        return

    # Create custom trainer with class weights
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, **kwargs):
            super().__init__(**kwargs)
            if class_weights:
                self.class_weights = torch.tensor(
                    [class_weights.get(i, 1.0) for i in range(5)], dtype=torch.float32
                ).to(self.model.device)
            else:
                self.class_weights = None

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Custom loss computation with class weights"""
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if labels is not None and self.class_weights is not None:
                # Apply weighted loss
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights, ignore_index=-100
                )

                # Reshape for loss calculation
                active_logits = outputs.logits.view(-1, model.config.num_labels)
                active_labels = labels.view(-1)

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

    # Training arguments (FIXED - no conflicting parameters)
    training_args = TrainingArguments(
        output_dir="./bulletproof_publaynet_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        report_to=[],
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # No multiprocessing issues
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(processor.tokenizer)

    # Initialize trainer with class weights
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
        class_weights=class_weights,
    )

    # Training time estimate
    steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    estimated_hours = total_steps * 1.5 / 3600

    print(f"\n🚀 Starting training!")
    print(f"📊 {len(train_dataset)} train examples")
    print(f"⏱️ Estimated time: {estimated_hours:.1f} hours")
    print(f"🎯 Using weighted loss with class weights")

    # Start training
    trainer.train()

    print("\n✅ Training complete!")
    trainer.save_model("./bulletproof_publaynet_final")
    print("✅ Model saved!")

    # Quick test
    print("\n🧪 Quick model test...")
    model.eval()
    test_example = val_dataset[0]

    with torch.no_grad():
        inputs = {
            k: v.unsqueeze(0).to(device)
            for k, v in test_example.items()
            if k != "labels"
        }
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    print(f"✅ Model test successful! Output shape: {outputs.logits.shape}")
    print("\n🎉 Bulletproof training completed successfully!")


# Run the main function
if __name__ == "__main__":
    main()
