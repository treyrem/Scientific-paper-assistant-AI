#!/usr/bin/env python3
"""
Debug script to check PublayNet data quality and fix category issues
"""
import os
from datasets import Dataset, concatenate_datasets
from collections import Counter

# Load your existing data
shard_dir = "publaynet_shards"
train_files = [
    os.path.join(shard_dir, "train-00000-of-00208-3f1d0dff7cee414a.parquet"),
    os.path.join(shard_dir, "train-00001-of-00208-09600884a028020b.parquet"),
    os.path.join(shard_dir, "train-00002-of-00208-71e7fc4394ba9f89.parquet"),
    os.path.join(shard_dir, "train-00003-of-00208-78b4a785746f31f4.parquet"),
    os.path.join(shard_dir, "train-00004-of-00208-7e2fec441fcf671d.parquet"),
]

print("Loading datasets...")
train_dsets = [Dataset.from_parquet(fp) for fp in train_files]
train_dataset = concatenate_datasets(train_dsets)

print(f"Dataset loaded: {len(train_dataset)} examples")

# Analyze category distribution
print("\n=== ANALYZING CATEGORY DISTRIBUTION ===")
all_categories = []
invalid_examples = []
valid_examples = []

for i, example in enumerate(train_dataset):
    annotations = example.get("annotations", [])
    if not annotations:
        continue
    
    example_categories = [ann["category_id"] for ann in annotations]
    all_categories.extend(example_categories)
    
    # Check if example has invalid categories
    has_invalid = any(cat < 0 or cat > 4 for cat in example_categories)
    
    if has_invalid:
        invalid_examples.append((i, example_categories))
    else:
        valid_examples.append(i)
    
    if i < 10:  # Print first 10 examples
        print(f"Example {i}: categories = {example_categories}")

print(f"\nCategory distribution: {Counter(all_categories)}")
print(f"Valid examples: {len(valid_examples)}")
print(f"Invalid examples: {len(invalid_examples)}")

if invalid_examples:
    print(f"\nFirst 10 invalid examples:")
    for i, cats in invalid_examples[:10]:
        print(f"  Example {i}: {cats}")

# Check if categories need remapping
unique_cats = set(all_categories)
print(f"\nUnique categories found: {sorted(unique_cats)}")

# PublayNet should have categories 0-4, but your data might have 1-5
if 5 in unique_cats and 0 not in unique_cats:
    print("\n🚨 ISSUE DETECTED: Categories are 1-5 instead of 0-4!")
    print("This needs to be fixed by subtracting 1 from all category_ids")
    
    # Create a fixed dataset
    def fix_categories(example):
        annotations = example.get("annotations", [])
        if annotations:
            for ann in annotations:
                ann["category_id"] = ann["category_id"] - 1  # Convert 1-5 to 0-4
        return example
    
    print("Creating fixed dataset...")
    fixed_dataset = train_dataset.map(fix_categories)
    
    # Verify the fix
    test_example = fixed_dataset[4]  # The one that had [1,1,1,1,5,5,5,5,2,2,2,2]
    test_cats = [ann["category_id"] for ann in test_example["annotations"]]
    print(f"Fixed example categories: {test_cats}")
    
    # Save fixed dataset
    print("Saving fixed dataset...")
    fixed_dataset.save_to_disk("./fixed_publaynet_train")
    print("✅ Fixed dataset saved to ./fixed_publaynet_train")
    
else:
    print("✅ Categories appear to be in correct range 0-4")

print("\n=== SUMMARY ===")
print(f"Total examples: {len(train_dataset)}")
print(f"Valid examples: {len(valid_examples)}")
print(f"Invalid examples: {len(invalid_examples)}")
print(f"Data quality: {len(valid_examples)/len(train_dataset)*100:.1f}% valid")