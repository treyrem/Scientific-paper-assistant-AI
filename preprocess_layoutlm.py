
from datasets import load_dataset

# This will download ~50 GB of Parquet files, so pick only the splits you need
ds = load_dataset("maveriq/DocBank")
# ds is a DatasetDict with "train" (400 K), "validation" (50 K), "test" (50 K)
print(ds["train"].features["label"].feature.names)

# Create your label2id map for the subset of labels you care about:
wanted = ["title","paragraph","list","table","figure"]  # rename as needed
label2id = {lbl:i for i,lbl in enumerate(wanted)}

def preprocess(example):
    # keep only tokens whose label is in `wanted`; else set to -100 (ignored)
    ids = [ label2id[t] if t in label2id else -100
            for t in example["label"] ]
    return {
        "image_path": example["image"]["path"],
        "words":      example["token"],
        "boxes":      example["bounding_box"],
        "labels":     ids
    }

ds = ds.map(preprocess, remove_columns=ds["train"].column_names)

