import json

filepath = "/Users/adamv/Library/CloudStorage/OneDrive-TUSMM/Untitled Folder/Code/LLm/SimpleLLM_V_031_PyTorch.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "[CELL 3] TOKENIZER & DATASET" in source:
            new_source = source.replace(
                "from datasets import load_dataset, interleave_datasets",
                "from datasets import load_dataset"
            )
            # Find and replace the dataset loading block
            import re
            old_block = r"# Load TinyStories\n.*?dataset = interleave_datasets\(\[tiny_train, wiki_train\], probabilities=\[0\.5, 0\.5\], seed=42\)\nval_dataset = interleave_datasets\(\[tiny_val, wiki_val\], probabilities=\[0\.5, 0\.5\], seed=42\)\n"
            new_block = (
                "# Load Wikitext (for diverse, \"adult\" vocabulary and grammar)\n"
                "dataset = load_dataset(\"wikitext\", \"wikitext-103-v1\", split=\"train\", streaming=True)\n"
                "val_dataset = load_dataset(\"wikitext\", \"wikitext-103-v1\", split=\"validation\", streaming=True)\n"
            )
            new_source = re.sub(old_block, new_block, new_source, flags=re.DOTALL)
            
            # Split back into lines to match ipynb structure
            # ipynb source lines end with \n
            lines = []
            for line in new_source.splitlines(True):
                lines.append(line)
            cell['source'] = lines

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
