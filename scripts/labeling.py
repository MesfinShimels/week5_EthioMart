import json
import os

def create_conll(input_file, output_file):
    """Converts preprocessed messages into CoNLL format."""
    with open(input_file, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for message in messages:
            tokens = message["text"].split()
            for token in tokens:
                # Placeholder labeling
                label = "O"  # Default to "Outside"
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # Separate sentences

if __name__ == "__main__":
    INPUT_DIR = "data/preprocessed/"
    OUTPUT_DIR = "data/labeled/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(INPUT_DIR):
        if file.startswith("preprocessed_") and file.endswith(".json"):
            input_path = os.path.join(INPUT_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, f"{file.replace('preprocessed_', '').replace('.json', '.conll')}")
            create_conll(input_path, output_path)
