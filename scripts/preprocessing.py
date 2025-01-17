import json
import os
import re

def clean_text(text):
    """Cleans and normalizes Amharic text."""
    if not text:
        return ""
    # Example normalization: remove special characters, normalize spaces
    text = re.sub(r"[^\w\s፡።፣፤፥፦፧፨]", "", text)  # Remove non-Amharic symbols
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def preprocess_messages(input_file, output_file):
    """Reads raw messages, cleans text, and saves preprocessed data."""
    with open(input_file, "r", encoding="utf-8") as f:
        messages = json.load(f)
    preprocessed = []
    for message in messages:
        cleaned_text = clean_text(message["text"])
        preprocessed.append({
            "id": message["id"],
            "text": cleaned_text,
            "timestamp": message["timestamp"],
            "sender_id": message["sender_id"]
        })
    # Save preprocessed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(preprocessed, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    INPUT_DIR = "data/raw/"
    OUTPUT_DIR = "data/preprocessed/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".json"):
            input_path = os.path.join(INPUT_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, f"preprocessed_{file}")
            preprocess_messages(input_path, output_path)
