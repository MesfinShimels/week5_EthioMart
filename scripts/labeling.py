# scripts/labeling.py
import os

# Directories for preprocessed and labeled data
PREPROCESSED_DATA_DIR = os.path.join("data", "preprocessed")
LABELED_DATA_DIR = os.path.join("data", "labeled")
os.makedirs(LABELED_DATA_DIR, exist_ok=True)

ENTITY_TAGS = ["B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE", "O"]

def display_sentence(sentence):
    """
    Display a sentence with each token for labeling.
    """
    print("\nSentence to Label:")
    print(" ".join(sentence))

def label_sentence(sentence):
    """
    Allow user to label each token in a sentence.
    """
    labeled_tokens = []
    for token in sentence:
        print(f"\nToken: {token}")
        print(f"Select entity tag: {', '.join(ENTITY_TAGS)}")
        while True:
            tag = input("Enter tag: ").strip()
            if tag in ENTITY_TAGS:
                labeled_tokens.append((token, tag))
                break
            else:
                print("Invalid tag. Please choose from the provided list.")
    return labeled_tokens

def save_labeled_data(labeled_sentences, filename):
    """
    Save labeled data in CoNLL format.
    """
    filepath = os.path.join(LABELED_DATA_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in labeled_sentences:
            for token, tag in sentence:
                f.write(f"{token} {tag}\n")
            f.write("\n")
    print(f"Labeled data saved: {filepath}")

def label_data():
    """
    Load preprocessed data, label it, and save in CoNLL format.
    """
    preprocessed_files = [f for f in os.listdir(PREPROCESSED_DATA_DIR) if f.endswith(".txt")]
    for file in preprocessed_files:
        filepath = os.path.join(PREPROCESSED_DATA_DIR, file)
        labeled_sentences = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip().split()
                if sentence:
                    display_sentence(sentence)
                    labeled_tokens = label_sentence(sentence)
                    labeled_sentences.append(labeled_tokens)
        save_labeled_data(labeled_sentences, f"labeled_{file}")

if __name__ == "__main__":
    label_data()
