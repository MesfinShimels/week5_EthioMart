# scripts/model_comparison.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_metric

# Paths
LABELED_DATA_DIR = os.path.join("data", "labeled")
MODEL_OUTPUT_DIR = os.path.join("models", "fine_tuned")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Models to compare
MODELS_TO_COMPARE = {
    "xlm-roberta-base": "XLM-Roberta",
    "distilbert-base-multilingual-cased": "DistilBERT",
    "bert-base-multilingual-cased": "mBERT"
}

LABEL_MAP = {"O": 0, "B-Product": 1, "I-Product": 2, "B-LOC": 3, "I-LOC": 4, "B-PRICE": 5, "I-PRICE": 6}
NUM_LABELS = len(LABEL_MAP)

def load_labeled_data():
    """
    Load labeled data in CoNLL format into a Hugging Face Dataset.
    """
    sentences = []
    labels = []

    for file in os.listdir(LABELED_DATA_DIR):
        filepath = os.path.join(LABELED_DATA_DIR, file)
        with open(filepath, "r", encoding="utf-8") as f:
            sentence = []
            sentence_labels = []
            for line in f:
                if line.strip():
                    token, tag = line.strip().split()
                    sentence.append(token)
                    sentence_labels.append(LABEL_MAP[tag])
                else:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
    return DatasetDict({"train": {"tokens": sentences, "ner_tags": labels}})

def tokenize_and_align_labels(dataset, tokenizer):
    """
    Tokenize and align labels with Hugging Face tokenizers.
    """
    def align_labels(batch):
        tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(batch["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dataset.map(align_labels, batched=True)

def train_and_evaluate_model(model_name, dataset):
    """
    Fine-tune and evaluate a model on the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUTPUT_DIR, model_name),
        evaluation_strategy="epoch",
        logging_dir="./logs",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    trainer.train()

    # Evaluate the model
    metric = load_metric("seqeval")
    predictions, labels, _ = trainer.predict(tokenized_dataset["train"])
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    true_labels = [[LABEL_MAP[label] for label in sentence] for sentence in labels]
    pred_labels = [[LABEL_MAP[label] for label in sentence] for sentence in predictions]

    results = metric.compute(predictions=pred_labels, references=true_labels)
    return results

def compare_models():
    """
    Compare multiple models and save their evaluation results.
    """
    dataset = load_labeled_data()

    results = {}
    for model_name, model_label in MODELS_TO_COMPARE.items():
        print(f"\nTraining and evaluating model: {model_label}")
        results[model_label] = train_and_evaluate_model(model_name, dataset)

    # Save comparison results
    comparison_file = os.path.join(MODEL_OUTPUT_DIR, "model_comparison_results.txt")
    with open(comparison_file, "w") as f:
        for model, metrics in results.items():
            f.write(f"Model: {model}\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write("\n")
    print(f"Model comparison results saved at {comparison_file}")

if __name__ == "__main__":
    compare_models()
