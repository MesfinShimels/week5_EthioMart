from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

def train_ner_model(dataset_path, output_dir):
    """Fine-tunes a pretrained model for NER."""
    # Load dataset
    dataset = load_dataset("conll2003", data_files={"train": dataset_path})
    
    # Load model and tokenizer
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5)
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )
    
    trainer.train()

if __name__ == "__main__":
    DATASET_PATH = "data/labeled/your_dataset.conll"
    OUTPUT_DIR = "models/fine_tuned/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_ner_model(DATASET_PATH, OUTPUT_DIR)
