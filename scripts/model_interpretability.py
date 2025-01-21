
import shap
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Paths
MODEL_DIR = os.path.join("models", "fine_tuned", "xlm-roberta-base")

def interpret_model():
    """
    Interpret fine-tuned NER model using SHAP.
    """
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

    # Example sentence
    sentence = "አዲስ አበባ ዋጋ 1000 ብር"
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    # SHAP interpretation
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(inputs)

    # Visualize explanation
    shap.plots.text(shap_values)

if __name__ == "__main__":
    interpret_model()
