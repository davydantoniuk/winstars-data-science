import torch
import argparse
from transformers import BertTokenizerFast, BertForTokenClassification


def predict_animals_ner(text, model_dir):
    """
    Function for NER prediction. Returns only detected class names in lowercase.
    """
    # Define label mapping (should match training labels)
    classes = ["DOG", "SPIDER", "CHICKEN", "HORSE", "BUTTERFLY",
               "COW", "SQUIRREL", "SHEEP", "CAT", "ELEPHANT"]
    label_list = ["O"] + [f"I-{cls}" for cls in classes]

    # Load trained model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Tokenize input (DO NOT USE .split() to avoid breaking multi-word entities)
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False
    )

    # Move to correct device
    tokens = {key: value.to(device) for key, value in tokens.items()}

    # Run model inference
    with torch.no_grad():
        outputs = model(**tokens).logits  # Get model predictions

    # Convert logits to predicted class indices
    predictions = torch.argmax(outputs, dim=2).cpu().numpy()[0]

    # Get tokenized words and align them correctly
    word_ids = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    detected_classes = set()  # Use a set to avoid duplicates

    for token, pred_label in zip(word_ids, predictions):
        label = label_list[pred_label]

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Extract class name and convert to lowercase
        if label.startswith("I-"):
            entity_class = label.split("-")[1].lower()  # Convert to lowercase
            detected_classes.add(entity_class)

    return detected_classes  # Convert set to list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NER Model Inference for Animal Detection"
    )
    parser.add_argument(
        "--model_dir", type=str, default="../../models/ner_model", help="Path to trained NER model directory"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Input text to predict animal entities"
    )
    args = parser.parse_args()

    detected_classes = predict_animals_ner(args.text, args.model_dir)

    # Output results
    if detected_classes:
        print(f"🐾 Detected Animal Class: {', '.join(detected_classes)}")
    else:
        print("🚫 No animals detected in the sentence.")
