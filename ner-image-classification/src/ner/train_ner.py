import argparse
import json
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_dataset(train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label_mappings():
    classes = ["DOG", "SPIDER", "CHICKEN", "HORSE", "BUTTERFLY",
               "COW", "SQUIRREL", "SHEEP", "CAT", "ELEPHANT"]
    label_list = ["O"] + [f"I-{cls}" for cls in classes]

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    return label_list, label2id, id2label


def tokenize_and_align_labels(example, tokenizer, label2id):
    tokenized_input = tokenizer(
        example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128
    )

    word_ids = tokenized_input.word_ids()  # Map tokens to words
    labels = example["tags"]
    aligned_labels = []

    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:  # Special tokens ([CLS], [SEP]) get label "O"
            aligned_labels.append(label2id["O"])
        elif word_id != prev_word_id:  # First token of a word
            aligned_labels.append(label2id[labels[word_id]])
        else:  # Subword tokens continue the entity
            aligned_labels.append(
                label2id[labels[word_id]] if labels[word_id] != "O" else label2id["O"])
        prev_word_id = word_id

    tokenized_input["labels"] = aligned_labels
    return tokenized_input


def prepare_datasets(dataset, tokenizer, label2id):
    """
    Convert dataset into Hugging Face format and split into training/validation sets.
    """
    hf_dataset = Dataset.from_list(
        [tokenize_and_align_labels(entry, tokenizer, label2id)
         for entry in dataset]
    )

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(hf_dataset))
    train_dataset = hf_dataset.select(range(train_size))
    eval_dataset = hf_dataset.select(range(train_size, len(hf_dataset)))

    return train_dataset, eval_dataset


def train_ner_model(args):
    """
    Main function to train the NER model.
    """
    # Set seed
    set_seed(args.seed)

    # Load dataset
    dataset = load_dataset(args.train_file)

    # Get label mappings
    label_list, label2id, id2label = get_label_mappings()

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        dataset, tokenizer, label2id)

    # Load BERT-based model
    model = BertForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train model
    trainer.train()

    # Save trained model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Model saved at {args.output_dir}")


if __name__ == "__main__":
    # Argument parser for parameterized training
    parser = argparse.ArgumentParser(
        description="Train a BERT-based NER model for animal entity recognition.")

    # Add parameters
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to training dataset JSON file")
    parser.add_argument("--model_name", type=str,
                        default="dslim/bert-base-NER", help="Pretrained BERT model to use")
    parser.add_argument("--output_dir", type=str, default="./ner_model",
                        help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Run training
    train_ner_model(args)
