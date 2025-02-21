import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
from PIL import Image
import random

#  Class label translation (Italian -> English)
translate = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

#  Target class distribution (balanced dataset)
target_class_counts = {cls: 3000 for cls in translate.values()}

#  Image Transforms
image_size = (224, 224)

preprocessing_transforms = Compose([
    Resize(image_size),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

augmentation_transforms = Compose([
    RandomRotation(15),
    RandomHorizontalFlip(),
    RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, translate, augment=False):
        self.data = []
        self.labels = []
        self.augment = augment
        self.class_counts = {}
        self.image_paths = {cls: [] for cls in target_class_counts}
        self.class_to_idx = {cls: idx for idx,
                             cls in enumerate(target_class_counts.keys())}

        # Load images & labels
        for folder in os.listdir(dataset_path):
            class_name = translate.get(folder, folder)
            class_folder = os.path.join(dataset_path, folder)

            if os.path.isdir(class_folder):
                images = os.listdir(class_folder)
                self.class_counts[class_name] = len(images)

                for img in images:
                    img_path = os.path.join(class_folder, img)
                    self.image_paths[class_name].append(img_path)

        # Balance dataset
        self.balance_classes()

    def balance_classes(self):
        """Balances dataset using oversampling & undersampling."""
        balanced_data = []

        for label, img_paths in self.image_paths.items():
            current_count = len(img_paths)
            target_count = target_class_counts[label]

            if current_count < target_count:
                extra_needed = target_count - current_count
                augmented_images = random.choices(img_paths, k=extra_needed)
                for img_path in augmented_images:
                    balanced_data.append(
                        (img_path, label, True))  # Augmented=True
            elif current_count > target_count:
                img_paths = random.sample(img_paths, target_count)

            for img_path in img_paths:
                balanced_data.append(
                    (img_path, label, False))  # Augmented=False

        self.image_paths = balanced_data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label, augmented = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        transform = augmentation_transforms if augmented else preprocessing_transforms
        image = transform(image)
        return image, self.class_to_idx[label]


def train_image_classifier(args):
    """
    Train the EfficientNetV2 model for animal classification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    animal_dataset = AnimalDataset(args.dataset_path, translate, augment=True)

    # Split into Training (80%), Validation (10%), and Test (10%)
    train_size = int(0.8 * len(animal_dataset))
    val_size = int(0.1 * len(animal_dataset))
    test_size = len(animal_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        animal_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load EfficientNetV2 model
    model = models.efficientnet_v2_s(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(target_class_counts))
    )
    model.to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Train Model
    print("🚀 Training Model on:", device)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        args.output_dir, "efficientnetv2_animal.pth"))
    print(f"✅ Model saved in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EfficientNetV2 model for animal classification.")

    # Add arguments
    parser.add_argument("--dataset_path", type=str,
                        required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str,
                        default="./image_model", help="Path to save trained model")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")

    args = parser.parse_args()

    # Train the model
    train_image_classifier(args)
