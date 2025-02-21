import torch
import argparse
import os
from torchvision import models, transforms
from PIL import Image


def predict_animal_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Image Transformations (Same as Training)
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Class label mapping (Ensure it's the same as used in training)
    class_labels = [
        "dog", "horse", "elephant", "butterfly",
        "chicken", "cat", "cow", "sheep", "spider", "squirrel"
    ]

    # Load Trained Model
    model = models.efficientnet_v2_s(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, len(class_labels))  # 10 classes
    )

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return class_labels[predicted_class]


if __name__ == "__main__":
    # Argument parser for image classification
    parser = argparse.ArgumentParser(
        description="Inference for Animal Image Classification"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="../../models/image_model/efficientnetv2_animal.pth",
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the input image for classification"
    )

    args = parser.parse_args()

    # Run Inference
    predicted_animal = predict_animal_image(args.image_path, args.model_path)

    # Output result
    print(f"Predicted Animal: {predicted_animal}")
