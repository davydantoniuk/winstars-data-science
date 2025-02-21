import argparse
import warnings
from ner.inference_ner import predict_animals_ner
from img_classification.inference_image import predict_animal_image

# Ignore warnings
warnings.filterwarnings("ignore")


def compare_animals(text, ner_model_dir, image_path, image_model_dir):
    """
    Compare animals detected in text using NER model with the animal predicted in the image.

    Args:
        text (str): Input text for animal NER detection.
        ner_model_dir (str): Path to the trained NER model directory.
        image_path (str): Path to the image to be classified.
        image_model_dir (str): Path to the trained image classification model.

    Returns:
        bool: True if the predicted animal is in the detected animals, False otherwise.
    """
    detected_animals = predict_animals_ner(text, ner_model_dir)
    predicted_animal = predict_animal_image(image_path, image_model_dir)
    return predicted_animal in detected_animals


def main():
    parser = argparse.ArgumentParser(
        description="Final pipeline to check if the text description of the animal matches the image."
    )
    parser.add_argument(
        "--ner_model_dir", type=str, default="../models/ner_model",
        help="Path to the trained NER model directory."
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Input text for animal NER detection."
    )
    parser.add_argument(
        "--image_model_dir", type=str, default="../models/image_model/efficientnetv2_animal.pth",
        help="Path to the trained image classification model (.pth)."
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to the image to be classified."
    )
    args = parser.parse_args()

    is_match = compare_animals(
        args.text, args.ner_model_dir, args.image_path, args.image_model_dir
    )

    print(is_match)


if __name__ == "__main__":
    main()
