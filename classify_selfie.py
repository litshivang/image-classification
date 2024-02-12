import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define image transformations
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = data_transform(image)
    return image

# Load the trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Classify selfie
def classify_selfie(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

    predicted_class_index = predicted_class.item()
    probability = probabilities[0][predicted_class_index].item()
    class_labels = ['inside_office', 'outside_office']
    predicted_label = class_labels[predicted_class_index]

    return predicted_label, probability

if __name__ == "__main__":
    # Load the trained model
    model_path = 'trained_model.pth'
    model = load_model(model_path)

    # Path to the folder containing selfie images
    folder_path = 'selfie'

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            predicted_label, probability = classify_selfie(model, image_path)
            print(f"Image: {filename}, Predicted label: {predicted_label}, Probability: {probability}")

