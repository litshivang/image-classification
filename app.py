from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from utils import preprocess_image

app = Flask(__name__)

@app.route("/")
def welcome():
    return "Welcome to Image Classification API!"

# Load the trained model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model_path = 'trained_model.pth'
model = load_model(model_path)

# Define image transformations
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/classify_selfie", methods=["POST"])
def classify_selfie_api():
    try:
        # Get image file from request
        image_file = request.files["image"]
        # Preprocess image
        image = preprocess_image(image_file)
        # Perform classification
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)
        # Convert predicted class index to label
        class_labels = ['inside_office', 'outside_office']
        predicted_label = class_labels[predicted_class.item()]
        # Send back the classification result
        return jsonify({"result": predicted_label, "probability": probabilities[0][predicted_class.item()].item()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False)
