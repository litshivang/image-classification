from torchvision import transforms
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
