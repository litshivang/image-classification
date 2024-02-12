import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to (224, 224)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Set dataset directory
data_dir = 'dataset'
inside_office_dir = os.path.join(data_dir, 'inside_office')
outside_office_dir = os.path.join(data_dir, 'outside_office')

# Load dataset
all_data = datasets.ImageFolder(data_dir, data_transforms['train'])

# Split dataset into training and validation sets
train_size = int(0.8 * len(all_data))
val_size = len(all_data) - train_size
train_data, val_data = random_split(all_data, [train_size, val_size])

# Define dataloaders
dataloaders = {
    'train': DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
}

# Define the model architecture
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Output: inside_office, outside_office

# Set device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct / total
            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    return model

if __name__ == '__main__':
    # Train the model
    trained_model = train_model(model, criterion, optimizer, num_epochs=10)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
