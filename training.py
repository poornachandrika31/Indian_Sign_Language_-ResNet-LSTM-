import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os

# Dataset class using your folder structure
class ISLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Map folder names to indices
        self.class_to_idx = {'house':0, 'love':1,'please':2, 'sleep':3,'understand':4}

        # Collect image paths and labels
        for label_name, label_idx in self.class_to_idx.items():
            label_folder = os.path.join(root_dir, label_name)
            if not os.path.exists(label_folder):
                print(f"Warning: {label_folder} does not exist.")
                continue
            for filename in os.listdir(label_folder):
                filepath = os.path.join(label_folder, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(filepath)
                    self.labels.append(label_idx)

        print(f"Found {len(self.image_paths)} images across classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for images
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

# Instantiate dataset and dataloader
dataset_path = r'C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\processed\isl'
train_dataset = ISLDataset(root_dir=dataset_path, transform=transform)

if len(train_dataset) == 0:
    raise RuntimeError("Dataset is empty. Please check your file paths and dataset structure.")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class ISLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ISLClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

num_classes = 5  # house, sleep, understand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ISLClassifier(num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, loader, criterion, optimizer, epochs=40):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, epochs=40)
