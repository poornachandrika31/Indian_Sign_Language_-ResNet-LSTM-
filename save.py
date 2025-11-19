import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
class ISLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ISLClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

num_classes = 5 # house, sleep, understand
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ISLClassifier(num_classes=num_classes)
torch.save(model.state_dict(), 'best_model_weights.pth')
print("Model weights saved to best_model_weights.pth")
