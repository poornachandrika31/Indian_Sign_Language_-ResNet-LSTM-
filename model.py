import torch
import torch.nn as nn
import torchvision.models as models

class ISLClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ISLClassifier, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Example usage:
num_classes = 5  # For single word classification, set to total number of words in your dataset
model = ISLClassifier(num_classes)
print(model)
