from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ISLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'house': 0, 'love': 1, 'please':2, 'sleep':3, 'understand':4}
  # Extend this dictionary for more classes

        # Collect images and labels
        for label_name in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label_idx = self.class_to_idx.get(label_name, -1)
            for img_file in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_file))
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_dataset = ISLDataset(root_dir='augmented', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate example
for images, labels in train_loader:
    print(images.shape)  # torch.Size([32, 3, 224, 224])
    print(labels)
    break
