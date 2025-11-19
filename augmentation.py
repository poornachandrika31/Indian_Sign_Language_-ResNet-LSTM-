import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    return augmented['image']

# Example usage:
gestures = ["house", "love","please","sleep","understand"]
for gesture in gestures:
    input_folder = f"processed/isl/{gesture}"
    output_folder = f"augmented/{gesture}"
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        in_path = os.path.join(input_folder, filename)
        augmented_image = augment_image(in_path)
        if augmented_image is not None:
            augmented_np = augmented_image.permute(1, 2, 0).cpu().numpy()
            augmented_np = (augmented_np * 255).astype(np.uint8)
            augmented_np = cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, augmented_np)
            print(f"Augmented and saved {gesture}/{filename}")
