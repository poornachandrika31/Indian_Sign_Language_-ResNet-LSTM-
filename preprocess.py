import os
import cv2
import numpy as np

# Image resize dimensions
resize_dim = (224, 224)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize_dim)
    image = image.astype(np.float32) / 255.0
    return image

def save_processed_image(image, save_path):
    image_to_save = (image * 255).astype(np.uint8)
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_to_save)

def preprocess_and_save_all(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename)
        processed_image = preprocess_image(in_path)
        if processed_image is not None:
            save_processed_image(processed_image, out_path)
            print(f"Processed and saved {filename}")

if __name__ == "__main__":
    for gesture in ["house","love","please","sleep","understand"]:
        input_folder = f'raw/isl/{gesture}'
        output_folder = f'processed/isl/{gesture}'
        os.makedirs(output_folder, exist_ok=True)
        preprocess_and_save_all(input_folder, output_folder)
