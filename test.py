import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import mediapipe as mp


# Define your ISLClassifier model (same as before)
class ISLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ISLClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        return self.model(x)
# Load trained weights
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ISLClassifier(num_classes=num_classes)
model.load_state_dict(torch.load('best_model_weights.pth', map_location=device))
model.eval()
model.to(device)


# Class index to label mapping
idx_to_class = {0: 'house', 1: 'love', 2: 'please',3:'sleep',4:'understand'}  # only 2 classes
# Image transformations
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)


# Open webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Flip frame horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)


    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Process frame for hand landmarks
    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on original frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            # Get bounding box of hand for cropping
            h, w, c = frame.shape
            all_x = []
            all_y = []
            for hand_landmarks in results.multi_hand_landmarks:
                all_x += [lm.x for lm in hand_landmarks.landmark]
                all_y += [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(all_x) * w) - 20
            x_max = int(max(all_x) * w) + 20
            y_min = int(min(all_y) * h) - 20
            y_max = int(max(all_y) * h) + 20


            # Crop hand region from frame
            hand_img = frame[y_min:y_max, x_min:x_max]


            if hand_img.size == 0:
                continue  # skip empty crops


            # Convert to PIL Image
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(hand_img)


            # Apply transformations
            input_tensor = transform(pil_img).unsqueeze(0).to(device)


            # Predict ISL class
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = idx_to_class[predicted.item()]


            # Display label near the hand bounding box
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    else:
        # If no hand detected, notify
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)


    cv2.imshow("ISL Hand Gesture Prediction", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


hands.close()
cap.release()
cv2.destroyAllWindows()