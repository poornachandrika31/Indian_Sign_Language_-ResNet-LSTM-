import os
import cv2
import mediapipe as mp
import numpy as np

# Paths
DATA_PATH = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\raw\isl_videos"
SEQUENCE_PATH = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\preprocessed"

actions = ["yawn", "man", "woman"]

# Create folders
for action in actions:
    os.makedirs(os.path.join(SEQUENCE_PATH, action), exist_ok=True)

# Mediapipe setup
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Process videos
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        input_folder = os.path.join(DATA_PATH, action)
        output_folder = os.path.join(SEQUENCE_PATH, action)

        for video_file in os.listdir(input_folder):
            cap = cv2.VideoCapture(os.path.join(input_folder, video_file))
            sequence = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            cap.release()

            if len(sequence) >= 30:
                sequence = np.array(sequence[:30])  # take first 30 frames
                filename = video_file.replace(".mp4", ".npy").replace(".avi", ".npy")
                np.save(os.path.join(output_folder, filename), sequence)
