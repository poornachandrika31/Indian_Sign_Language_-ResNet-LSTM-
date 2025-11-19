import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\isl_model2.h5")
actions = ["yawn", "man", "woman"]  # update as per your classes

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def hands_present(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_fancy_box(image, text, color=(0, 255, 0), pos=(10, 30)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x, y = pos
    cv2.rectangle(image, (x-10, y-40), (x + w + 10, y + 10), color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2, cv2.LINE_AA)

sequence = []
threshold = 0.5  # Lowered threshold for debugging and better sensitivity
predicted_action = ""
hand_detected = False

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Draw landmarks for hands only
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        hand_detected = hands_present(results)

        if len(sequence) == 30 and hand_detected:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print("Predicted probabilities:", res)  # Debug: Show scores for all classes
            action_idx = np.argmax(res)
            confidence = res[action_idx]
            if confidence > threshold:
                predicted_action = f"{actions[action_idx]} ({confidence:.2f})"
                draw_fancy_box(image, predicted_action, color=(0,130,0), pos=(10,60))
            else:
                predicted_action = ""
                draw_fancy_box(image, "No gesture detected", color=(150,0,200), pos=(10,60))
        elif not hand_detected:
            draw_fancy_box(image, "No hand detected", color=(0,0,255), pos=(10,60))

        cv2.imshow("ISL Real-time Recognition", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
