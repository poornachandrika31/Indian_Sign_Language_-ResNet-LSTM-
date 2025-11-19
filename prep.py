from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

SEQUENCE_PATH = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\preprocessed"
actions = ["yawn", "man", "woman"]

X, y = [], []
label_map = {label:num for num, label in enumerate(actions)}

for action in actions:
    action_path = os.path.join(SEQUENCE_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
    for file in files:
        sequence = np.load(os.path.join(action_path, file))
        # pad if sequence <30 frames
        if sequence.shape[0] < 30:
            padding = np.zeros((30 - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack((sequence, padding))
        else:
            sequence = sequence[:30]
        X.append(sequence)
        y.append(label_map[action])

X = np.array(X)
y = to_categorical(y).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
