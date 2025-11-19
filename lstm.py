from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

SEQUENCE_PATH = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\preprocessed"
actions = ["yawn", "man", "woman"]  # Your gesture classes, update as needed

# Check if the data folder and class folders exist
for action in actions:
    class_path = os.path.join(SEQUENCE_PATH, action)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Folder for action '{action}' not found at {class_path}")

X, y = [], []
label_map = {label:num for num, label in enumerate(actions)}

for action in actions:
    class_folder = os.path.join(SEQUENCE_PATH, action)
    for file in os.listdir(class_folder):
        if file.endswith('.npy'):  # Only consider .npy files
            sequence = np.load(os.path.join(class_folder, file))
            if sequence.shape[0] == 30:
                X.append(sequence)
                y.append(label_map[action])

X = np.array(X)
y = to_categorical(y).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ---- Model ----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
