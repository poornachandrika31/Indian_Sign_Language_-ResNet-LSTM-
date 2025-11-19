# training3.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os


# -----------------------------
# Load preprocessed sequences
# -----------------------------
SEQUENCE_PATH = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\preprocessed"
actions = ["yawn", "man", "woman"]


X, y = [], []
label_map = {label:num for num, label in enumerate(actions)}


for action in actions:
    action_path = os.path.join(SEQUENCE_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith(".npy")]
    for file in files:
        sequence = np.load(os.path.join(action_path, file))
        # pad/truncate to 30 frames
        if sequence.shape[0] < 30:
            padding = np.zeros((30 - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack((sequence, padding))
        else:
            sequence = sequence[:30]
        X.append(sequence)
        y.append(label_map[action])


X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# One-hot encode and split
# -----------------------------
y = to_categorical(y, num_classes=len(actions))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# -----------------------------
# Build LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X_train.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(actions), activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# -----------------------------
# Train model (for testing)
# -----------------------------
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))


# -----------------------------
# Save model
# -----------------------------
model.save(r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\isl_model2.h5")
print("Model saved as isl_model2.h5")