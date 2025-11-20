**Indian Sign Language Recognition (ISL) System**


📌 **Project Overview**

This project presents a comprehensive Indian Sign Language (ISL) recognition system, encompassing both static image classification and dynamic gesture recognition from video sequences. The system aims to facilitate real-time, accurate interpretation of ISL signs to support communication accessibility. Utilizing state-of-the-art computer vision and deep learning frameworks, this project bridges the gap between raw visual data and semantic gesture understanding.

📌 **Dataset Description**
1. Static Gesture Dataset
The static gesture dataset consists of images representing isolated Indian Sign Language (ISL) signs such as "house," "love," "please," "sleep," and "understand." These images were collected from varied participants under different lighting and backgrounds to ensure diversity. Each image is resized to 224×224 pixels and normalized to prepare it for convolutional neural network input. To improve robustness and generalization during training, a set of augmentations—including horizontal flips, brightness and contrast adjustments, rotations up to ±15 degrees, and small translations and scaling—were applied using the Albumentations library. The dataset is organized in folders named by gesture class, facilitating straightforward loading and label mapping.

2. Dynamic Gesture Dataset
The dynamic dataset captures continuous ISL gestures such as "yawn," "man," and "woman" in video format. Gesture videos were recorded using a webcam with controlled start/stop recording commands, saved in class-specific directories. Each video frame was processed using the MediaPipe Holistic model to extract 3D pose and hand landmarks, converting videos into time-series landmark data. These sequences were standardized to fixed lengths of 30 frames by padding or truncating to allow uniform model input. The resulting .npy landmark sequences represent detailed temporal dynamics of gestures and serve as input for training an LSTM-based deep learning model. This structured and balanced dataset enables effective learning of temporal patterns in sign language gestures.

📌 **Methodology**

This project explores Indian Sign Language (ISL) gesture recognition by gradually evolving from a static image classification pipeline to a dynamic sequence-based video recognition model. The main goal was to understand how spatial information alone compares to a model that also captures temporal motion, which is essential for realistic sign interpretation.
The methodology is divided into two stages:

a) Static image-based gesture recognition (ResNet18)

b) Dynamic video-based gesture recognition (CNN + LSTM)

📍 1. Static Gesture Recognition (ResNet18)

The first phase focused on recognizing ISL gestures using single images, where each gesture was treated as a standalone frame.
A ResNet18 CNN was trained on a dataset of isolated hand signs. The model extracts spatial features like shape, orientation, and finger patterns, making it effective for gestures that do not require movement.

This stage helped establish a baseline accuracy and provided insights into how well spatial-only information performs in controlled, static scenarios.

📍 2. Dynamic Gesture Recognition (CNN + LSTM)

To handle gestures involving motion, trajectory, and sequence progression, the project moved toward a video-based approach. Each video clip was converted into a series of frames, and a CNN backbone (ResNet) was used to extract per-frame spatial features. These extracted features were then passed to an LSTM layer, which learns temporal dependencies across frames.
This enables the model to understand:

Direction of movement

Speed of gesture

Transitions between hand shapes

Continuous motion patterns

The CNN+LSTM hybrid architecture bridges spatial and temporal modeling, making it suitable for real-world continuous signing.

📌 **TECH STACK:**

Languages: Python

Deep Learning Frameworks: TensorFlow/Keras (dynamic model), PyTorch (static model)

Computer Vision: OpenCV, MediaPipe Holistic

Data Augmentation: Albumentations

Model Architectures: ResNet18 for static gestures, LSTM for dynamic gestures

📌 **Why the Shift from ResNet to LSTM-based Model?**

While the static ResNet model worked well for gestures recognizable from a single frame, many ISL signs cannot be captured through static information alone. They rely heavily on:
The motion path of the hand
How the hand moves or transitions over time
Multi-frame dependencies
Dynamic cues like acceleration or rhythmic patterns

ResNet processes frames independently and therefore fails to retain context between frames.
This limitation motivated the move to a temporal model, where LSTMs maintain memory across time steps, enabling the system to interpret gestures as continuous sequences instead of isolated moments.

The transition from static images to a video dataset was a natural progression to better capture the complexity of dynamic sign language gestures.

📌 **Results**
1. Static gesture model achieved strong accuracy on a vocabulary of 5 isolated ISL signs using ResNet18 with extensive augmentation.
2. Dynamic LSTM model captured temporal information, enabling recognition of continuous gestures with effective confidence-based inference.
3. Real-time system operates at interactive frame rates, incorporating fallback detections and clean UI overlays.

🏁 **Conclusion**

By evolving from a static ResNet18 classifier to a dynamic CNN+LSTM model, the project demonstrates how incorporating temporal information drastically improves performance on real-world sign language gestures. The final hybrid architecture captures both spatial details and temporal flow, making it significantly more robust, accurate, and aligned with how ISL gestures are truly performed.
