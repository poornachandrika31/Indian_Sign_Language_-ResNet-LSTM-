import cv2
import os

def capture_videos_to_class_folder(label="focus"):
    base_dir = r"C:\Users\VSR BALASUBRAHMANYAM\Desktop\data\raw\isl_videos"
    save_dir = os.path.join(base_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    # Ensure video counts don't overwrite (scan folder for .avi's)
    existing_videos = [f for f in os.listdir(save_dir) if f.endswith('.avi')]
    count = len(existing_videos)

    cap = cv2.VideoCapture(0)
    recording = False
    out = None

    print(f"Ready to record videos to {save_dir}")
    print("Press 'r' to start recording, 's' to stop and save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('Gesture Video Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            recording = True
            video_path = os.path.join(save_dir, f"gesture_{count}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            print(f"Started recording: {video_path}")

        elif key == ord('s') and recording:
            recording = False
            out.release()
            out = None
            print("Stopped recording and saved video.")
            count += 1

        elif key == ord('q'):
            print("Quitting capture.")
            break

        if recording:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can pass any class/label here, or prompt the user for it if needed
    capture_videos_to_class_folder("focus")
