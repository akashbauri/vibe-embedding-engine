import cv2
import os

def extract_frames(video_path, output_folder="frames", interval_sec=2):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)

    frames = []
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            filename = f"{output_folder}/frame_{frame_id}.jpg"
            cv2.imwrite(filename, frame)
            frames.append(filename)
            frame_id += 1

        count += 1

    cap.release()
    return frames
