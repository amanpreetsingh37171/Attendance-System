import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
facenet = FaceNet()

attendance_mean_embedding = None

def capture_attendance_face_streamlit(max_faces=5):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:  # 🔁 Continuous loop for multiple persons

            captured_embeddings = []
            captured_keypoints = []
            frame_count = 0
            mean_embedding = None

            while frame_count < max_faces:

                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_frame)

                if len(faces) == 0:
                    yield rgb_frame, None, None, None
                    continue

                # Take first detected face
                face = faces[0]
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)

                face_crop = rgb_frame[y:y+h, x:x+w]

                if face_crop.size != 0:
                    embedding = facenet.embeddings([face_crop])[0]
                    embedding = np.array(embedding, dtype=np.float32)

                    captured_embeddings.append(embedding)
                    captured_keypoints.append(face['keypoints'])

                    frame_count += 1

                    # 🔹 Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # 🔹 Draw keypoints (eyes, nose, mouth)
                    for name, point in face['keypoints'].items():
                        px, py = point
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None, None

            # After capturing 5 frames
            if len(captured_embeddings) == max_faces:
                mean_embedding = np.mean(captured_embeddings, axis=0)

                # 🔹 Store globally
                global attendance_mean_embedding
                attendance_mean_embedding = mean_embedding

                yield None, captured_embeddings, captured_keypoints, mean_embedding

            # 🔁 Automatically resets for next person

    finally:
        cap.release()
