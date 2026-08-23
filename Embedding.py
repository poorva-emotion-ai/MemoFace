import cv2
from face_engine import FaceEngine

engine = FaceEngine()

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces, extra = engine.find_faces(rgb_frame)

    if len(faces) > 0:
        kpss = extra["kpss"]

        embeddings = engine.compute_embeddings(
            rgb_frame,
            faces,
            kpss=kpss
        )

        print("Face detected!")
        print("Embedding generated!")
        print("Embedding shape:", embeddings.shape)

        break

camera.release()
cv2.destroyAllWindows()