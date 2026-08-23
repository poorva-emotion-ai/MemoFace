import cv2
import numpy as np
from face_engine import FaceEngine

engine = FaceEngine()

camera = cv2.VideoCapture(0)

print("Press R to register your face.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("FaceVault - Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, extra = engine.find_faces(rgb_frame)

        if len(faces) == 0:
            print("No face detected. Try again.")
            continue

        kpss = extra["kpss"]

        embeddings = engine.compute_embeddings(
            rgb_frame,
            faces,
            kpss=kpss
        )

        face_embedding = embeddings[0]

        np.save("registered_face.npy", face_embedding)

        print("Face registered successfully! ")
        print("Embedding saved.")

        break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()