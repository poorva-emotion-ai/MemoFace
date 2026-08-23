import cv2
import numpy as np
from face_engine import FaceEngine

engine = FaceEngine()

# Load the registered face embedding
registered_face = np.load("registered_face.npy")

camera = cv2.VideoCapture(0)

print("Look at the camera...")
print("Press V to verify.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("FaceVault - Verification", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("v"):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, extra = engine.find_faces(rgb_frame)

        if len(faces) == 0:
            print("No face detected.")
            continue

        kpss = extra["kpss"]

        embeddings = engine.compute_embeddings(
            rgb_frame,
            faces,
            kpss=kpss
        )

        current_face = embeddings[0]

        score = engine.compare(
            registered_face,
            current_face
        )

        print("Similarity score:", score)

        if score >= 0.5:
            print(" Face matched! Access granted.")
        else:
            print(" Face not matched! Access denied.")

        break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()