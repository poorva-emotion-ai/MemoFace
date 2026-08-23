import cv2
from face_engine import FaceEngine
from database import get_users_with_embeddings

engine = FaceEngine()

camera = cv2.VideoCapture(0)

print("Look at the camera.")
print("Press V to verify your identity.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("FaceVault - Login", frame)

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

        users = get_users_with_embeddings()

        best_score = -1
        best_user = None

        for user_id, name, registered_embedding in users:

            score = engine.compare(
                registered_embedding,
                current_face
            )

            print(f"{name} similarity score: {score}")

            if score > best_score:
                best_score = score
                best_user = (user_id, name)

        if best_user is not None and best_score >= 0.5:

            user_id, name = best_user

            print(" Face matched!")
            print("Welcome,", name)
            print("User ID:", user_id)
            print("Similarity score:", best_score)

        else:

            print("Face not recognized.")
            print("Access denied.")

        break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()