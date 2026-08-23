import cv2
from face_engine import FaceEngine
from face_engine.exceptions import FaceNotFoundError

engine = FaceEngine()

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces, extra = engine.find_faces(rgb_frame)

        for face in faces:
            face = face.astype(int)

            x1, y1, x2, y2 = face

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    except FaceNotFoundError:
        pass

    cv2.imshow("FaceVault - Face Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()