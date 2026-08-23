import numpy as np
from database import add_user

# Load the registered face embedding
face_embedding = np.load("registered_face.npy")

# Add the user to the database
add_user("Poorva", face_embedding)