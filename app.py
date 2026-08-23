import streamlit as st
import numpy as np
from PIL import Image

from face_engine import FaceEngine
from database import (
    add_user,
    get_users_with_embeddings,
    add_note,
    get_user_notes,
    delete_note,
    update_note
)


# FaceEngine instance

engine = FaceEngine()

# SESSION STATE

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "editing_note_id" not in st.session_state:
    st.session_state.editing_note_id = None

if "editing_title" not in st.session_state:
    st.session_state.editing_title = ""

if "editing_content" not in st.session_state:
    st.session_state.editing_content = ""

# PAGE CONFIGURATION

st.set_page_config(
    page_title="FaceVault",
    page_icon="🔐",
    layout="centered"
)

# Header

st.title("🔐 FaceVault")

st.write(
    "Your personal face-authenticated notes vault."
)

st.divider()

# NAVIGATION

option = st.radio(
    "Choose an option:",
    ["Register", "Login"]
)

# REGISTER

if option == "Register":

    st.header("👤 Register")

    name = st.text_input(
        "Enter your name"
    )

    photo = st.camera_input(
        "Take a photo of your face"
    )

    if photo is not None:

        image = Image.open(photo)

        st.image(
            image,
            caption="Captured image",
            width=300
        )

        image_array = np.array(image)

        # Find face
        faces, extra = engine.find_faces(
            image_array
        )

        if len(faces) == 0:

            st.error(
                "No face detected. Please try again."
            )

        else:

            st.success(
                " Face detected!"
            )

            st.write(
                f"Number of faces detected: {len(faces)}"
            )

            # Get facial key points
            kpss = extra["kpss"]

            # Create embedding
            embeddings = engine.compute_embeddings(
                image_array,
                faces,
                kpss=kpss
            )

            face_embedding = embeddings[0]

            st.success(
                "Face embedding created successfully!"
            )

            # Save user
            if name:

                add_user(
                    name,
                    face_embedding
                )

                st.success(
                    f" {name} registered successfully!"
                )

            else:

                st.warning(
                    "Please enter your name to register."
                )

# LOGIN

else:

    # USER IS ALREADY LOGGED IN

    if st.session_state.logged_in:

        st.header(
            f"🔓 Welcome, {st.session_state.user_name}!"
        )

        st.write(
            "This is your private FaceVault."
        )

        # LOGOUT

        if st.button("🔒 Logout"):

            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = None

            st.session_state.editing_note_id = None
            st.session_state.editing_title = ""
            st.session_state.editing_content = ""

            st.rerun()


        st.divider()

# EDIT NOTE form 

        if st.session_state.editing_note_id is not None:

            st.subheader("✏️ Edit Note")

            edit_title = st.text_input(
                "Title",
                value=st.session_state.editing_title
            )

            edit_content = st.text_area(
                "Content",
                value=st.session_state.editing_content
            )

        # SAVE CHANGES

            if st.button("💾 Save Changes"):

                if edit_title and edit_content:

                    updated = update_note(
                        st.session_state.editing_note_id,
                        st.session_state.user_id,
                        edit_title,
                        edit_content
                    )

                    if updated:

                        st.success(
                            "✅ Note updated successfully!"
                        )

                        st.session_state.editing_note_id = None
                        st.session_state.editing_title = ""
                        st.session_state.editing_content = ""

                        st.rerun()

                    else:

                        st.error(
                            "❌ Could not update the note."
                        )

                else:

                    st.warning(
                        "Please enter both title and content."
                    )


            
            # CANCEL EDIT

            if st.button("❌ Cancel Edit"):

                st.session_state.editing_note_id = None
                st.session_state.editing_title = ""
                st.session_state.editing_content = ""

                st.rerun()


        st.divider()

        # ADD NOTE

        st.subheader("➕ Add a Note")

        note_title = st.text_input(
            "Note title",
            key="new_note_title"
        )

        note_content = st.text_area(
            "Note content",
            key="new_note_content"
        )


        if st.button("💾 Save Note"):

            if note_title and note_content:

                add_note(
                    st.session_state.user_id,
                    note_title,
                    note_content
                )

                st.success(
                    "✅ Note saved successfully!"
                )

                st.rerun()

            else:

                st.warning(
                    "Please enter both a title and content."
                )


        st.divider()

        # DISPLAY NOTES

        st.subheader("📝 My Notes")

        notes = get_user_notes(
            st.session_state.user_id
        )


        if len(notes) == 0:

            st.info(
                "You don't have any notes yet."
            )


        else:

            for (
                note_id,
                title,
                content,
                created_at
            ) in notes:

                st.markdown(
                    f"### {title}"
                )

                st.write(
                    content
                )

                st.caption(
                    f"Created: {created_at}"
                )

                # BUTTONS
                
                col1, col2 = st.columns(2)

                # EDIT BUTTON

                with col1:

                    edit_clicked = st.button(
                        "✏️ Edit",
                        key=f"edit_{note_id}"
                    )

                # DELETE BUTTON
                
                with col2:

                    delete_clicked = st.button(
                        "🗑️ Delete",
                        key=f"delete_{note_id}"
                    )

                # EDIT ACTION

                if edit_clicked:

                    st.session_state.editing_note_id = note_id

                    st.session_state.editing_title = title

                    st.session_state.editing_content = content

                    st.rerun()

                # DELETE ACTION
               
                if delete_clicked:

                    deleted = delete_note(
                        note_id,
                        st.session_state.user_id
                    )


                    if deleted:

                        st.success(
                            "✅ Note deleted successfully!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "❌ Could not delete this note."
                        )


                st.divider()

    # USER IS NOT LOGGED IN
   
    else:

        st.header("🔐 Login")

        photo = st.camera_input(
            "Look at the camera and take a photo"
        )


        if photo is not None:

            image = Image.open(photo)

            st.image(
                image,
                caption="Captured image",
                width=300
            )

            image_array = np.array(image)

            # FIND FACE
           
            faces, extra = engine.find_faces(
                image_array
            )


            if len(faces) == 0:

                st.error(
                    "❌ No face detected."
                )


            else:

                st.success(
                    "✅ Face detected!"
                )

                # CREATE EMBEDDING

                kpss = extra["kpss"]

                embeddings = engine.compute_embeddings(
                    image_array,
                    faces,
                    kpss=kpss
                )

                current_face = embeddings[0]

                # GET REGISTERED USERS
               
                users = get_users_with_embeddings()
                best_score = -1
                best_user = None

                # COMPARE FACE
        
                for (
                    user_id,
                    name,
                    registered_embedding
                ) in users:

                    score = engine.compare(
                        registered_embedding,
                        current_face
                    )


                    if score > best_score:

                        best_score = score

                        best_user = (
                            user_id,
                            name
                        )

                # CHECK MATCH
               
                if (
                    best_user is not None
                    and best_score >= 0.5
                ):

                    user_id, name = best_user


                    # Save login information
                    st.session_state.logged_in = True

                    st.session_state.user_id = user_id

                    st.session_state.user_name = name


                    st.success(
                        f"🔓 Face matched! Welcome, {name}"
                    )

                    st.write(
                        f"Similarity score: "
                        f"{best_score:.2f}"
                    )


                    st.rerun()


                else:

                    st.error(
                        "❌ Face not recognized. "
                        "Access denied."
                    )