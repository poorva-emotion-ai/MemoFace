from database import add_note, get_user_notes

add_note(
    1,
    "CNN Revision",
    "CNN uses convolutional filters to extract features."
)

notes = get_user_notes(1)

print(notes)