import sqlite3
import numpy as np


def create_database():
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        face_embedding BLOB NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    connection.commit()
    connection.close()


def add_user(name, face_embedding):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    embedding_bytes = face_embedding.tobytes()

    cursor.execute(
        "INSERT INTO users (name, face_embedding) VALUES (?, ?)",
        (name, embedding_bytes)
    )

    connection.commit()
    connection.close()

    print(f"User '{name}' added successfully! ")


def get_users():
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute("SELECT id, name, face_embedding FROM users")

    users = cursor.fetchall()

    connection.close()

    return users

def get_users_with_embeddings():
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute("SELECT id, name, face_embedding FROM users")

    users = cursor.fetchall()

    connection.close()

    result = []

    for user_id, name, embedding_bytes in users:

        embedding = np.frombuffer(
            embedding_bytes,
            dtype=np.float32
        )

        result.append(
            (user_id, name, embedding)
        )

    return result

def delete_user(user_id):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM users WHERE id = ?",
        (user_id,)
    )

    connection.commit()

    if cursor.rowcount > 0:
        print(f"User with ID {user_id} deleted successfully!")
    else:
        print(f"No user found with ID {user_id}.")

    connection.close()

def add_note(user_id, title, content):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO notes (user_id, title, content)
        VALUES (?, ?, ?)
        """,
        (user_id, title, content)
    )

    connection.commit()
    connection.close()

    print("Note added successfully!")

def get_user_notes(user_id):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, title, content, created_at
        FROM notes
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    )

    notes = cursor.fetchall()

    connection.close()

    return notes

def delete_note(note_id, user_id):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM notes
        WHERE id = ? AND user_id = ?
        """,
        (note_id, user_id)
    )

    connection.commit()

    deleted = cursor.rowcount > 0

    connection.close()

    return deleted 

def update_note(note_id, user_id, title, content):
    connection = sqlite3.connect("facevault.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE notes
        SET title = ?, content = ?
        WHERE id = ? AND user_id = ?
        """,
        (title, content, note_id, user_id)
    )

    connection.commit()

    updated = cursor.rowcount > 0

    connection.close()

    return updated