from database import get_users

users = get_users()

for user in users:
    user_id = user[0]
    name = user[1]

    print("ID:", user_id)
    print("Name:", name)
    print("----------------")