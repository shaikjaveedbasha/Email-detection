# inspect_db.py
import os
import sys
from app import app, db, User
from werkzeug.security import check_password_hash

# Run with: python inspect_db.py

def show_users():
    with app.app_context():
        users = User.query.all()
        if not users:
            print("No users found in DB.")
            return
        for u in users:
            print("-----")
            print("id:", u.id)
            print("username:", u.username)
            print("email:", u.email)
            pw = getattr(u, "password_hash", None)
            print("has password_hash?:", bool(pw))
            if pw:
                print("password_hash (prefix):", pw[:50])
            else:
                print("password_hash is EMPTY or None")
            # quick verify: try a sample password test (dev only)
            sample = "testpassword"
            try:
                ok = u.check_password(sample)
            except Exception as e:
                ok = f"check_password raised: {e}"
            print(f"check_password('testpassword') -> {ok}")
        print("----- Done -----")

if __name__ == "__main__":
    show_users()
