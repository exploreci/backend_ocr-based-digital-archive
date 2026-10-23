import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(user):
    payload = {
        "id_user": user["id_user"],
        "id_role": user["id_role"],
        "id_unit": user["id_unit"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }

    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")