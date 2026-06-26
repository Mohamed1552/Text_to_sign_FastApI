import json

from cryptography.fernet import Fernet

from config import POSE_KEY


class PoseLoader:

    def __init__(self):

        cipher = Fernet(POSE_KEY.encode())

        with open("pose.enc", "rb") as f:
            encrypted = f.read()

        decrypted = cipher.decrypt(encrypted)

        self.pose_data = json.loads(decrypted)

    def get(self):

        return self.pose_data