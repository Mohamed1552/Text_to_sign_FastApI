import json
import requests
from io import BytesIO
from pose_format import Pose


class PoseRetriever:

    def __init__(self, json_path):

        with open(json_path, "r", encoding="utf-8") as f:
            self.pose_db = json.load(f)
        
        self.cache = {}
        
    def retrieve(self, tokens):

        pose_paths = []

        for token in tokens:

            # check token exists
            if token not in self.pose_db:
                print(f"Missing pose: {token}")
                continue

            url = self.pose_db[token]["url"]

            try:
                if token in self.cache:
                    pose_paths.append(self.cache[token])
                    continue
                response = requests.get(url)

                if response.status_code != 200:
                    print(f"Failed download: {token}")
                    continue

                pose = Pose.read(BytesIO(response.content))
                
                self.cache[token] = pose

                pose_paths.append(pose)

                print(f"Downloaded: {token}")

            except Exception as e:
                print(f"Error downloading {token}: {e}")

        return pose_paths