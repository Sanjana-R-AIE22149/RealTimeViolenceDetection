from pymongo import MongoClient

class MongoDBLogger:
    def __init__(self, uri="mongodb://localhost:27017", db_name="violenceDB"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["predictions"]

    def insert_prediction(self, video_name, prediction):
        self.collection.insert_one({
            "video": video_name,
            "prediction": prediction
        })
