import pymongo
from datetime import datetime, timedelta
from bson.objectid import ObjectId

class DatabaseManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="auravision_db"):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        self._first_check_done = False
        # Initialize client structure without blocking on server selection
        try:
            self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self._init_collections()
        except Exception as e:
            print(f"Failed to initialize MongoDB client: {e}")

    def connect(self):
        return self.check_connection()

    def _init_collections(self):
        # Collections will be auto-created on first insert, but we can verify here
        if self.db is not None:
            self.users = self.db["users"]
            self.mood_logs = self.db["mood_logs"]
            self.journals = self.db["journals"]
            self.chats = self.db["chats"]
            self.calibration = self.db["calibration"]
        else:
            self.users = None
            self.mood_logs = None
            self.journals = None
            self.chats = None
            self.calibration = None

    def check_connection(self):
        was_connected = self.connected
        try:
            if self.client is None:
                self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=2000)
                self.db = self.client[self.db_name]
                self._init_collections()
            
            # Force server info check (will timeout in 2 seconds if offline)
            self.client.server_info()
            self.connected = True
            if not was_connected:
                print(f"Successfully connected to MongoDB at {self.uri}")
            self._first_check_done = True
            return True
        except Exception as e:
            self.connected = False
            if was_connected or not self._first_check_done:
                print(f"Failed to connect to MongoDB or connection lost: {e}")
            self._first_check_done = True
            return False

    def is_connected(self):
        return self.connected


    # --- User Methods ---
    def get_or_create_user(self, username="DefaultUser"):
        if not self.is_connected():
            return None
        user = self.users.find_one({"username": username})
        if not user:
            user_id = self.users.insert_one({
                "username": username,
                "created_at": datetime.now()
            }).inserted_id
            return user_id
        return user["_id"]

    # --- Mood / Focus Logging Methods ---
    def log_mood(self, user_id, emotion, confidence, focus_score, blink_count=0):
        if not self.is_connected():
            return False
        try:
            self.mood_logs.insert_one({
                "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
                "timestamp": datetime.now(),
                "emotion": emotion,
                "confidence": float(confidence),
                "focus_score": float(focus_score),
                "blink_count": int(blink_count)
            })
            return True
        except Exception as e:
            print(f"Error logging mood: {e}")
            return False

    def get_mood_history(self, user_id, limit=50):
        if not self.is_connected():
            return []
        try:
            cursor = self.mood_logs.find(
                {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
            ).sort("timestamp", pymongo.DESCENDING).limit(limit)
            
            # Return sorted chronologically for charting
            return list(cursor)[::-1]
        except Exception as e:
            print(f"Error fetching mood history: {e}")
            return []

    def get_aggregate_emotions(self, user_id, hours=24):
        if not self.is_connected():
            return {}
        try:
            since = datetime.now() - timedelta(hours=hours)
            pipeline = [
                {
                    "$match": {
                        "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
                        "timestamp": {"$gte": since}
                    }
                },
                {
                    "$group": {
                        "_id": "$emotion",
                        "count": {"$sum": 1}
                    }
                }
            ]
            results = self.db["mood_logs"].aggregate(pipeline)
            return {r["_id"]: r["count"] for r in results}
        except Exception as e:
            print(f"Error aggregating emotions: {e}")
            return {}

    # --- Journal Methods ---
    def save_journal_entry(self, user_id, entry_text, sentiment_polarity, sentiment_label):
        if not self.is_connected():
            return False
        try:
            self.journals.insert_one({
                "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
                "timestamp": datetime.now(),
                "entry_text": entry_text,
                "sentiment_polarity": float(sentiment_polarity),
                "sentiment_label": sentiment_label
            })
            return True
        except Exception as e:
            print(f"Error saving journal: {e}")
            return False

    def get_journal_history(self, user_id, limit=10):
        if not self.is_connected():
            return []
        try:
            cursor = self.journals.find(
                {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
            ).sort("timestamp", pymongo.DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"Error fetching journals: {e}")
            return []

    # --- Chat Methods ---
    def save_chat_message(self, user_id, role, message, current_mood="Neutral"):
        if not self.is_connected():
            return False
        try:
            self.chats.insert_one({
                "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
                "timestamp": datetime.now(),
                "role": role,
                "message": message,
                "user_mood_context": current_mood
            })
            return True
        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False

    def get_chat_history(self, user_id, limit=30):
        if not self.is_connected():
            return []
        try:
            cursor = self.chats.find(
                {"user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id}
            ).sort("timestamp", pymongo.DESCENDING).limit(limit)
            return list(cursor)[::-1]
        except Exception as e:
            print(f"Error fetching chat history: {e}")
            return []

    # --- Calibration Methods (for custom scikit-learn classifier) ---
    def save_calibration_data(self, user_id, emotion_label, feature_list):
        """
        Saves feature vectors extracted from webcam images for a specific emotion.
        feature_list is a list of lists containing floats (facial landmark ratios).
        """
        if not self.is_connected():
            return False
        try:
            self.calibration.update_one(
                {
                    "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id,
                    "emotion": emotion_label
                },
                {
                    "$set": {
                        "updated_at": datetime.now(),
                        "features": feature_list
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving calibration data: {e}")
            return False

    def get_all_calibration_data(self, user_id):
        if not self.is_connected():
            return []
        try:
            cursor = self.calibration.find({
                "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id
            })
            return list(cursor)
        except Exception as e:
            print(f"Error fetching calibration data: {e}")
            return []

    def clear_calibration_data(self, user_id):
        if not self.is_connected():
            return False
        try:
            self.calibration.delete_many({
                "user_id": ObjectId(user_id) if isinstance(user_id, str) else user_id
            })
            return True
        except Exception as e:
            print(f"Error clearing calibration data: {e}")
            return False
