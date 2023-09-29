import redis
import time
import json
from pymongo import MongoClient
import os

# This code receives input requests for the laban-generation service and saves them to the MongoDB.

# Connect to redis
topic = "laban-generation"
client = redis.Redis(host="", port=, password="")  # Need to add specific values here!
p = client.pubsub()
p.subscribe(topic)
print("[INFO] Connected to redis!")

# Connect to mongoDB
MONGO_URI = ""  # Need to add a specific value here!
db = MongoClient(MONGO_URI) #, ssl_cert_reqs=ssl.CERT_NONE)
print("[INFO] Connected to MongoDB!")

print("[INFO] Laban-Generation waits for a message...")

while True:
    message = p.get_message()
    if message:
        print(message)

        # Check data type in message
        if message and type(message['data']) is bytes:
            input_data = json.loads(message["data"])
            print(input_data)

            # Save input message locally
            path = "Data/Input_messages/"
            os.makedirs(path, exist_ok=True)
            with open(path + "input_" + time.strftime("%Y%m%d%H%M%S") + ".json", 'w') as fp:
                json.dump(json.loads(message["data"]), fp)
            print("[INFO] Message is saved in ", path + "input_" + time.strftime("%Y%m%d%H%M%S") + ".json")

            # Save input message in MongoDB
            db.ImagesDB.Chromata_InputQueue_LG.insert_one(input_data)
            print("[INFO] Message is saved in MongoDB")

        print("\n[INFO] Laban-Generation waits for a message...")



