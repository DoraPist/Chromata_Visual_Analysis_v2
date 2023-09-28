import redis
import time
import json
from pymongo import MongoClient

# This code receives input requests for the 3d-pose-estimation service and saves them to the MongoDB.

# Connect to redis
topic = "avatar-extraction"
client = redis.Redis(host="", port=, password="")  # Need to add specific values!
p = client.pubsub()
p.subscribe(topic)
print("[INFO] Connected to redis!")

# Connect to mongoDB
MONGO_URI = ""  # Need to add a specific value!
db = MongoClient(MONGO_URI) #, ssl_cert_reqs=ssl.CERT_NONE)
print("[INFO] Connected to MongoDB!")

print("[INFO] 3D-pose estimation waits for a message...")

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
            with open(path + "input_" + time.strftime("%Y%m%d%H%M%S") + ".json", 'w') as fp:
                json.dump(json.loads(message["data"]), fp)
            print("[INFO] Message is saved in ", path + "input_" + time.strftime("%Y%m%d%H%M%S") + ".json")

            # Save input message in MongoDB
            db.ImagesDB.Chromata_InputQueue_Poses.insert_one(input_data)  # May need adaptation.
            print("[INFO] Message is saved in MongoDB")

        print("\n[INFO] 3D-pose estimation waits for a message...")
