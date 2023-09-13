import pymongo
from bson.json_util import dumps
import time
import json
from video_preprocessing import video_preprocessing

# This code listens to changes of the MongoDB collection and triggers the video-preprocessing code

# Connect to MongoDB 
client = pymongo.MongoClient('') # fill in value
print("[INFO] Connected to MongoDB!")

print("[INFO] ******* Video-preprocessing listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.Chromata_InputQueue_VP.watch()  # change appropriately 
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    print("\n\n [INFO] ******* Received a new request: \n")

    try:
        print(json_request["fullDocument"])

        # Call the visual analysis function
        video_preprocessing(json_request["fullDocument"])
    except:
        print("KeyError: 'fullDocument'")

    print("\n[INFO] ******* Video-preprocessing listener is ready!")
