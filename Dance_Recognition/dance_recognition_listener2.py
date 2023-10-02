import pymongo
from bson.json_util import dumps
import time
import json
from dance_recognition import dance_recognition

# This code listens to changes of the MongoDB collection and triggers the dance recognition code.

# Connect to MongoDB
client = pymongo.MongoClient('')   # Need to add a specific value here!!
print("[INFO] Connected to MongoDB!")

print("[INFO] ******* Dance-recognition listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.Chromata_InputQueue_DR.watch()   # May need adaptation!
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    json_keys = json_request.keys()
    # print(json_request)
    # print('') # for readability only
    print("\n\n [INFO] ******* Received a new request: \n")

    print(json_keys)

    if "fullDocument" in json_keys:
        print(json_request["fullDocument"])

    # print("\n[INFO] Analyzing input. Please wait...\n")
    # Call the visual analysis function
        dance_recognition(json_request["fullDocument"])

    print("\n[INFO] ******* Dance-recognition listener is ready!")
