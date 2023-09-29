import pymongo
from bson.json_util import dumps
import time
import json
from laban_generation import laban_generation

# This code listens to changes in the MongoDB collection and triggers the laban generation code.

# Connect to MongoDB
client = pymongo.MongoClient('')   # Need to add a specific value here!
print("[INFO] Connected to MongoDB!")

print("[INFO] ******* Laban-Generation listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.Chromata_InputQueue_LG.watch()   # May need adaptation.
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    json_keys = json_request.keys()
    print(json_keys)

    if "fullDocument" in json_keys:
        print("\n\n [INFO] ******* Received a new request: \n")
        print(json_request["fullDocument"])

        # Call the visual analysis function
        laban_generation(json_request["fullDocument"])


    print("\n[INFO] ******* Laban-Generation listener is ready!")

