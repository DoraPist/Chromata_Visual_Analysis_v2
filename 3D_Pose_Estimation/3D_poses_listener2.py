import pymongo
from bson.json_util import dumps
import json
from pose_estimation_functions_v2 import pose_estimation_preview

# This code listens to changes in the MongoDB collection and triggers the 3D pose estimation code.

# Connect to MongoDB
client = pymongo.MongoClient('')   # Need to add a specific value!
print("[INFO] Connected to MongoDB!")

print("[INFO] ******* 3D-pose estimation listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.Chromata_InputQueue_Poses1.watch()  # May need adaptation.
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    print("\n\n [INFO] ******* Received a new request: \n")

    try:
        print(json_request["fullDocument"])

        # Call the visual analysis function
        pose_estimation_preview(json_request["fullDocument"])
    except:
        print(json_request["fullDocument"])
        print("KeyError: 'fullDocument'")

    print("\n[INFO] ******* 3D-pose estimation listener is ready!")
