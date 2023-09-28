import pymongo
from bson.json_util import dumps
import time
import json
from pose_estimation_functions_v2 import pose_estimation_avatar
from upload_download_files import S3Handler
import redis

# This code listens to changes in the MongoDB collection and triggers the 3d-pose-estimation code.

# Connect to MongoDB
client = pymongo.MongoClient('')  # Needs a specific value!
print("[INFO] Connected to MongoDB!")

print("[INFO] ******* 3D-pose estimation listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.Chromata_InputQueue_Poses.watch()  # May need adaptation.
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    # print(json_request)
    # print('') # for readability only
    print("\n\n [INFO] ******* Received a new request: \n")

    message = json_request["fullDocument"]
    print(message)

    print("\n[INFO] Analyzing input. Please wait...\n")

    # input_path = "Data/Output/" + "video_" + str(message["video_id"]) + "/shot_" \
    #              + str(message["shot_id"]) + "/vibe_output.pkl"
    output_path = "Data/Output/" + "video_" + str(message["video_id"]) + "/shot_" \
                 + str(message["shot_id"]) + "/rgb_demo_no_split/results_out/"   # just the path, not the file (without person_id)

    # ==================> Define output JSON
    output_dict = {}
    output_dict["workspace"] = message["workspace"]
    output_dict["video_id"] = message["video_id"]
    output_dict["project_id"] = message["project_id"]
    output_dict["shot_id"] = message["shot_id"]
    output_dict["user_id"] = message["user_id"]
    output_dict["bvh_urls"] = []   
    output_dict["fbx_urls"] = []
    output_dict["sender"] = "avatar-extraction"

    # Create an instance of the S3Handler
    s3 = S3Handler(                                    # Need to add specific values!
        aws_access_key_id="",
        aws_secret_access_key="",
        bucket_name=""
    )

    # Define path to save the derived video shots' clips
    ROOT_PATH_FOR_POSE_ESTIMATION = "subservices/pose_estimation/" + str(message["video_id"])  # this is a static path

    # Call the visual analysis function
    dancers_ids = message["dancers_ids"]
    for i in range(0, len(dancers_ids)):
        print("Producing animated avatar for dancer {}".format(dancers_ids[i]["id"]))
        gender = dancers_ids[i]["form"]
        print("Gender: {}".format(gender))

        if gender == "default":
            gender = "neutral"

        try:
            print(dancers_ids[i]["id"])
            pose_estimation_avatar(message)
        except:
            print("Error in creating the animation!")

        try:
            # Upload fbx file to s3
            url_file = s3.upload_file(file_name=output_path + "/person_" + str(dancers_ids[i]["id"]) + "/humor_output_person_" + str(dancers_ids[i]["id"]) + ".fbx",
                # object_name=os.path.join(ROOT_PATH_FOR_DANCE_RECOGNITION, "video_shot_" + str(i) + ".mp4")
                object_name=ROOT_PATH_FOR_POSE_ESTIMATION + "/shot_" + str(message["shot_id"]) +
                            "/humor_output_person_" + str(dancers_ids[i]["id"]) + ".fbx")
            print("Successful upload to s3!")
            print("url_file: {}".format(url_file))
            output_dict["fbx_urls"].append({"id": dancers_ids[i]["id"], "form": dancers_ids[i]["form"], "path": url_file})
        except:
            print("Error in uploading fbx file to s3!")

        # Send output json to 'completed jobs' topic
        environment = 'completed-jobs'
        x = json.dumps(output_dict)

        try:
            client = redis.Redis(host="", port=, password="")   # Need to add specific values!
            client.publish(environment, x)
            print("[INFO] 3D Pose Estimation result was successfully sent!")
        except Exception as e:
            print(f"ERROR:  {e}")

        path = 'Data/Output/' + 'video_' + str(message["video_id"]) + '/shot_' + str(message["shot_id"])
        with open(path + '/pose_avatar_output.json', 'w') as outfile:
            json.dump(output_dict, outfile)
        print("Output JSON was successfully saved!")       

    print("\n[INFO] ******* 3D-pose estimation listener is ready!")
