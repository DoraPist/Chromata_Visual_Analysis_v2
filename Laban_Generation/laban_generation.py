import os
import json
import time
from upload_download_files import S3Handler
import redis
import cv2
import numpy as np
from laban_utils0 import *

def laban_generation(message):
    start_time = time.time()

    # Create an instance of the S3Handler   
    s3 = S3Handler(                       # Need to add specific values here!
        aws_access_key_id="",
        aws_secret_access_key="",
        bucket_name=""
    )

    # Create a folder for the current media
    video_id = message["video_id"]
    shot_id = message["shot_id"]

    ROOT_PATH_FOR_LABAN_GENERATION='subservices/laban_generation/'
    path = os.getcwd() + "/Data/" + str(video_id) + "/" + str(shot_id) + "/"    
    bvh_file = "C:/Chromata/Services_v1/Pose-Estimation_v2/Data/Output/"+"video_"+ str(video_id) + "/" + "shot_"+str(shot_id) + "/rgb_demo_no_split/results_out/person_1/humor_output_person_1.bvh"
    os.makedirs(path, exist_ok=True)

    # Save the input json
    with open(path + "input.json", "w") as f:
        json.dump(message, f)
      
    ###Run Laban Generation algorithm....
    print("[INFO] Generating Labanotation...")
    laban_script = labanotation(bvh_file)
    labanotation_filename = "labanotation_" + str(video_id) + "_" + str(shot_id) + ".png"   
    cv2.imwrite(path+labanotation_filename, laban_script)

    # Upload it to s3
    url_file = s3.upload_file(
        file_name=path+labanotation_filename,
        object_name=ROOT_PATH_FOR_LABAN_GENERATION + labanotation_filename
    )

    labgen_info = {}
    labgen_info["workspace"] = message["workspace"]
    labgen_info["project_id"] = message["project_id"]
    labgen_info["video_id"] = message["video_id"]
    labgen_info["shot_id"] = message["shot_id"]
    labgen_info["laban_url"] = url_file
    labgen_info["user_id"] = message["user_id"]
    labgen_info["sender"] = "laban-generation"
    labgen_info["timestamp"] = message["timestamp"]
    with open(path + 'labgen_info_output.json', 'w') as outfile:
        json.dump(labgen_info, outfile)

    print("[INFO] Laban-Generation info output was saved in ", path + 'labgen_info_output.json')

    end_time = time.time()
    print("[INFO] Laban-Generation took {} seconds".format(end_time - start_time))

    # Send output json to 'completed-jobs' topic
    environment = 'completed-jobs'
    # action = outfile
    x = json.dumps(labgen_info)

    try:
        client = redis.Redis(host="", port=, password="")   # Need to add specific values here!
        client.publish(environment, x)
        print("[INFO] Laban-Generation result was successfully sent!")
    except Exception as e:
        print(f"ERROR:  {e}")



