import time
from upload_download_files import S3Handler
import redis
import json
from datetime import datetime
import subprocess
import os
import bpy
import numpy as np
import gc
import joblib


def fbx2bvh(data_path, file):
    sourcepath = data_path+"/"+file
    bvh_path = data_path+"/"+file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if  action.frame_range[1] > frame_end:
      frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
      frame_start = action.frame_range[0]

    frame_end = np.max([60, frame_end])   # todo!!! set to 60 as it was?
    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start=frame_start,
                            frame_end=frame_end, root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(data_path+"/"+file+" processed.")


def pose_estimation_preview(message):
    print(message)
   
    # Get video from local path
    video_id = str(message["video_id"])
    shot_id = message["shot_id"]
    video_path = "C:/Chromata/Services_v1/Video_Preprocessing/Data/" + str(video_id) + '/Keyframes/' + 'video_shot_' + str(shot_id) + ".mp4"   
    print(video_path)

    # Define path to save the derived video shots' clips
    ROOT_PATH_FOR_POSE_ESTIMATION = "subservices/pose_estimation/" + str(video_id)  # this is a static path
   
    # Run humor (only person detection/tracking)
    project_path = "C:/Chromata/Services_v1/Pose-Estimation_v2/"  # May need adaptation   
    print(os.getcwd())
    os.chdir(project_path)   # Change directory
    print("Current directory: ")
    print(os.getcwd())
   
    run_cmds = "python humor_main/humor/fitting/run_fitting_preview.py --data-path " + video_path + " --out Data/Output/video_" \
               + str(video_id) + "/shot_" + str(shot_id) + "/rgb_demo_use_split"
    print(run_cmds)
    subprocess.call(run_cmds, shell=True)

    # Clear memory
    gc.collect()

    # Send video to Chromata platform (u2m)
    # Create an instance of the S3Handler
    s3 = S3Handler(                                 # Need to add specific values!!
        aws_access_key_id="",
        aws_secret_access_key="",
        bucket_name=""
    )

    # Upload it to s3    
    url_file = s3.upload_file(
        file_name="C:/Chromata/Services_v1/Pose-Estimation_v2/Data/Output/video_" + str(video_id) +
                  "/shot_" + str(shot_id) + "/rgb_demo_use_split/rgb_preprocess/op_keypoints_overlay.mp4",        
        object_name=ROOT_PATH_FOR_POSE_ESTIMATION + "/shot_" + str(shot_id) + "_poses_preview.mp4"
    )

    print("url_file: {}".format(url_file))

    output_dict = {}
    output_dict["workspace"] = message["workspace"]
    output_dict["video_id"] = message["video_id"]
    output_dict["project_id"] = message["project_id"]
    output_dict["shot_id"] = message["shot_id"]
    output_dict["user_id"] = message["user_id"]
    output_dict["sender"] = '3d-pose'
    output_dict["preview_path"] = url_file

    print("Output for 3D Pose Estimation preview: \n{}".format(output_dict))

    # Send output json to 'completed jobs' topic
    environment = 'completed-jobs'   
    x = json.dumps(output_dict)

    try:
        client = redis.Redis(host="", port=, password="")   # Need to add specific values!
        client.publish(environment, x)
        print("[INFO] 3D Pose Estimation preview result was successfully sent!")
    except Exception as e:
        print(f"ERROR:  {e}")
   
    path = 'C:/Chromata/Services_v1/Pose-Estimation_v2/Data/Output/' + 'video_' + \
           str(video_id) + '/shot_' + str(shot_id)
    with open(path + '/pose_preview_output.json', 'w') as outfile:
        json.dump(output_dict, outfile)
    print("Output JSON was successfully saved!")


def pose_estimation_avatar(message):
    print(message)

    # Get video from local path
    video_id = str(message["video_id"])
    shot_id = message["shot_id"]   
    video_path = "C:/Chromata/Services_v1/Video_Preprocessing/Data/" + str(
        video_id) + '/Keyframes/' + 'video_shot_' + str(shot_id) + ".mp4"
    print(video_path)

    person_ids = message["dancers_ids"]
    print(person_ids)

    for person in range(0, len(person_ids)):

        # Run humor (pose estimation per dancer)
        # project_path = "/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/"
        project_path = "C:/Chromata/Services_v1/Pose-Estimation_v2/"
        # humor_path = "/home/tpistola/Projects/Chromata/Services_v1/"
        # print(os.getcwd())
        os.chdir(project_path)  # Change directory
        print("Current directory: ")
        print(os.getcwd())
        person_id = int(person_ids[person]["id"])
        gender = person_ids[person]["form"]

        run_cmds = "python humor_main/humor/fitting/run_fitting.py --data-path " + video_path + " --out Data/Output/video_" \
                     + str(video_id) + "/shot_" + str(shot_id) + "/rgb_demo_no_split/" + " --person_id " + str(person_id) \
                     + " --gender " + gender

        print(run_cmds)
        subprocess.call(run_cmds, shell=True)

        # Clear memory
        gc.collect()

        # Create pkl file
        os.chdir(project_path)  # Change directory
        print("Current directory: ")
        print(os.getcwd())

        run_cmds = "python humor_main/humor/fitting/create_pkl_ala_VIBE_multi.py --results  Data/Output/video_" + str(video_id) \
                   + "/shot_" + str(shot_id) + "/rgb_demo_no_split/results_out/person_" + str(person_id) + " --out Data/Output/video_" \
                   + str(video_id) + "/shot_" + str(shot_id) + "/rgb_demo_no_split/results_out/person_" + str(person_id) + \
                   "/viz_out --viz-final-only --person_id " + str(person_id)

        print(run_cmds)
        subprocess.call(run_cmds, shell=True)

        # Clear memory
        gc.collect()

        # Create fbx file from pkl file
        print("[INFO] Creating fbx file from pkl file...")
 
        # Read pickle file
        print("fbx current dir: ", os.getcwd())
        os.chdir("C:/Chromata/Services_v1/Pose-Estimation_v2/")
        print("fbx current dir: ", os.getcwd())
        input_path = "C:/Chromata/Services_v1/Pose-Estimation_v2/Data/Output/video_" + str(video_id) + "/shot_" + str(shot_id) + "/rgb_demo_no_split/results_out/person_" + str(person_id)
        pkl_file = "/humor_output_person_" + str(person_id) + ".pkl"
        print("For joblib: ", input_path + pkl_file)
        data = joblib.load(input_path + pkl_file)

        print("Found {} person ids".format(len(data)))

        fbx_file_basename = "/humor_output_person_" + str(person_id)
        fps_source = 30
        fps_target = 30
        gender = "male"
        num_person = len(data)

        # Get person ids
        keys = list(data.keys())
        print("Person ids detected: ", keys)

        person_i = 1
        run_cmds = "python humor_main/fbx_output_mixamo.py --input " + str(input_path + pkl_file) + " --output " + \
                       str(input_path + fbx_file_basename + ".fbx") + " --fps_source " + \
                       str(fps_source) + " --fps_target " + str(fps_target) + " --gender " + str(gender) + \
                       " --person_id " + str(person_i)
        subprocess.call(run_cmds, shell=True)

        # Clear memory
        gc.collect()

        # Create bvh from fbx
        fbx2bvh(input_path , fbx_file_basename + ".fbx")

        # Clear memory
        gc.collect()
  
