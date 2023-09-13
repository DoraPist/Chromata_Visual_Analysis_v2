import subprocess
import cv2
import os
import json
import time
from os import listdir
from os.path import isfile, join, exists
from natsort import natsorted
from upload_download_files import S3Handler
import redis
import gc


def video_preprocessing(message):

    start_time = time.time()

    # Create an instance of the S3Handler
    s3 = S3Handler(
        aws_access_key_id="",  # fill in value
        aws_secret_access_key="", # fill in value
        bucket_name="" # fill in value
    )

    # Create a folder for the current video
    video_id = message["video_id"]
    path = "Data/" + str(video_id) + "/"
    os.makedirs(path, exist_ok=True)

    # Save the input json
    with open(path + "input.json", "w") as f:
        json.dump(message, f)

    # Get video url
    video_url = message["video_path"]

    # --------------------------> Download video
    filename = str(video_id) + ".mp4"

    # Define path to save the derived video shots' clips
    ROOT_PATH_FOR_DANCE_RECOGNITION = "subservices/dance_recongition/" + str(video_id)  # this is a static path

    print("[INFO] Downloading video...")
    s3.download_file(
        object_name=video_url,
        output_file=path+filename
    )
    
    # --------------------------> Shot Detection
    print("[INFO] Splitting video into shots...")
    print(os.getcwd())
    # subprocess.run("python TransNetV2-master/inference/transnetv2.py " + path + filename)
    os.system("python TransNetV2-master/inference/transnetv2.py " + path + filename)
    
    # Clear memory
    gc.collect()

    # --------------------------> Save Keyframes (locally)
    # Create frames and keyframes' folders
    frames_path = path + "Frames/"
    os.makedirs(frames_path, exist_ok=True)
    keyframes_path = path + "Keyframes/"
    os.makedirs(keyframes_path, exist_ok=True)

    # Read txt with shot info
    start_frame = []
    key_frame = []   # save the middle frame of each video shot
    end_frame = []
    shot_filename = path + filename+".scenes.txt"
    shot_file = open(shot_filename, "r")
    shots_num = 0
    shot_ids = []
    shot_path = []
    for line in shot_file:
        start_frame.append(line.split()[0])
        end_frame.append(line.split()[-1])
        middle = int(end_frame[-1]) - int((int(end_frame[-1]) - int(start_frame[-1]))/2)
        key_frame.append(middle)
        shot_ids.append(shots_num)
        # shots_num = shots_num + 1
        shot_path.append(frames_path+'shot_'+str(shots_num)+'/')
        shots_num = shots_num + 1
        os.makedirs(shot_path[-1], exist_ok=True)
    print("[INFO] Shot boundary detection finished!")

    # --------------------------> Split video in frames
    print(path + filename)
    video_cap = cv2.VideoCapture(path + filename)
    print("[INFO] Loaded video: ", path + filename)

    # Get fps
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    # Read and save video keyframes
    print("[INFO] Getting video frames...")

    frame_idx = 0
    # shot_paths = []
    keyframes_paths = []
    shot = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("[INFO] Can't get another frame. Exiting ...")
            break
        else:
            if str(frame_idx - 1) in end_frame:
                shot = shot + 1
                keep_frame = 0
            if str(frame_idx) in start_frame:
                keep_frame = 1
            if frame_idx in key_frame:
                keyframes_paths.append(shot_path[shot] + "frame_" + str(frame_idx) + ".jpg")
                cv2.imwrite(keyframes_path + "frame_" + str(frame_idx) + ".jpg", frame)
                keep_frame = 1
            if keep_frame == 1:
                cv2.imwrite(shot_path[shot] + "frame_" + str(frame_idx) + ".jpg", frame)
                # shot_paths.append(shot_path[shot] + "frame_" + str(frame_idx) + ".jpg")
            frame_idx = frame_idx + 1

    video_cap.release()
    print("[INFO] Number of frames: {}, Frames are saved in {}".format(frame_idx - 1, frames_path))

    # # Define path to save the derived video shots' clips
    # ROOT_PATH_FOR_DANCE_RECOGNITION = "subservices/dance_recognition"  # this is a static path

    # --------------------------> Define the fields of the output JSON with video shot information
    shot_info = {}
    shot_info["workspace"] = message["workspace"]
    shot_info["user_id"] = message["user_id"]
    shot_info["video_id"] = message["video_id"]
    shot_info["project_id"] = message["project_id"]
    shot_info["sender"] = "video-preprocessing"
    shot_info["paths"] = []

    # -----------------------------------> Create a video file for every detected shot & upload it to datastorage
    for i in range(0, len(shot_path)):  #for idx, val in enumerate(my_list)
        print("Writing video for: ", shot_path[i])

        # load all the frames of current video shot
        onlyfiles = [f for f in listdir(shot_path[i]) if isfile(join(shot_path[i], f))]

        # Sort names numerically
        onlyfiles_n = natsorted(onlyfiles)

        img_array = []
        flag_get_size = 0
        for file in onlyfiles_n:
            print(file)
            img = cv2.imread(shot_path[i]+file)

            if flag_get_size == 0:
                # Get frames' size
                height, width, layers = img.shape
                # size = (width, height)

                # Set size to resize
                new_size = (int(width*0.4), int(height*0.4))

                # Set flag to 1
                flag_get_size = 1

            # Resize image
            new_img = cv2.resize(img, new_size, cv2.INTER_AREA)

            # Add resized image to the array
            img_array.append(new_img)

        try:
            out = cv2.VideoWriter(keyframes_path + "pre_video_shot_" + str(i) + ".mp4",
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, new_size)

            # Write video
            for j in range(len(img_array)):
                out.write(img_array[j])
            out.release()

            if not exists(keyframes_path + "video_shot_" + str(i) + ".mp4"):
                print("ffmpeg -i " + keyframes_path + "pre_video_shot_" + str(i) + ".mp4" + " -vcodec libx264 " + keyframes_path + "video_shot_" + str(i) + ".mp4")
                os.system("ffmpeg -i " + keyframes_path + "pre_video_shot_" + str(i) + ".mp4" + " -vcodec libx264 " + keyframes_path + "video_shot_" + str(i) + ".mp4")
            else:
                print(keyframes_path + "video_shot_" + str(i) + ".mp4 exists")

            # Delete pre-file
            os.remove(keyframes_path + "pre_video_shot_" + str(i) + ".mp4")
        except:
            print("Could not write video: ", keyframes_path + "video_shot_" + str(i) + ".mp4")
            pass

        # Upload it to s3
        url_file = s3.upload_file(
            file_name=keyframes_path + "video_shot_" + str(i) + ".mp4",
            # object_name=os.path.join(ROOT_PATH_FOR_DANCE_RECOGNITION, "video_shot_" + str(i) + ".mp4")
            object_name=ROOT_PATH_FOR_DANCE_RECOGNITION + "/video_shot_" + str(i) + ".mp4"
        )

        shot_info["paths"].append({"id": shot_ids[i], "path": url_file, "keyframe_time": key_frame[i]/fps})

    with open(path + 'shot_info_output.json', 'w') as outfile:
        json.dump(shot_info, outfile)

    print("[INFO] Shot info output was saved in ", path + 'shot_info_output.json')

    end_time = time.time()
    print("[INFO] Video-preprocessing took {} seconds".format(end_time-start_time))

    # Send output json to specific topic
    environment = '' # fill in value
    # action = outfile
    x = json.dumps(shot_info)

    try:
        client = redis.Redis(host="", port=, password="")  # fill in values
        client.publish(environment, x)
        print("[INFO] Video-preprocessing result was successfully sent!")
    except Exception as e:
        print(f"ERROR:  {e}")
