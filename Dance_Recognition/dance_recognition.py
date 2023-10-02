import os
import torch
import cv2
import numpy as np
import redis
import json
import time
from utils00 import *
import C3D_model as C3D_model


def dance_recognition(message):
    print(message)

    start_time = time.time()

    video_id = str(message["video_id"])
    shot_id = message["shot_id"]
    
    video_path = "C:/Chromata/Services_v1/Video_Preprocessing/Data/" + str(video_id) + '/Frames/' + 'shot_' + str(shot_id) + "/"
    path = "C:/Chromata/Services_v1/dance2/Data/" + str(video_id) + "/"
    os.makedirs(path, exist_ok=True)

    dances = ['Γίκνα', 'Μπαιντούσκα', 'Καρσιλαμάς', 'Χασάπικος']
  
    # Transform video to appropriate format
    X = load_data(video_path, 16, 2, True)  # 16 is the batch size, can change it to 30

    # Transform numpy array to torch tensor
    X = torch.tensor(X, dtype=torch.float)

    # Load SlowFast trained model
    print("[INFO] Loading recognition model...")

    model = C3D_model.C3D(num_classes=4, pretrained=False, gray_input=True)
    pretrained_path = 'C:/Chromata/Services_v1/dance2/C3D_skeleton_l.pt'    # To get the pre-trained model, please contact loupgeor@iti.gr or tpistola@iti.gr
    pretrain = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(pretrain)    

    # Predict dance type
    print("[INFO] Recognizing dance type...")

    model.eval()
    y_pred = model(X)
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    predicted = torch.max(y_pred.data, 1)[1]
    pred_freq_per_class = np.bincount(predicted) / predicted.shape[0]

    predicted_dance = np.argmax(np.bincount(predicted))
    print("[INFO] Predicted dance is: ", dances[predicted_dance])

    # --------------------------> Create output JSON with video shot information

    dance_info = {}
    dance_info["workspace"] = message["workspace"]
    dance_info["video_id"] = message["video_id"]
    dance_info["project_id"] = message["project_id"]
    dance_info["shot_id"] = message["shot_id"]
    dance_info["sender"] = "dance-recognition"
    dance_info["user_id"] = message["user_id"]
    dance_info["dance-type"] = dances[predicted_dance]

    with open(path+'dance_info_output.json', 'w', encoding='utf8') as outfile:
        json.dump(dance_info, outfile, ensure_ascii=False)

  print("[INFO] Dance info output was saved in " + path + 'dance_info_output.json')

    end_time = time.time()
    print("[INFO] Dance-recognition took {} seconds".format(end_time - start_time))

    # Send output json to 'completed-jobs' topic
    environment = 'completed-jobs'
    x = json.dumps(dance_info)

    try:
        client = redis.Redis(host="", port=, password="")  # Need to add specific values here!
        client.publish(environment, x)
        print("[INFO] Dance-recognition result was successfully sent!")
    except Exception as e:
        print(f"ERROR:  {e}")
