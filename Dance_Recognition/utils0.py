import os
import subprocess
import cv2
import json
import numpy as np


POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Body
    (11, 13), (12, 14), (13, 15), (14, 16)
]


color = [
    (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
    (77, 222, 255), (255, 156, 127),
    (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]


right =[6 ,8 ,10 ,12 ,14 ,16]
left =[5 ,7 ,9 ,11 ,13 ,15]


def num_of_frames(b ,video):
    path ='Downloads/videos'
    video =path +video
    cap = cv2.VideoCapture(video)

    frames =0
    while 1:
        ret, frame = cap.read()
        if ret:
            # print(frames)
            frames = frames + 1
        else:
            break
    cap.release()
    return frames


def change_format(video_path):
    cap = cv2.VideoCapture(video_path)

    tot_frames = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            # print(frames)
            tot_frames = tot_frames + 1
        else:
            break
    cap.release()

    ######
    keypoints_file =video_path +'.alphapose.json'
    b = []

    for line in open(keypoints_file, 'r'):
        b.append(json.loads(line))

    frame_set =set()
    new_b =[]
    count =0
    # prev_frame=0
    for i in range(len(b[0])):

        if int(b[0][i]['image_id'][:-4]) not in frame_set:
            curr_frame =int(b[0][i]['image_id'][:-4])

            frame_set.add(curr_frame)
            # if curr_frame
            if i> 0:
                count = count + 1
                # print(count)
                new_b.append(temp_dict)

            temp_dict = {}
            temp_dict['frame'] = curr_frame + 1
            temp_dict['predictions'] = []

            temp_dict2 = {}
            temp_dict2['keypoints'] = b[0][i]['keypoints']
            temp_dict2['box'] = b[0][i]['box']
            temp_dict2['score'] = b[0][i]['score']
            temp_dict2['category_id'] = b[0][i]['category_id']
            temp_dict2['idx'] = b[0][i]['idx']
            temp_dict['predictions'].append(temp_dict2)
            prev_frame = curr_frame

        else:
            temp_dict2 = {}
            temp_dict2['keypoints'] = b[0][i]['keypoints']
            temp_dict2['box'] = b[0][i]['box']
            temp_dict2['score'] = b[0][i]['score']
            temp_dict2['category_id'] = b[0][i]['category_id']
            temp_dict2['idx'] = b[0][i]['idx']
            temp_dict['predictions'].append(temp_dict2)

        # if i == len(b[0])-1:
    new_b.append(temp_dict)

    for i in range(tot_frames):
        if i not in frame_set:
            temp_dict = {}
            temp_dict['frame'] = i + 1
            temp_dict['predictions'] = []
            new_b.insert(i, temp_dict)

    return new_b


def alpha_skeletal(video_path):
    # cap = cv2.VideoCapture('test_mask.wmv')

    # fcc = cap.get(cv2.CAP_PROP_FOURCC)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fcc = cap.get(cv2.CAP_PROP_FOURCC)
    ret, im = cap.read()

    # outt = cv2.VideoWriter('test_GIKNA_POSE.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (im.shape[1], im.shape[0]))

    b = change_format(video_path)

    cap = cv2.VideoCapture(video_path)

    framelist = []
    for frame in range(len(b)):
        ret, image = cap.read()

        if ret == False:
            break

        image = np.zeros((image.shape[0], image.shape[1], image.shape[2])).astype(np.uint8)
        # print(frame)

        for pose in range(len(b[frame]['predictions'])):
            points = []
            probs = []
            if 'idx' not in b[frame]['predictions'][pose].keys():
                continue
            elif np.array(b[frame]['predictions'][pose]['keypoints'])[np.arange(2, 51, 3)].mean() <= 0.6:
                continue
            else:
                # for i in range(0,17):
                for i in range(0, 17):

                    point = b[frame]['predictions'][pose]['keypoints'][(i * 3):(i * 3) + 3]
                    # print(point)
                    points.append([point[0], point[1]])
                    probs.append(point[2])
                    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
                        continue
                    if point[0] == 0:
                        continue
                    else:
                        pass
                        
                cr_ind = -1
                for pair in POSE_PAIRS:

                    if pair == (0, 1) or pair == (0, 2) or pair == (1, 3) or pair == (2, 4):
                        continue
                    ##print(points[pair[0] - 1][0])

                    ##if points[pair[0] - 1][0] == 0 or points[pair[1] - 1][0] == 0:
                    # #   continue
                    if points[pair[0]][0] == 0 or points[pair[1]][0] == 0:
                        continue
                    # if pair== (12,13) or pair==(6,7):
                    #    clr=color[2]

                    # elif pair[0] in left:
                    #    clr=color[0]
                    # else:
                    #    clr=color[1]

                    # cv2.line(image, (int(round(points[pair[0] - 1][0])), int(round(points[pair[0] - 1][1]))),
                    #        (int(round(points[pair[1] - 1][0])), int(round(points[pair[1] - 1][1]))), clr, 2,
                    #       cv2.LINE_AA)
                    cr_ind = cr_ind + 1
                    cv2.line(image, (int(round(points[pair[0]][0])), int(round(points[pair[0]][1]))),
                             (int(round(points[pair[1]][0])), int(round(points[pair[1]][1]))), (255, 255, 255), 2
                             )

        framelist.append(image)

    return framelist


def extract_skeletons(rgb_path, vid_id):
    # def vid_to_skeleton(video_path, out_path)
    # alphapose_path = "/home/tpistola/Projects/Chromata/Services_v1/dance2/AlphaPose/"
    alphapose_path = "C:/Chromata/Services_v1/dance2/AlphaPose/"

    og_cwd = os.getcwd()
    os.chdir(alphapose_path)

    video = rgb_path + vid_id + '.mp4'
    out_path = rgb_path

    # run_cmds = "python /home/tpistola/Projects/Chromata/Services_v1/dance2/AlphaPose/scripts/demo_inference.py --video " + video + " --outdir " + out_path + " --cfg /home/tpistola/Projects/Chromata/Services_v1/dance2/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint /home/tpistola/Projects/Chromata/Services_v1/dance2/AlphaPose/pretrained_models/fast_res50_256x192.pth --pose_track"
    run_cmds = "python C:/Chromata/Services_v1/dance2/AlphaPose/scripts/demo_inference.py --video " + video + " --outdir " + out_path + " --cfg C:/Chromata/Services_v1/dance2/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint C:/Chromata/Services_v1/dance2/AlphaPose/pretrained_models/fast_res50_256x192.pth --pose_track"

    print(run_cmds)
    subprocess.call(run_cmds, shell=True)

    os.rename(out_path + 'alphapose-results.json', out_path + vid_id + '.mp4.alphapose.json')

    return 1


def make_skeleton_vid(rgb_path, vid_id):
    skeleton_path = rgb_path.split('rgb_vids')[0] + 'skeleton_vids/' + vid_id + '/'
    os.makedirs(skeleton_path, exist_ok=True)
    fr = alpha_skeletal(rgb_path + vid_id + '.mp4')

    cap = cv2.VideoCapture(rgb_path + vid_id + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = cap.get(cv2.CAP_PROP_FOURCC)
    ret, im = cap.read()

    out_skeleton = cv2.VideoWriter(skeleton_path + vid_id + '_skeleton.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
                                   round(fps), (im.shape[1], im.shape[0]))

    for i in range(len(fr)):
        out_skeleton.write(fr[i])
    out_skeleton.release()

    return skeleton_path + vid_id + '_skeleton.mp4'


def load_video_data(video_path, batch_size, frame_interval, gray):
    # video_path = os.path.join(path, video)

    cap = cv2.VideoCapture(video_path)
    # frames = os.listdir(video_path)

    tot_frames = 0
    framearray = []
    volume_list = []
    # for fr in frames:
    while 1:
        # frame = cv2.imread(os.path.join(video_path, fr))
        ret, frame = cap.read()
        if ret:
            # tot_frames = tot_frames + 1
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (112, 112))
            if gray:
                frame = frame.reshape(112, 112, 1)
            framearray.append(frame / 255)
        else:
            break

        # else:
        #    break

    # cap.release()

    framearray = np.array(framearray)
    keep_frames = np.arange(0, framearray.shape[0], frame_interval)
    framearray = framearray[keep_frames]

    volume_list.extend(np.array_split(framearray[:framearray.shape[0] - (framearray.shape[0] % batch_size)],
                                      framearray.shape[0] // batch_size))

    return np.array(volume_list).transpose((0, 4, 1, 2, 3))
    # return np.array(volume_list).transpose((0, 1, 4, 2, 3))



# Function that reads video frames, samples per flrame_interval frames and splits them in batches of size batch_size
# Returns a 5d array (Number of batches, Image channels, Batch size, Image width, Image height)
def load_data(video_path, batch_size, frame_interval, gray):
    vid_id = video_path.split('/')[-4]
    # cap = cv2.VideoCapture(video_path)

    frames = os.listdir(video_path)
    frame = cv2.imread(os.path.join(video_path, frames[0]))

  rgb_path = 'C:/Chromata/Services_v1/dance2/Data/rgb_vids/' + vid_id + '/'
    os.makedirs(rgb_path, exist_ok=True)
    original_video_path = video_path.split('Frames')[0]
    cap = cv2.VideoCapture(original_video_path+vid_id + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out_rgb = cv2.VideoWriter(rgb_path + vid_id + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), round(fps),
                              (frame.shape[1], frame.shape[0]))

    for i in range(len(frames)):

        frame = cv2.imread(os.path.join(video_path, frames[i]))
        out_rgb.write(frame)
    out_rgb.release()

    flag = extract_skeletons(rgb_path, vid_id)

    if flag == 1:
        print("[INFO] Extracting skeleton representations...")
        skeleton_vid_path = make_skeleton_vid(rgb_path, vid_id)

    vid_data = load_video_data(skeleton_vid_path, batch_size, frame_interval, gray)

    return vid_data
