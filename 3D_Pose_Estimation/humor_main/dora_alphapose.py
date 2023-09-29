import os
import subprocess

# alphapose_path = "/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/humor_main/AlphaPose/"
# img_dir = "/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/humor_main/AlphaPose/examples/demo"
# out_dir = "/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/humor_main/AlphaPose/examples"


alphapose_path = "/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/humor_main/AlphaPose/"
img_dir = "examples/demo/"
out_dir = "examples/demo/res/"

og_cwd = os.getcwd()
os.chdir(alphapose_path)
print("Current directory: ")
print(os.getcwd())

run_cmds = "python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml " \
           "--checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir " + img_dir + " --pose_track --format open --outdir " + out_dir + " --save_img --showbox --vis_fast"
print(run_cmds)
subprocess.call(run_cmds, shell=True)