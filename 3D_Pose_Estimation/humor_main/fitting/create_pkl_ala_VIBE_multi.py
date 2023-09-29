import trimesh

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time, math, shutil, csv, random

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from humor.utils.config import SplitLineParser
# from humor.utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues
# from humor.utils.torch import load_state
from humor.utils.logging import mkdir

from humor.fitting.fitting_utils import load_res, prep_res, run_smpl
from humor.fitting.eval_utils import SMPL_SIZES

from humor.body_model.body_model import BodyModel
from humor.body_model.utils import SMPL_PATH, SMPLH_PATH, SMPL_JOINTS, SMPLX_PATH

# from humor.viz.utils import viz_smpl_seq, viz_results, create_gif, create_video, create_multi_comparison_images, smpl_connections, imapper_connections, comp_connections
# from humor.viz.mesh_viewer import COMPRESS_PARAMS

from humor.utils.torch import copy2cpu as c2c

import joblib

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'stage3_results'
PRED_PRIOR_RES_NAME = 'stage3_results_prior'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results'] # results in camera frame
STAGES_PRIOR_RES_NAMES = ['stage2_results_prior', 'stage3_init_results_prior'] # results in prior frame (w.r.t final floor fit)
FINAL_RES_NAME = 'final_results'
FINAL_PRIOR_RES_NAME = 'final_results_prior'
OBS_NAME = 'observations'
FPS = 30

# visualization options
GROUND_ALPHA = 1.0
BODY_ALPHA = None # use to make body mesh translucent
IM_EXTN = 'jpg' # png # to use for rendering jpg saves a lot of space

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    parser.add_argument('--results', type=str, required=True, help='Path to the results_out directory from fitting to run viz on.')
    parser.add_argument('--out', type=str, required=True, help='Path to save visualizations to.')
    parser.add_argument('--person_id', type=int, required=True, help='Person id number to extract pkl for.')

    # visualization options
    parser.add_argument('--viz-final-only', dest='viz_final_only', action='store_true', help="If given only visualize the final full sequence result and not the subsequences.")
    parser.set_defaults(viz_final_only=False)
    parser.add_argument('--viz-stages', dest='viz_stages', action='store_true', help="If given, visualizes intermediate optimization stages and comparison to final pred.")
    parser.set_defaults(viz_stages=False)
    parser.add_argument('--viz-prior-frame', dest='viz_prior_frame', action='store_true', help="If given, also visualizes results in the HuMoR canonical coordinate frame.")
    parser.set_defaults(viz_prior_frame=False)
    parser.add_argument('--viz-obs-2d', dest='viz_obs_2d', action='store_true', help="If given, visualizes 2D joint observations on top of og video")
    parser.set_defaults(viz_obs_2d=False)
    parser.add_argument('--viz-no-render-cam-body', dest='viz_render_cam_body', action='store_false', help="If given, does not render body mesh from camera view")
    parser.set_defaults(viz_render_cam_body=True)
    parser.add_argument('--viz-pred-floor', dest='viz_pred_floor', action='store_true', help="Render the predicted floor from the camera view.")
    parser.set_defaults(viz_pred_floor=False)
    parser.add_argument('--viz-contacts', dest='viz_contacts', action='store_true', help="Render predicted contacts on the joints")
    parser.set_defaults(viz_contacts=False)
    parser.add_argument('--viz-wireframe', dest='viz_wireframe', action='store_true', help="Render body and floor in wireframe")
    parser.set_defaults(viz_wireframe=False)
    parser.add_argument('--viz-bodies-static', type=int, default=None, help="If given, renders all body predictions at once at this given frame interval interval.")
    parser.add_argument('--viz-no-bg', dest='viz_bg', action='store_false', help="If given will not overlay the rendering on top of OG video.")
    parser.set_defaults(viz_bg=True)

    parser.add_argument('--viz-render-width', type=int, default=1280, help="Width of rendered output images")
    parser.add_argument('--viz-render-height', type=int, default=720, help="Width of rendered output images")

    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles viz ordering")
    parser.set_defaults(shuffle=False)

    parser.add_argument('--flip-img', dest='flip_img', action='store_true', help="Flips the loaded image about y-axis. This is useful for PROX result.")
    parser.set_defaults(flip_img=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args


def create_pkl(args, person_id):
    print(args.results)
    # print("Current directory: ", os.getcwd())
    # # Change the current working directory
    # os.chdir('../../..')
    # print("Current directory after change: ", os.getcwd())
    # the_list = os.listdir(args.results)

    print(args)
    mkdir(args.out)
    qual_out_path = args.out
    D_IMW, D_IMH = args.viz_render_width, args.viz_render_height

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # collect our results directories
    # all_result_dirs = [os.path.join(args.results, f) for f in sorted(os.listdir(args.results)) if f[0] != '.']
    #
    # print(args.results)
    # print("Current directory: ", os.getcwd())
    # # Change the current working directory
    # os.chdir('../../..')
    # print("Current directory after change: ", os.getcwd())
    # # the_list = os.listdir(args.results)

    all_result_dirs = [args.results + '/' + f for f in sorted(os.listdir(args.results)) if f[0] != '.']
    all_result_dirs = [f for f in all_result_dirs if os.path.isdir(f)]
    if args.shuffle:
        random.seed(0)
        random.shuffle(all_result_dirs)
    print(all_result_dirs)

    seq_name_list = []
    body_model_dict = dict()
    print("all_result_dirs: ", all_result_dirs)

    # Dora's change in order to create pkl file for specific person_id
    for residx, result_dir in enumerate(all_result_dirs):  #This was the dafault code - It runs for all person_ids

        print(result_dir)
        #
        # if not str(person_id) in result_dir:
        #     continue
        seq_name = result_dir.replace('\\', '/').split('/')[-1]
        is_final_res = seq_name == 'final_results'
        if not is_final_res:
            if args.viz_final_only:
                continue
            # seq_name = '_'.join(result_dir.split('/')[-1].split('_')[:-1])
            seq_name = '_/' + result_dir.split('/')[-1].split('_')[0]

        print('Visualizing %s %d / %d...' % (seq_name, residx, len(all_result_dirs)))

        obs_dict = load_res(result_dir, OBS_NAME + '.npz')
        print(result_dir)
        print(OBS_NAME)
        print(obs_dict)
        cur_img_paths = obs_dict['img_paths'] # used to load in results from baselines
        # cur_frame_names = ['.'.join(f.replace('\\', '/').split('/')[-1].split('.')[:-1]) for f in cur_img_paths]
        # cur_frame_names = ['./' + f.split('/')[-1].split('.')[:-1] for f in cur_img_paths]
        cur_frame_names = ['./' + f.split('/')[-1].split('.')[0] for f in cur_img_paths]

        # load in humor prediction
        pred_res = load_res(result_dir, PRED_RES_NAME + '.npz')
        if pred_res is None:
            print('Could not find final pred (stage 3) results for %s, skipping...' % (seq_name))
            continue
        T = pred_res['trans'].shape[0]
        # check if have any nans valid
        for smpk in SMPL_SIZES.keys():
            cur_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res[smpk])))).item() == 0)
            if not cur_valid:
                print('Found NaNs in prediction for %s, filling with zeros...' % (smpk))
                # print(pred_res[smpk].shape)
                if smpk == 'betas':
                    pred_res[smpk] = np.zeros((pred_res[smpk].shape[0]), dtype=np.float)
                else:
                    pred_res[smpk] = np.zeros((T, pred_res[smpk].shape[1]), dtype=np.float)
        floor_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res['floor_plane'])))).item() == 0)
        if not floor_valid:
            print('Predicted floor is NaN, replacing with up.')
            pred_res['floor_plane'] = np.array([0.0, -1.0, 0.0, 0.0])

        pred_res = prep_res(pred_res, device, T)
        num_pred_betas = pred_res['betas'].size(1)

        pred_floor_plane = torch.Tensor(pred_res['floor_plane']).to(device)

        # humor prediction in prior frame
        pred_res_prior = None
        if args.viz_prior_frame:
            pred_res_prior = load_res(result_dir, PRED_PRIOR_RES_NAME + '.npz')
            if pred_res_prior is None:
                    print('Could not find final prior pred (stage 3) results for %s, skipping...' % (seq_name))
                    continue
            pred_res_prior = prep_res(pred_res_prior, device, T)

        # load stages results if needed
        cur_viz_stages = args.viz_stages and not is_final_res
        cur_stages_res = None
        if cur_viz_stages:
            cur_stages_res = dict()
            for stage_name in STAGES_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_res[stage_name] = prep_res(stage_res, device, T)

        # load prior stages results if needed
        cur_stages_prior_res = None
        if args.viz_prior_frame and cur_viz_stages:
            cur_stages_prior_res = dict()
            for stage_name in STAGES_PRIOR_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_prior_res[stage_name] = prep_res(stage_res, device, T)

        #
        # create body models for each
        #
        meta_path = os.path.join(result_dir, 'meta.txt')
        if not os.path.exists(meta_path):
            print('Could not find metadata for %s, skipping...' % (seq_name))
            continue
        optim_bm_path = gt_bm_path = None
        with open(meta_path, 'r') as f:
            optim_bm_str = f.readline().strip()
            optim_bm_path = optim_bm_str.split(' ')[1]
            gt_bm_str = f.readline().strip()
            gt_bm_path = gt_bm_str.split(' ')[1]

        # humor model
        # os.chdir("../")
        pred_bm = None
        if optim_bm_path not in body_model_dict:
            pred_bm = BodyModel(bm_path=optim_bm_path,
                            num_betas=num_pred_betas,
                            batch_size=T).to(device)
            if not is_final_res:
                # final results will be different length, so want to re-load for subsequences
                body_model_dict[optim_bm_path] = pred_bm
        if not is_final_res:
            pred_bm = body_model_dict[optim_bm_path]

        # we are using this sequence for sure
        seq_name_list.append(seq_name)

        # run through SMPL
        pred_body = run_smpl(pred_res, pred_bm)

        # convert tensor to ndarray
        pred_pose = pred_body.full_pose.cpu().detach().numpy()
        pred_verts = pred_body.v.cpu().detach().numpy()
        pred_betas = pred_body.betas.cpu().detach().numpy()
        pred_joints3d = pred_body.Jtr.cpu().detach().numpy()

        print("Jtr size: ", pred_body.Jtr.size())
        print("betas size: ", pred_body.betas.size())
        print("f size: ", pred_body.f.size())
        print("full_pose size: ", pred_body.full_pose.size())
        print("pose_body size: ", pred_body.pose_body.size())
        print("pose_hand size: ", pred_body.pose_hand.size())
        print("v size: ", pred_body.v.size())

        # data = np.load("../../out/rgb_demo_no_split/results_out/person_" + str(person_id) + "/final_results/stage3_results_prior.npz")
        print("New current dir: ", os.getcwd())
        # os.chdir("/home/tpistola/Projects/Chromata/Services_v1/Pose-Estimation_v2/humor_main/")
        os.chdir("C:/Chromata/Services_v1/Pose-Estimation_v2/humor_main/")
        print("CHECK THIS --> Results: ", args.results)
        data = np.load("C:/Chromata/Services_v1/Pose-Estimation_v2/" + args.results + "/final_results/stage3_results_prior.npz")
        trans = data.f.trans
        root_orient = data.f.root_orient

        output_dict = {
            'verts': pred_verts,
            'pose': pred_pose[:, 0:72],
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'trans': trans,
            'root_orient': root_orient
        }

        humor_results = {}
        humor_results[1] = output_dict

        # write output_dict in pkl file
        # output_path = "../../out/"
        output_path = args.results
        print(f'Saving output results to \"{os.path.join(output_path, "humor_output_person_" + str(person_id) + ".pkl")}\".')
        print("CHECK THIS --> ", os.path.join(output_path, "humor_output_person_" + str(person_id) + ".pkl"))
        # joblib.dump(humor_results, os.path.join(output_path, "humor_output_person_" + str(person_id) + ".pkl"))
        joblib.dump(humor_results, os.path.join("C:/Chromata/Services_v1/Pose-Estimation_v2/", output_path, "humor_output_person_" + str(person_id) + ".pkl"))

        print("Ok")



        # my_tensor = pred_body.v
        # tsize = my_tensor.size()
        #
        # faces = c2c(pred_body.f)
        # mesh_list = [trimesh.Trimesh(vertices=c2c(pred_body.v[i]), faces=faces, process=False) for i in range(pred_body.v.size(0))]
        #
        # mesh = trimesh.util.concatenate(mesh_list)
        #
        # # for t in range(tsize[0]):
        # #     verts = my_tensor[t].cpu().detach().numpy()
        # #     faces = pred_body.f.cpu().detach().numpy()
        # #
        # #     mesh = trimesh.Trimesh(vertices=verts,
        # #                            faces=faces,
        # #                            process=False,
        # #                            maintain_order=True)
        # mesh_fname = 'my_mesh_new_new.obj'
        # mesh.export(mesh_fname)
        #
        # print("OK")

        # output_dict = {
        #     'pred_cam': pred_cam,
        #     'orig_cam': orig_cam,
        #     'verts': pred_verts,
        #     'pose': pred_pose,
        #     'betas': pred_betas,
        #     'joints3d': pred_joints3d,
        #     'joints2d': joints2d,
        #     'bboxes': bboxes,
        #     'frame_ids': frames,
        # }


if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    person_id = args.person_id
    create_pkl(args, person_id)