# import blenderpy

# input_path = "output/"
# fbx_file = "humor_output_person_1_gamos_female1.fbx"
# bvh_file_basename = "humor_output_person_1_gamos_female1"

import bpy
import numpy as np
from os import listdir, path


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


# if __name__ == '__main__':
#     # data_path = "./fbx/"
#     # data_path = input_path + fbx_file
#     fbx2bvh(input_path, fbx_file)
