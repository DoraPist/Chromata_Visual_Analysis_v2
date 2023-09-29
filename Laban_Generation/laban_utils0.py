import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import filters
import cv2
from scipy.spatial.transform import Rotation as R

class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0
        self.selected_joints = ['hips',
                                'neck',
                                'head',
                                'shoulder.L',
                                'upper_arm.L',
                                'forearm.L',
                                'hand.L',
                                'shoulder.R',
                                'upper_arm.R',
                                'forearm.R',
                                'hand.R',
                                'thigh.L',
                                'shin.L',
                                'foot.L',
                                'thigh.R',
                                'shin.R',
                                'foot.R']

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)
        plt.show()

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))
        p[:, [1, 2]] = p[:, [2, 1]]
        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)
        p[:, [1, 2]] = p[:, [2, 1]]
        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None):  # pos
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        # pos=np.array([ 0,  4,  5,  7,  8,  9, 10, 35, 36, 37, 38, 64, 65, 66, 73, 74, 75])
        pos = np.array([1, 2, 3, 6, 7, 8, 13, 14, 16, 17, 18, 19, 23, 24, 25, 26])
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        # ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        # ax.scatter(p[pos, 0]-p[0,0], p[pos, 2]-p[0,2], p[pos, 1]-p[0,1])
        ax.scatter(p[pos, 0], p[pos, 1], p[pos, 2])

        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_zlim(-40, 40)

        # draw limbs

        # POSE_PAIRS = [(1, 2), (1, 3), (1, 7), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9),
        #             (9, 10), (11, 12), (12, 13), (14, 15), (15, 16)]

        POSE_PAIRS = [(1, 2), (2, 3), (6, 7), (7, 8), (13, 14), (16, 17), (17, 18),
                      (18, 19), (23, 24), (24, 25), (25, 26)]
        # p=p/10
        pp = p.copy()
        for i in range(len(POSE_PAIRS)):
            pair = POSE_PAIRS[i]
            plt.plot([pp[pair[0], 0], pp[pair[1], 0]], [pp[pair[0], 1], pp[pair[1], 1]],
                     [pp[pair[0], 2], pp[pair[1], 2]], color='blue')

        cp, v2, cpz = self.body_front_coordinates(p)
        m = (p[12] + p[0]) / 2  # 2,0
        sc = 10
        ax.plot([m[0], sc * cp[0] + m[0]], [m[1], sc * cp[1] + m[1]], [m[2], sc * cp[2] + m[2]])
        ax.plot([m[0], sc * v2[0] + m[0]], [m[1], sc * v2[1] + m[1]], [m[2], sc * v2[2] + m[2]])
        ax.plot([m[0], sc * cpz[0] + m[0]], [m[1], sc * cpz[1] + m[1]], [m[2], sc * cpz[2] + m[2]])

        limbs = limb_vecs(p)
        lab = LabanotationOfLimb(cp, v2, cpz, np.array(limbs['r_down_arm']))
        ax.legend(lab)

        plt.draw()
        plt.pause(0.001)

    def plot_frame(self, frame, fig=None, ax=None):  # pos
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax)  # pos

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def plot_all_frames(self):  # pos
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0, self.frames, 10):
            self.plot_frame(i, fig, ax)  # pos

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"


def limb2body_coordinates(x_body, y_body, z_body, limb_vector, type):
    flag_1d = 0
    if np.ndim(limb_vector) == 1:
        flag_1d = 1
        limb_vector = np.concatenate((limb_vector.reshape((1, -1)), np.array([0, 0, 0]).reshape((1, -1))))
        x_body = np.concatenate((x_body.reshape((1, -1)), np.array([0, 0, 0]).reshape((1, -1))))
        y_body = np.concatenate((y_body.reshape((1, -1)), np.array([0, 0, 0]).reshape((1, -1))))
        z_body = np.concatenate((z_body.reshape((1, -1)), np.array([0, 0, 0]).reshape((1, -1))))

    dot_prodz = (limb_vector @ z_body.transpose()).diagonal()

    lz = dot_prodz.reshape((-1, 1)) * z_body
    dot_prodx = (limb_vector @ x_body.transpose()).diagonal()

    lx = dot_prodx.reshape((-1, 1)) * x_body

    dot_prody = (limb_vector @ y_body.transpose()).diagonal()

    ly = dot_prody.reshape((-1, 1)) * y_body

    # lz = np.dot(limb_vector,z_body) *z_body
    # lx = np.dot(limb_vector,x_body) * x_body
    # ly = np.dot(limb_vector,y_body) *y_body

    sign_x = np.sign(dot_prodx)
    sign_y = np.sign(dot_prody)
    sign_z = np.sign(dot_prodz)

    # sign_x = np.sign(np.dot(limb_vector, x_body))
    # sign_y=np.sign(np.dot(limb_vector,y_body))
    # sign_z = np.sign(np.dot(limb_vector, z_body))

    r = np.linalg.norm(limb_vector, axis=1)
    # r=np.linalg.norm(limb_vector)
    theta = np.rad2deg(np.arctan2(np.sqrt(np.linalg.norm(lx, axis=1) ** 2 + np.linalg.norm(ly, axis=1) ** 2),
                                  sign_z * np.linalg.norm(lz, axis=1)))

    # theta=np.rad2deg(np.arctan2(np.sqrt(np.linalg.norm(lx)**2+np.linalg.norm(ly)**2),(np.dot(limb_vector,z_body)/np.abs(np.dot(limb_vector,z_body)))*np.linalg.norm(lz)))

    phi = np.rad2deg(np.arctan2(sign_y * np.linalg.norm(ly, axis=1), sign_x * np.linalg.norm(lx, axis=1)))

    # phi=np.rad2deg(np.arctan2((np.dot(limb_vector,y_body)/np.abs(np.dot(limb_vector,y_body)))*np.linalg.norm(ly),(np.dot(limb_vector,x_body)/np.abs(np.dot(limb_vector,x_body)))*np.linalg.norm(lx)))

    if type == 'sphere':
        if flag_1d == 1:
            return r[0], phi[0], theta[0]
        return r, phi, theta
    else:
        if flag_1d == 1:
            return lx[0], ly[0], lz[0]
        return lx, ly, lz


def body_front_coordinates(p):
    # p[:, [1, 2]] = p[:, [2, 1]]
    left_up_leg = p[:, 8]
    right_up_leg = p[:, 1]
    spine1 = p[:, 14]
    v1 = spine1 - left_up_leg
    v2 = right_up_leg - left_up_leg
    x_body = v2
    x_body[:, 2] = 0
    # y_body = np.cross(v1, v2)
    z_body = np.zeros((x_body.shape[0], x_body.shape[1]))
    z_body[:, 2] = 1
    y_body = np.cross(z_body, x_body)  # z
    norm_x = np.array([np.linalg.norm(x_body, axis=1), np.linalg.norm(x_body, axis=1),
                       np.linalg.norm(x_body, axis=1)]).transpose()
    norm_y = np.array([np.linalg.norm(y_body, axis=1), np.linalg.norm(y_body, axis=1),
                       np.linalg.norm(y_body, axis=1)]).transpose()
    norm_z = np.array([np.linalg.norm(z_body, axis=1), np.linalg.norm(z_body, axis=1),
                       np.linalg.norm(z_body, axis=1)]).transpose()

    x_body = np.divide(x_body, norm_x)
    y_body = np.divide(y_body, norm_y)
    z_body = np.divide(z_body, norm_z)
    return x_body, y_body, z_body


def angle(x1, x2):
    if len(x1) == 3:
        cos = np.clip((x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2]) / (
                np.sqrt(x1[0] ** 2 + x1[1] ** 2 + x1[2] ** 2) * np.sqrt(x2[0] ** 2 + x2[1] ** 2 + x2[2] ** 2)),
                      -1.0, 1.0)
    else:
        cos = np.clip(
            (x1[0] * x2[0] + x1[1] * x2[1]) / (np.sqrt(x1[0] ** 2 + x1[1] ** 2) * np.sqrt(x2[0] ** 2 + x2[1] ** 2)),
            -1.0, 1.0)

    return np.rad2deg(np.arccos(cos))


def JointEnergy(p):
    up = np.diff(p, axis=0)

    u2p = np.square(up)

    ke = 1 / 2 * (np.sum(u2p, axis=2))

    return ke


def correct_mask_segs(mask, sum_th):
    # mask = mask0.copy()
    flag = 0
    sum = 0
    for i in range(len(mask)):

        if mask[i] == 0:
            if flag == 0:
                start = i
                flag = 1
            sum = sum + 1
            continue
        if flag == 1:
            # print(i-1)
            # if i-1==952:
            # print('sum',sum)
            stop = i - 1

            if sum <= sum_th:
                zero_pos = start + np.ceil((stop - start) / 2).astype(int)
                mask[start:stop + 1] = 1
                mask[zero_pos.astype(int)] = 0
            sum = 0
            flag = 0

    new_mask = np.ones((len(mask)))
    for i in range(len(mask) - 1):
        if mask[i] != mask[i + 1]:
            new_mask[i] = 0
    segments = [0] + list(np.where(new_mask == 0)[0])

    segments_array = np.ones((len(segments) - 1, 2))
    for i in range(len(segments) - 1):
        # print('seg: ',script_segments[i],script_segments[i+1])
        segments_array[i, 0] = segments[i]
        segments_array[i, 1] = segments[i + 1]

    segments_array = segments_array.astype(int)

    return new_mask, segments_array


def motion_segmentation(ke, limb, joints_dict, min_energy_th):
    if limb == 'right_leg':
        x = ke[:, joints_dict['LeftFoot']]
    else:
        x = ke[:, joints_dict['RightFoot']]
    # x = filters.gaussian_filter1d(x, sigma=3)
    # x = x / x.max()
    peaks, _ = find_peaks(-x)

    mask = (x >= min_energy_th).astype(int)

    mask_slide = -mask[1:]

    xoor = mask[:-1] ^ mask_slide
    pts = list(np.where(xoor == -1)[0]) + list(np.where(xoor == 1)[0] + 1)
    pts.sort()

    all_pts = np.arange(0, len(x))
    all_pts[pts] = -1
    dpts = np.where(all_pts != -1)[0]
    pos = np.where(mask[dpts] == 0)[0]
    dpts = dpts[pos]

    min_points = list(peaks)
    min_points.extend(pts)
    min_points = np.setdiff1d(min_points, list(dpts))
    min_points.sort()

    pos = np.where(mask[min_points] == 1)[0]
    mask[min_points] = 0
    mask[min_points[pos] + 1] = 1

    #segments = [0] + list(min_points) + [len(x)]
    #segments = list(np.unique(np.array(segments)))

    #segments_array = np.ones((len(segments) - 1, 2))
    #for i in range(len(segments) - 1):
        # print('seg: ',script_segments[i],script_segments[i+1])
    #    segments_array[i, 0] = segments[i]
    #    segments_array[i, 1] = segments[i + 1]

    #segments_array = segments_array.astype(int)
    mask, segments_array = correct_mask_segs(mask, 16)
    return segments_array, mask



def movement_analysis2(segments, p, ke, limbs, joints_dict, limb_segs, fps, leg_move_threshold):
    pu = np.diff(p, axis=0)
    u_norm = np.linalg.norm(pu, axis=2)

    for i in range(segments.shape[0] - 1):
        # for segment in segments:
        current_segment = segments[i]
      

        current_segment_secs = np.round(current_segment / fps, 1)

        for limb in limbs:

            if limb == 'right_leg':
                gesture = 'right_leg_gesture'
                support = 'right_support'
                this_leg_vec = p[current_segment[1], joints_dict['LeftFoot'], :] - p[current_segment[0],
                                                                                   joints_dict['LeftFoot'], :]
                this_leg_vec[-1]=0
                this_leg_energy = np.median(ke[current_segment[0]:current_segment[1], joints_dict['LeftFoot']])

            else:
                gesture = 'left_leg_gesture'
                support = 'left_support'
                this_leg_vec = p[current_segment[1], joints_dict['RightFoot'], :] - p[current_segment[0],
                                                                                    joints_dict['RightFoot'], :]

                this_leg_vec[-1] = 0
                this_leg_energy = np.median(ke[current_segment[0]:current_segment[1], joints_dict['RightFoot']])

            if this_leg_energy < leg_move_threshold:
                # place
                limb_segs[support][0].append([current_segment_secs[0], current_segment_secs[1]])
                laban_symbol = ''
                limb_segs[support][1].append(laban_symbol)
            else:



            # move
                limb_segs[support][0].append([current_segment_secs[0], current_segment_secs[1]])
                laban_symbol = LabanotationOfLimb(np.array([1, 0, 0]), np.array([0, 1, 0]),
                                                  np.array([0, 0, 1]), this_leg_vec)
                limb_segs[support][1].append(laban_symbol)


    return limb_segs



def LabanotationOfLimb(x_body, y_body, z_body, limb_vector):
    laban = ['place', 'low']
    r, phi, theta = limb2body_coordinates(x_body, y_body, z_body, limb_vector, 'sphere')

    if (phi > -22.5 and phi < 0) or (phi >= 0 and phi <= 22.5):
        laban[0] = 'Right'
    elif phi > 22.5 and phi <= 67.5:
        laban[0] = 'Right Forward'
    elif phi > 67.5 and phi <= 112.5:
        laban[0] = 'Forward'
    elif phi > 112.5 and phi <= 157.5:
        laban[0] = 'Left Forward'
    elif (phi > 157.5 and phi <= 180) or (phi >= -180 and phi <= -157.5):
        laban[0] = 'Left'
    elif phi > -157.5 and phi <= -112.5:
        laban[0] = 'Left Backward'
    elif phi > -112.5 and phi <= -67.5:
        laban[0] = 'Backward'
    elif phi > -67.5 and phi <= -22.5:
        laban[0] = 'Right Backward'

    if theta < 22.5:
        laban = ['Place', 'High']
    elif theta < 67.5:
        laban[1] = 'High'
    elif theta < 112.5:
        laban[1] = 'Normal'
    elif theta < 157.5:
        laban[1] = 'Low'
    else:
        laban = ['Place', 'Low']

    return laban



def place_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]], np.int32)


# def right_forward_symbol(x1, y1, x2, y2):
#    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1 + round((y2 - y1) / 3)], [x1, y1]], np.int32)

def right_forward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1 + round((y2 - y1) / 3)], [x1, y2], [x2, y2], [x2, y1], [x1, y1 + round((y2 - y1) / 3)]],
                    np.int32)

    # def left_forward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1 + round((y2 - y1) / 3)], [x1, y2], [x2, y2], [x2, y1], [x1, y1 + round((y2 - y1) / 3)]],
                    np.int32)


def left_forward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1 + round((y2 - y1) / 3)], [x1, y1]], np.int32)


def right_backward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2 - round((y2 - y1) / 3)], [x2, y2], [x2, y1], [x1, y1]], np.int32)


def left_backward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2], [x2, y2 - round((y2 - y1) / 3)], [x2, y1], [x1, y1]], np.int32)


def rf_forward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1 + round((y2 - y1) / 3)],
                     [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 3)], [x1 + round((x2 - x1) / 2), y1], [x1, y1]],
                    np.int32)


def lf_forward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1 + round((y2 - y1) / 3)], [x1, y2], [x2, y2], [x2, y1], [x1 + round((x2 - x1) / 2), y1],
                     [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 3)], [x1, y1 + round((y2 - y1) / 3)]], np.int32)


def rf_backward_symbol(x1, y1, x2, y2):
    return np.array(
        [[x1, y1], [x1, y2], [x1 + round((x2 - x1) / 2), y2], [x1 + round((x2 - x1) / 2), y2 - round((y2 - y1) / 3)],
         [x2, y2 - round((y2 - y1) / 3)], [x2, y1], [x1, y1]], np.int32)


def lf_backward_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2 - round((y2 - y1) / 3)], [x1 + round((x2 - x1) / 2), y2 - round((y2 - y1) / 3)],
                     [x1 + round((x2 - x1) / 2), y2], [x2, y2], [x2, y1], [x1, y1]], np.int32)


def right_symbol(x1, y1, x2, y2):
    return np.array([[x1, y1], [x1, y2], [x2, y2 - round((y2 - y1) / 2)], [x1, y1]], np.int32)


def left_symbol(x1, y1, x2, y2):
    return np.array([[x2, y1], [x1, y2 - round((y2 - y1) / 2)], [x2, y2], [x2, y1]], np.int32)


def draw_symbol(laban_symbol, limb, script_columns_distance, duration):
    x1 = 0
    y1 = 0
    x2 = script_columns_distance
    y2 = duration
    limb = limb.split('_')[0]

    if laban_symbol[0] == '':
        return 0, 0
    if laban_symbol[0] == 'Place':
        pts = place_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Backward':
        if limb == 'right':
            pts = rf_backward_symbol(x1, y1, x2, y2)
        else:
            pts = lf_backward_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Forward':
        if limb == 'right':
            pts = rf_forward_symbol(x1, y1, x2, y2)
        else:
            pts = lf_forward_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Right Forward':
        pts = right_forward_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Left Forward':
        pts = left_forward_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Right Backward':
        pts = right_backward_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Left Backward':
        pts = left_backward_symbol(x1, y1, x2, y2)


    elif laban_symbol[0] == 'Right':
        pts = right_symbol(x1, y1, x2, y2)

    elif laban_symbol[0] == 'Left':
        pts = left_symbol(x1, y1, x2, y2)

    return 1, pts


def draw_script(script_columns, script_columns_distance, script_height, vertical_margin):
    script_points = []

    # script_column_names = ['left_margin','left_arm','left_body','left_leg_gesture','left_support','right_support', 'right_leg_gesture', 'right_body','right_arm', 'head','right_margin']
    # script_columns = {}
    # for i in range(11):
    # script_columns[script_column_names[i]] = np.array([i*script_columns_distance,i*script_columns_distance+script_columns_distance])

    # print(script_columns)

    script_top = vertical_margin
    script_bottom = script_height - 1 + vertical_margin

    script_points.append(np.array(
        [[script_columns['left_leg_gesture'][0], script_top], [script_columns['right_leg_gesture'][1], script_top],
         [script_columns['right_leg_gesture'][1], script_bottom],
         [script_columns['left_leg_gesture'][0], script_bottom], [script_columns['left_leg_gesture'][0], script_top]],
        np.int32))

    script_points.append(
        np.array([[script_columns['left_support'][1], script_top], [script_columns['left_support'][1], script_bottom]],
                 np.int32))

    return script_points


def draw_symbol_on_script(limb, limb_segs, script_columns, script_height, script_columns_distance, scale):
    circle_points = []
    laban_symbol_points = []

    # ii=0
    for n, duration_interval in enumerate(limb_segs[0]):
        if duration_interval[0] * scale == 0:
            start = 0
        else:

            start = (duration_interval[0] * scale + 1).astype(int)
        stop = (duration_interval[1] * scale).astype(int)

        if limb_segs[1][n] == '':
            flag, pts = draw_symbol([limb_segs[1][n]], limb, script_columns_distance, stop - start)
        else:
            flag, pts = draw_symbol([limb_segs[1][n][0]], limb, script_columns_distance, stop - start)
        if flag == 0:
            # ii=ii+1
            continue
        # pts = pts + np.array([script_columns[limb][0],script_height-stop])
        pts = pts + np.array([script_columns[limb][0], script_height - pts[:, 1].max() - start])

        circle_points.append(np.array(
            [int(script_columns[limb][0] + (script_columns[limb][1] - script_columns[limb][0]) / 2),
             script_height - stop + (stop - start) / 2]).astype(int))
        # circle_points.append(np.array([int(script_columns[limb][0]+(script_columns[limb][1]-script_columns[limb][0])/2),script_height-pts[:,1].max()-stop+(stop-start)/2]).astype(int))
        laban_symbol_points.append(pts.copy())
        # ii=ii+1

    return laban_symbol_points, circle_points


# define script parameters

def laban_script(limb_segs, script_height, script_columns_distance, script_total_width, vertical_margin, scale):
    script_column_names = ['left_margin', 'left_arm', 'left_body', 'left_leg_gesture', 'left_support', 'right_support',
                           'right_leg_gesture', 'right_body', 'right_arm', 'head', 'right_margin']
    script_columns = {}

    for i in range(11):
        script_columns[script_column_names[i]] = np.array(
            [i * script_columns_distance, i * script_columns_distance + script_columns_distance])

    # initialize script image
    script_img = np.ones((script_height + 2 * vertical_margin, script_columns['right_margin'][1], 3),
                         dtype="uint8") * 255

    # draw empty script
    scr_pts = draw_script(script_columns, script_columns_distance, script_height, vertical_margin)

    cv2.polylines(script_img, scr_pts, False, (0, 0, 0), thickness=1)
    # cv2_imshow(script_img)

    for limb in limb_segs.keys():
        # draw laban symbols on script
        # laban_symbols, symbol_dots = draw_symbol_on_script('right_arm', limb_segs['right_arm'], script_height)
        laban_symbols, symbol_dots = draw_symbol_on_script(limb, limb_segs[limb], script_columns,
                                                           script_height + vertical_margin, script_columns_distance,
                                                           scale)

        cv2.polylines(script_img, laban_symbols, False, (0, 0, 0), thickness=1)

        for dot_point in symbol_dots:
            cv2.circle(script_img, tuple(dot_point), radius=0, color=(0, 0, 0), thickness=2)

    return script_img










def labanotation(file):
    # joints = ['LeftUpLeg','LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot', 'Neck', 'Head', 'LeftShoulder','LeftArm',
    #   'LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand']

    anim = Bvh()
    anim.parse_file(file)
    p, r = anim.all_frame_poses()

    if anim.fps > 30:
        p = p[np.arange(0, p.shape[0], anim.fps // 30), :, :]

    # p -= np.amin(p, axis=(0, 1))
    # p /= np.amax(p, axis=(0, 1))

    joints_dict = {}
    for i, j in enumerate(anim.joints.keys()):
        j = j.split(':')[1]
        joints_dict[j] = i

    ke = JointEnergy(p)
    ke = filters.gaussian_filter1d(ke, sigma=3, axis=0)
    ke = ke / ke.max(axis=0)

    x, y, z = body_front_coordinates(p)

    pn = p - p[:, 0:1, :]
    x0 = [1, 0, 0]
    y0 = [0, 1, 0]
    z0 = [0, 0, 1]



    pnn = pn.copy()
    for i in range(pn.shape[0]):
        ang = angle(x0, x[i])
        ang_sign = - np.sign(x[i][1])
        r = R.from_euler('z', ang_sign * ang, degrees=True)
        pnn[i] = r.apply(pn[i])

    limbs = ['right_leg', 'left_leg']
    limb_segs = {'right_support': [[], []], 'left_support': [[], []], 'right_leg_gesture': [[], []],
                 'left_leg_gesture': [[], []]}
    ke_th = 0.06
    for limb in limbs:
        segments, mask = motion_segmentation(ke, limb, joints_dict, 0.06)

        limb_segs = movement_analysis2(segments, pnn, ke, [limb], joints_dict, limb_segs, 30, 0.06)
    scale = 60
    script_height = np.ceil(p.shape[0] / 30).astype(int) * scale
    script_columns_distance = 20
    script_total_width = 20 * 11
    vertical_margin = 2

    script_img = laban_script(limb_segs, script_height, script_columns_distance, script_total_width, vertical_margin,
                              scale)

    return script_img

