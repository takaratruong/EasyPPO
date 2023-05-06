import collections

import numpy as np
from gym import utils
from env.mujoco_env import MujocoEnv
from gym.spaces import Box
import ipdb
from scipy.interpolate import interp1d
import copy
import os

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class Skeleton(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": -1,  # redefined
    }

    def __init__(self, healthy_y_range=(.5, 20.0), config=None, **kwargs):
        utils.EzPickle.__init__(self, healthy_y_range, config, **kwargs)
        self._healthy_y_range = healthy_y_range
        args = config['env']

        self.train = config['train']
        self.agent_obs_size = config['policy']['obs_size']

        self.xml_file = args['xml_file']
        self.ref_path = args['ref_path']
        self.frame_skip = args['frame_skip']
        self.max_ep_time = args['max_ep_time']

        # Load References
        self.pos_ref, self.vel_ref, self.torque_ref = self.load_references()
        self.initial_phase_offset = None

        # Mujoco Environment
        MujocoEnv.__init__(self, self.xml_file, self.frame_skip, observation_space=Box(low=-np.inf, high=np.inf, shape=(self.agent_obs_size,), dtype=np.float64), **kwargs)

    def load_references(self):
        pos_ref, vel_ref, torque_ref, grf_ref = None, None, None, None

        if os.path.exists(self.ref_path + '/pos_ref.npy'):
            pos_ref = np.load(self.ref_path + '/pos_ref.npy')
            self.ref_time = pos_ref.shape[0] * 0.01  # Fixed timestep in data
            pos_ref = interp1d(np.arange(0, pos_ref.shape[0]) / (pos_ref.shape[0] - 1), pos_ref, axis=0)
        else:
            print("Position reference file does not exist")

        if os.path.exists(self.ref_path + '/vel_ref.npy'):
            vel_ref = np.load(self.ref_path + '/vel_ref.npy')
            vel_ref = interp1d(np.arange(0, vel_ref.shape[0]) / (vel_ref.shape[0] - 1), vel_ref, axis=0)

        return pos_ref, vel_ref, torque_ref

    @property
    def terminated(self):

        joint_pos_err = np.sum((self.data.qpos[6:-1] - self.pos_ref(self.phase)[6:-1]) ** 2)
        joint_pos_reward = np.exp(-0.5 * joint_pos_err)

        root_rot_error = np.sum((self.data.qpos[3:6] - self.pos_ref(self.phase)[3:6]) ** 2)
        root_rot_reward = np.exp(-1.0 * root_rot_error)

        root_pos_error = np.sum((self.data.qpos[0:3] - self.pos_ref(self.phase)[0:3]) ** 2)
        root_pos_reward = np.exp(-1 * root_pos_error)

        terminated = root_pos_reward < .3 or root_rot_reward < 0.3 or joint_pos_reward < 0.3

        return terminated

    @property
    def done(self):
        done = self.phase >= 1 * .99
        return done

    @property
    def phase(self):
        initial_offset_time = self.initial_phase_offset * self.ref_time
        total_time = initial_offset_time + self.data.time
        return (total_time % self.ref_time) / self.ref_time

    def get_target_refs(self):
        frame_skip_time = self.frame_skip * self.model.opt.timestep
        initial_offset_time = self.initial_phase_offset * self.ref_time
        total_time = frame_skip_time + initial_offset_time + self.data.time
        phase_target = (total_time % self.ref_time) / self.ref_time

        # Get target references
        target_pos_ref = self.pos_ref(phase_target)
        target_vel_ref = self.vel_ref(phase_target) if self.vel_ref is not None else None

        return target_pos_ref, target_vel_ref

    def _get_obs(self):

        exlude_idxs = [0, 6, 11, 16, 20, 21]  # world, toes_r, toes_l, hand_r, hand_l, treadmill
        xipos = np.delete(self.data.xipos.copy(), exlude_idxs, axis=0)
        ximat = np.delete(self.data.ximat.copy(), exlude_idxs, axis=0)
        cvel = np.delete(self.data.cvel.copy(), exlude_idxs, axis=0)

        target_pos_ref, target_vel_ref = self.get_target_refs()

        observation = np.hstack((xipos.flatten(), ximat.flatten(), cvel.flatten(), target_pos_ref[6:-1]))  # 311

        return observation

    def get_obs(self):
        return self._get_obs()

    def calc_reward(self, pos_ref, vel_ref):

        joint_pos_err = np.sum((self.data.qpos[6:-1] - pos_ref[6:-1]) ** 2)
        joint_pos_reward = np.exp(-5.0 * joint_pos_err)

        joint_vel_err = np.sum((self.data.qvel[6:-1] - vel_ref[6:-1]) ** 2)
        joint_vel_reward = np.exp(-.05 * joint_vel_err)  # for walk

        # Root Rewards
        root_pos_error = np.sum((self.data.qpos[0:3] - pos_ref[0:3]) ** 2)
        root_pos_reward = np.exp(-1 * root_pos_error)

        root_vel_error = np.sum((self.data.qvel[0:3] - vel_ref[0:3]) ** 2)
        root_vel_reward = np.exp(-1.0 * root_vel_error)

        root_rot_error = np.sum((self.data.qpos[3:6] - pos_ref[3:6]) ** 2)
        root_rot_reward = np.exp(-1.0 * root_rot_error)

        root_rot_vel_error = np.sum((self.data.qvel[3:6] - vel_ref[3:6]) ** 2)
        root_rot_vel_reward = np.exp(-0.1 * root_rot_vel_error)

        root_reward = root_pos_reward * root_vel_reward * root_rot_reward * root_rot_vel_reward

        reward = joint_pos_reward * joint_vel_reward * root_reward * .1

        return reward


    def step(self, action):
        action = np.array(action.tolist()).copy()

        target_pos_ref, target_vel_ref = self.get_target_refs()

        self.pose = target_pos_ref[6:-1].copy() + action

        for k in range(self.frame_skip):
            joint_obs = self.data.qpos[6:-1].copy()
            joint_vel_obs = self.data.qvel[6:-1].copy()

            error = self.pose - joint_obs
            error_der = joint_vel_obs

            torque = 1 * error - 0.05 * error_der

            self.torque = torque.copy()

            self.data.qvel[-1] = -1.25
            self.do_simulation(torque, 1)

            self.renderer.render_step()

        observation = self._get_obs()
        reward = self.calc_reward(target_pos_ref, target_vel_ref)

        terminated = self.terminated
        done = self.done or terminated

        info = {}

        return observation, reward, terminated, done, info


    def reset_model(self):
        self.initial_phase_offset = 0

        qpos = self.pos_ref(self.phase)
        qvel = self.vel_ref(self.phase)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
