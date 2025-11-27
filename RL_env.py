import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


# ============================================================
# Reward function using correct state indices
# ============================================================

def walking_reward_gt(obs):
    torso_z = obs[..., 2]
    torso_angle = obs[..., 1]
    forward_vel = obs[..., 8]  # qvel[rootx_vel]

    # Upright reward
    upright = (np.cos(torso_angle) + 1) / 2

    # Standing reward
    standing = np.clip(1 - np.abs(torso_z), 0, 1)
    standing_reward = (3 * standing + upright) / 4

    # Move forward
    move_reward = np.clip(forward_vel / 1.0, 0.0, 1.0)
    reward = standing_reward * move_reward

    return float(reward)

class MuscleWalkerEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, xml_path="assets/walker/muscle_new.xml", frame_skip=10, render_mode=None):
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Viewer for human render
        self.viewer = None

        # Action space = actuator muscle ctrl ranges
        low = 0.0
        high = 1.0
        low = np.array([low] * self.model.nu)
        high = np.array([high] * self.model.nu)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Compute observation to know its shape
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        # qpos dimension AFTER removing rootx
        self.qpos_dim = self.model.nq - 1

    # ---------------------------
    # Helper: rootx qpos index
    # ---------------------------
    @property
    def _rootx_id(self):
        return self.model.joint("rootx").qposadr

    @property
    def _rootz_id(self):
        return self.model.joint("rootz").qposadr

    # ---------------------------
    # Reset
    # ---------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    # ---------------------------
    # Step
    # ---------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Compute reward using corrected indices
        reward = walking_reward_gt(obs)

        # Terminate if fallen
        terminated = self._is_fallen()
        # print("Terminated:", type(terminated))  # Debugging line
        # print(terminated)
        return obs, reward, terminated, False, {}

    # ---------------------------
    # Observation construction
    # ---------------------------
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Remove rootx
        qpos = np.delete(qpos, self._rootx_id)

        # Save qpos dimension after removal
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        # print("Observation:", obs.shape)  # Debugging line
        return obs

    # ---------------------------
    # Termination
    # ---------------------------
    def _is_fallen(self):
        torso_z = self.data.qpos[self._rootz_id[0]]
        return bool(torso_z < -0.7)

    # ---------------------------
    # Rendering
    # ---------------------------
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            width, height = 640, 480
            img = mujoco.mjr_render(self.model, self.data, width, height)
            return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None