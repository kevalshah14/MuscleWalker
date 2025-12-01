import argparse
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


class MuscleWalkerEnv(gym.Env):
    """
    Custom Gymnasium environment for the Muscle Walker MuJoCo model.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        model_path="assets/walker/muscle_new.xml",
        frame_skip=10,
        render_mode=None,
        terminate_on_fall=True,
        auto_render=False,
    ):
        super().__init__()

        # Load MuJoCo model
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "../../", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.terminate_on_fall = terminate_on_fall
        self.auto_render = auto_render

        # Action space: 6 muscles, range [0, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Observation space:
        # qpos (excluding rootx to be translation invariant) + qvel
        # qpos: 9 dims (rootx, rootz, rooty, 6 leg joints)
        # qvel: 9 dims
        # Total obs dim = 8 + 9 = 17

        # Match MPPI startup: reset to model defaults and cache that nominal pose
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        obs_dim = (self.model.nq - 1) + self.model.nv

        # Add actuator activations if they exist (muscles usually have internal states)
        # But standard muscle model in MuJoCo might be stateless if no activation dynamics
        # Checking model.na (number of activation states)
        if self.model.na > 0:
            obs_dim += self.model.na

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset sim just like MPPI
        mujoco.mj_resetData(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add very small noise only to joint coordinates (skip root index 0-2)
        joint_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq-3)
        qpos[3:] += joint_noise
        qvel += self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

        # Neutral activation (0.5) mimics the zero-control point used by MPPI before command()
        self.data.ctrl[:] = 0.5

        mujoco.mj_forward(self.model, self.data)

        # Let the robot settle under gravity so it starts on the ground (mirrors MPPI play loop)
        settle_steps = 200
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # Scale action from [-1, 1] to [0, 1] for muscles
        # 0.5 * (action + 1)
        ctrl = 0.5 * (action + 1.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)

        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Calculate reward
        reward = self._get_reward(action)

        # Check termination
        terminated = self._is_terminated()
        truncated = False # Can be handled by TimeLimit wrapper

        info = {
            "x_velocity": self.data.qvel[0],
            "z_height": self.data.qpos[1]
        }

        if self.auto_render:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Skip rootx (index 0) for translation invariance
        qpos = self.data.qpos[1:].copy()
        qvel = self.data.qvel.copy()

        if self.model.na > 0:
            act = self.data.act.copy()
            return np.concatenate([qpos, qvel, act])

        return np.concatenate([qpos, qvel])

    def _get_reward(self, action):
        # Reward function using MPPI logic (multiplicative structure)

        # Indices based on XML structure:
        # 0: rootx (forward)
        # 1: rootz (height)
        # 2: rooty (pitch)
        # 3: right_hip, 4: right_knee, 5: right_ankle
        # 6: left_hip, 7: left_knee, 8: left_ankle

        # 1. Upright reward (maintain pitch near 0)
        # Using cos(pitch) like in MPPI
        pitch = self.data.qpos[2]
        upright = (np.cos(pitch) + 1) / 2.0

        # 2. Standing height reward
        # MPPI uses: standing = clip(1 - 1.0*abs(state[2]), 0, 1)
        height = self.data.qpos[1]
        standing = np.clip(1.0 - 1.0 * np.abs(height), 0.0, 1.0)

        # 3. Joint posture rewards - encourage natural walking posture
        # Based on user feedback: control=1 should give straight posture
        # Since control=1 activates flexors which straighten knees but flex hips,
        # we need to reward the corresponding joint angles

        # For knees: control=1 gives straight knees (-90°), so reward straight knees
        right_knee = self.data.qpos[4]
        left_knee = self.data.qpos[7]
        right_knee_posture = np.clip((right_knee + 90) / 90, 0.0, 1.0)  # 1 when straight
        left_knee_posture = np.clip((left_knee + 90) / 90, 0.0, 1.0)    # 1 when straight
        knee_posture = (right_knee_posture + left_knee_posture) / 2.0

        # For hips: control=1 gives flexed hips (~85°), but user wants control=1 = straight
        # Since mechanical mapping is opposite, we need to invert the hip reward
        # Hip range 0-90°, where 0° = straight. So reward smaller hip angles.
        right_hip = self.data.qpos[3]
        left_hip = self.data.qpos[6]
        right_hip_straight = np.clip(1.0 - (right_hip / (np.pi/2)), 0.0, 1.0)  # 1 when straight (0°)
        left_hip_straight = np.clip(1.0 - (left_hip / (np.pi/2)), 0.0, 1.0)    # 1 when straight (0°)
        hip_posture = (right_hip_straight + left_hip_straight) / 2.0

        # 4. Combined standing reward (like MPPI) - now includes posture
        # standing_reward = (3*standing + upright + knee_posture + hip_posture)/6
        standing_reward = (2.0 * standing + upright + knee_posture + hip_posture) / 5.0

        # 5. Forward movement reward
        # MPPI uses: move_reward = clip(state[10]/1.0, 0.0, 1)
        # state[10] was likely forward velocity in their state representation
        forward_vel = self.data.qvel[0]
        move_reward = np.clip(forward_vel / 1.0, 0.0, 1.0)

        # 6. Multiplicative reward structure (EXACTLY like MPPI)
        reward = standing_reward * move_reward

        return reward

    def _is_terminated(self):
        if not self.terminate_on_fall:
            return False
        # Terminate if falling over

        height = self.data.qpos[1]
        pitch = self.data.qpos[2]

        # Fall detection - robot settles to ~ -0.33 height
        is_fallen = height < -0.5

        # Extreme pitch detection
        is_unbalanced = np.abs(pitch) > 1.2  # > ~69 degrees (more lenient)

        return is_fallen or is_unbalanced

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class EpisodeLoggerCallback(BaseCallback):
    """Print reward and length for every finished episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                episode = info.get("episode")
                if episode:
                    reward = episode.get("r", 0.0)
                    length = episode.get("l", 0)
                    print(f"[Episode {self.episode_count}] reward={reward:.3f}, length={length}")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the Muscle Walker environment.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer and render every training step.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100k).",
    )
    parser.add_argument(
        "--log-episodes",
        action="store_true",
        help="Print reward/length for every finished episode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    render_mode = "human" if args.render else None

    base_env = MuscleWalkerEnv(render_mode=render_mode, auto_render=args.render)
    env = Monitor(base_env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="../../tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
    )

    print("Starting walking training...")
    callback = EpisodeLoggerCallback() if (args.render or args.log_episodes) else None
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)

    model.save("../../trained_models/ppo_muscle_walker")
    print("Walker model saved.")

    if args.render:
        print("Training run finished; continuing to render a short rollout...")
        obs, _ = base_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = base_env.step(action)
            if done or truncated:
                obs, _ = base_env.reset()
        base_env.close()
