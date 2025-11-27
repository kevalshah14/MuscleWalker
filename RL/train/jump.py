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


class MuscleKangarooEnv(gym.Env):
    """
    Custom Gymnasium environment for the Muscle Walker MuJoCo model - trained for kangaroo-like hopping.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        model_path="assets/walker/muscle_new.xml",
        frame_skip=40,
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

        # Observation space: same as walker but track vertical velocity more
        obs_dim = (self.model.nq - 1) + self.model.nv
        if self.model.na > 0:
            obs_dim += self.model.na

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Match MPPI startup
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.viewer = None

        # For hopping: track previous height for reward calculation
        self.prev_height = 0.0
        self.max_height_achieved = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add noise
        joint_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq-3)
        qpos[3:] += joint_noise
        qvel += self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.5

        mujoco.mj_forward(self.model, self.data)

        # Let settle but shorter than walking
        settle_steps = 100
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Initialize hopping tracking
        self.prev_height = self.data.qpos[1]
        self.max_height_achieved = self.data.qpos[1]

        return self._get_obs(), {}

    def step(self, action):
        # Scale action from [-1, 1] to [0, 1] for muscles
        ctrl = 0.5 * (action + 1.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)

        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_hopping_reward(action)
        terminated = self._is_terminated()
        truncated = False

        info = {
            "x_velocity": self.data.qvel[0],
            "z_height": self.data.qpos[1],
            "vertical_velocity": self.data.qvel[1]
        }

        if self.auto_render:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Skip rootx for translation invariance
        qpos = self.data.qpos[1:].copy()
        qvel = self.data.qvel.copy()

        if self.model.na > 0:
            act = self.data.act.copy()
            return np.concatenate([qpos, qvel, act])

        return np.concatenate([qpos, qvel])

    def _get_hopping_reward(self, action):
        height = self.data.qpos[1]
        vertical_vel = self.data.qvel[1]
        forward_vel = self.data.qvel[0]

        # Update max height achieved
        self.max_height_achieved = max(self.max_height_achieved, height)

        # 1. Vertical movement reward - encourage jumping up and down (increased weight)
        vertical_energy = abs(vertical_vel)
        height_reward = max(0, height - self.prev_height) * 8.0  # Increased reward for upward movement
        vertical_reward = vertical_energy * 1.2 + height_reward  # Increased vertical energy multiplier

        # 2. Jump height bonus - reward achieving higher absolute heights
        jump_height_bonus = max(0, height + 0.5) * 3.0  # Reward being higher off ground

        # 3. Forward hopping reward - reduced weight to discourage walking
        forward_reward = np.clip(forward_vel / 2.0, 0.0, 1.0) * 0.5

        # 4. Hopping pattern reward - encourage periodic bouncing
        # Reward when changing from downward to upward motion (start of hop)
        hopping_pattern = 0.0
        if self.prev_height > height and vertical_vel > 0.2:  # Coming up from bottom with higher threshold
            hopping_pattern = 4.0  # Increased reward for bounce detection

        # 5. Air time reward - bonus for being airborne
        air_time_bonus = 2.0 if height > 0.0 else 0.0  # Above ground level, stricter threshold

        # 6. Peak height reward - bonus for reaching new height records
        peak_bonus = 0.0
        if height > self.max_height_achieved * 0.9:  # Near peak height
            peak_bonus = 1.0

        # 7. Leg symmetry reward - encourage both legs working together
        # Muscles: [right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle]
        right_leg = action[:3]  # indices 0-2
        left_leg = action[3:]   # indices 3-5
        symmetry_penalty = np.mean(np.abs(right_leg - left_leg))  # Penalize differences
        symmetry_reward = (1.0 - symmetry_penalty) * 2.5  # Increased reward for similarity

        # 8. Control cost - small penalty for muscle usage
        ctrl_cost = 0.0005 * np.sum(np.square(action))

        # Total reward - adjusted weights to favor bigger hops
        reward = (vertical_reward * 0.4 +
                 jump_height_bonus * 0.15 +
                 forward_reward * 0.05 +
                 hopping_pattern * 0.15 +
                 air_time_bonus * 0.1 +
                 peak_bonus * 0.05 +
                 symmetry_reward * 0.1 -
                 ctrl_cost)

        self.prev_height = height
        return reward

    def _is_terminated(self):
        if not self.terminate_on_fall:
            return False

        height = self.data.qpos[1]
        pitch = self.data.qpos[2]

        # More lenient termination for hopping
        is_fallen = height < -1.0  # Allow deeper crouches for hopping
        is_unbalanced = np.abs(pitch) > 2.0  # More lenient pitch limit

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
    parser = argparse.ArgumentParser(description="Train PPO on the Muscle Kangaroo environment.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer and render every training step.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200k).",
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

    base_env = MuscleKangarooEnv(render_mode=render_mode, auto_render=args.render)
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
        # Adjust for hopping dynamics
        clip_range=0.2,
        n_epochs=10,
    )

    print("Starting kangaroo hopping training...")
    callback = EpisodeLoggerCallback() if (args.render or args.log_episodes) else None
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)

    model.save("../../trained_models/ppo_muscle_kangaroo")
    print("Kangaroo model saved.")

    if args.render:
        print("Training run finished; continuing to render a short rollout...")
        obs, _ = base_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = base_env.step(action)
            if done or truncated:
                obs, _ = base_env.reset()
        base_env.close()
