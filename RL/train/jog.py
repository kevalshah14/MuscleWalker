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


class MuscleJoggerEnv(gym.Env):
    """
    Custom Gymnasium environment for the Muscle Walker MuJoCo model - trained for jogging.
    Jogging involves faster, more dynamic locomotion with rhythmic leg alternation.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        model_path="assets/walker/muscle_new.xml",
        frame_skip=10,  # Same as walking for stability, but faster targets
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

        # Extended observation space: add temporal memory for rhythm
        base_obs_dim = (self.model.nq - 1) + self.model.nv
        if self.model.na > 0:
            base_obs_dim += self.model.na

        # Add temporal features: previous actions, velocities, phase
        temporal_dim = 6 + 2 + 1  # prev_actions (6) + prev_vels (2) + phase (1)
        obs_dim = base_obs_dim + temporal_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Match MPPI startup
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.viewer = None

        # Temporal memory for jogging rhythm
        self.prev_action = np.zeros(6)
        self.prev_forward_vel = 0.0
        self.prev_vertical_vel = 0.0
        self.step_count = 0
        self.jogging_phase = 0.0  # Track jogging rhythm phase

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

        # Let settle - same as walking for consistency
        settle_steps = 200
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Reset temporal memory
        self.prev_action = np.zeros(6)
        self.prev_forward_vel = self.data.qvel[0]
        self.prev_vertical_vel = self.data.qvel[1]
        self.step_count = 0
        self.jogging_phase = 0.0

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
        reward = self._get_jogging_reward(action)
        terminated = self._is_terminated()
        truncated = False

        # Update temporal memory
        self.prev_action = action.copy()
        self.prev_forward_vel = self.data.qvel[0]
        self.prev_vertical_vel = self.data.qvel[1]
        self.step_count += 1
        # Update jogging phase based on leg alternation pattern
        left_power = np.mean([abs(action[0]), abs(action[1]), abs(action[2])])
        right_power = np.mean([abs(action[3]), abs(action[4]), abs(action[5])])
        self.jogging_phase = (self.jogging_phase + (left_power - right_power) * 0.1) % (2 * np.pi)

        info = {
            "x_velocity": self.data.qvel[0],
            "z_height": self.data.qpos[1],
            "jogging_phase": self.jogging_phase
        }

        if self.auto_render:
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Skip rootx for translation invariance
        qpos = self.data.qpos[1:].copy()
        qvel = self.data.qvel.copy()

        base_obs = []
        if self.model.na > 0:
            act = self.data.act.copy()
            base_obs = [qpos, qvel, act]
        else:
            base_obs = [qpos, qvel]

        # Add temporal features for jogging rhythm
        temporal_features = np.concatenate([
            self.prev_action,                          # 6 dims: previous muscle actions
            [self.prev_forward_vel, self.prev_vertical_vel],  # 2 dims: previous velocities
            [self.jogging_phase]                      # 1 dim: jogging rhythm phase
        ])

        return np.concatenate(base_obs + [temporal_features])

    def _get_jogging_reward(self, action):
        """
        Simplified jogging reward - EXACTLY like walking but with higher speed target
        This ensures the agent learns to stand/walk first, then speed up naturally
        """

        height = self.data.qpos[1]
        pitch = self.data.qpos[2]
        forward_vel = self.data.qvel[0]

        # ===== EXACT WALKING REWARD STRUCTURE =====

        # 1. Upright reward (maintain pitch near 0)
        upright = (np.cos(pitch) + 1) / 2.0

        # 2. Standing height reward
        standing = np.clip(1.0 - 1.0 * np.abs(height), 0.0, 1.0)

        # 3. Combined standing reward (like MPPI/walking)
        standing_reward = (3.0 * standing + upright) / 4.0

        # 4. Forward movement reward - HIGHER TARGET for jogging
        # Walking uses 1.0, jogging uses 1.5
        move_reward = np.clip(forward_vel / 1.5, 0.0, 1.0)

        # 5. Multiplicative reward structure (EXACTLY like walking)
        reward = standing_reward * move_reward

        # 6. Control cost (same as walking)
        ctrl_cost = 0.005 * np.sum(np.square(action))

        reward = reward - ctrl_cost

        return reward

    def _is_terminated(self):
        if not self.terminate_on_fall:
            return False

        height = self.data.qpos[1]
        pitch = self.data.qpos[2]
        forward_vel = self.data.qvel[0]

        # Fall detection - same as walking for consistency
        is_fallen = height < -0.5

        # Balance detection - same as walking
        is_unbalanced = np.abs(pitch) > 1.2

        # Don't terminate for backward movement during learning
        # is_stuck = forward_vel < -0.1  # REMOVED - too harsh during early training

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
    parser = argparse.ArgumentParser(description="Train PPO on the Muscle Jogger environment.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer and render every training step.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,  # Moderate timesteps for jogging (between walking and jumping)
        help="Total training timesteps (default: 300k).",
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

    base_env = MuscleJoggerEnv(render_mode=render_mode, auto_render=args.render)
    env = Monitor(base_env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="../../tensorboard_logs/",
        learning_rate=3e-4,  # Standard learning rate (like walking)
        n_steps=2048,        # Standard batch size
        batch_size=64,       # Standard batch size
        gamma=0.99,          # Standard discount factor
        ent_coef=0.01,       # Standard entropy for exploration
        # Standard PPO settings work well with simplified reward
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
    )

    print("Starting jogging training...")
    callback = EpisodeLoggerCallback() if (args.render or args.log_episodes) else None
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)

    model.save("../../trained_models/ppo_muscle_jogger")
    print("Jogger model saved.")

    if args.render:
        print("Training run finished; continuing to render a short rollout...")
        obs, _ = base_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = base_env.step(action)
            if done or truncated:
                obs, _ = base_env.reset()
        base_env.close()
