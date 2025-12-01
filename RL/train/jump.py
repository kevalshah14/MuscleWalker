import argparse
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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

        # Action space: 6 muscles, PPO outputs in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation space: rootx removed for translation invariance
        obs_dim = (self.model.nq - 1) + self.model.nv
        if self.model.na > 0:
            obs_dim += self.model.na

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Initialize state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.viewer = None

        # Hopping-related tracking
        self.prev_height = 0.0
        self.max_height_achieved = 0.0

        self.current_action = np.zeros(self.model.nu, dtype=np.float64)

        # Track previous stuff for reward
        self.prev_x = 0.0
        self.prev_any_contact = True
        self.prev_left_contact = True
        self.prev_right_contact = True
        self.prev_ctrl = np.zeros(self.model.nu, dtype=np.float64)

        # Cache geom ids for contact checks
        self.left_foot_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot"
        )
        self.right_foot_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot"
        )
        self.floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if (
            self.left_foot_geom < 0
            or self.right_foot_geom < 0
            or self.floor_geom < 0
        ):
            raise ValueError(
                "Could not find 'left_foot', 'right_foot', or 'floor' geoms in the model."
            )

        # Cache torso body id to get world position/velocity
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        if self.torso_body_id < 0:
            raise ValueError("Body 'torso' not found in model.")

        # Reward weights (tune as needed)
        self.w_forward = 5.0          # forward progress
        self.w_height = 1.0           # stay tall
        self.w_posture = 0.5          # penalize torso pitch
        self.w_flight = 0.5           # being in the air
        self.w_takeoff = 1.0          # clean takeoff
        self.w_stance = 0.05          # small penalty for staying on ground
        self.w_contact_sync = 0.3     # both feet same contact state
        self.w_action_sym = 0.3       # symmetric muscle use
        self.w_energy = 0.05          # muscle effort
        self.w_smooth = 0.1           # changes in muscle activations

        # Height & posture targets
        self.desired_min_height = 1.0   # encourage staying above this
        self.max_height_for_bonus = 1.6 # cap height bonus
        self.max_forward_step = 0.25    # clip crazy progress

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add small noise to joints (not to rootx/rootz/rooty)
        if self.model.nq > 3:
            joint_noise = self.np_random.uniform(
                low=-0.01, high=0.01, size=self.model.nq - 3
            )
            qpos[3:] += joint_noise
        qvel += self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.5

        mujoco.mj_forward(self.model, self.data)

        # Let it settle
        settle_steps = 100
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

        # Initialize tracking using *world* torso position
        torso_pos = self.data.xpos[self.torso_body_id]
        self.prev_height = torso_pos[2]
        self.max_height_achieved = torso_pos[2]
        self.prev_x = torso_pos[0]

        left_c, right_c = self._get_foot_contacts()
        self.prev_left_contact = left_c
        self.prev_right_contact = right_c
        self.prev_any_contact = left_c or right_c
        self.prev_ctrl[:] = self.data.ctrl[:]

        self.current_action = np.zeros(self.model.nu, dtype=np.float64)

        return self._get_obs(), {}

    def step(self, action):
        # Store current action (PPO outputs in [-1, 1])
        self.current_action = action

        # Scale action from [-1, 1] to [0, 1] for muscles
        ctrl = 0.5 * (action + 1.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_hopping_reward()
        terminated = self._is_terminated()
        truncated = False

        torso_pos = self.data.xpos[self.torso_body_id]
        torso_vel = self.data.subtree_linvel[self.torso_body_id]

        info = {
            "x_velocity": float(torso_vel[0]),
            "z_height": float(torso_pos[2]),
            "vertical_velocity": float(torso_vel[2]),
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

    # ---------- CONTACTS ----------

    def _get_foot_contacts(self):
        """
        Check if left/right foot are in contact with the floor.
        """
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2

            # left foot vs floor
            if (
                (g1 == self.left_foot_geom and g2 == self.floor_geom)
                or (g2 == self.left_foot_geom and g1 == self.floor_geom)
            ):
                left_contact = True

            # right foot vs floor
            if (
                (g1 == self.right_foot_geom and g2 == self.floor_geom)
                or (g2 == self.right_foot_geom and g1 == self.floor_geom)
            ):
                right_contact = True

        return left_contact, right_contact

    # ---------- REWARD ----------

    def _get_hopping_reward(self):
        """
        Reward designed for kangaroo-like hopping:
        - Move forward
        - Stay upright and tall
        - Periodic hops with clear flight phases
        - Legs in phase (both on/off together)
        - Symmetric muscle activations
        - Energy & smoothness penalties
        """
        # World pose & velocity of torso
        torso_pos = self.data.xpos[self.torso_body_id]
        torso_vel = self.data.subtree_linvel[self.torso_body_id]

        x = float(torso_pos[0])   # forward position
        z = float(torso_pos[2])   # height
        vx = float(torso_vel[0])  # forward velocity
        vz = float(torso_vel[2])  # vertical velocity

        # Pitch angle still comes from qpos (rooty joint)
        pitch = float(self.data.qpos[2])

        # Forward progress (clip large jumps to keep reward smooth)
        forward_progress = x - self.prev_x
        self.prev_x = x
        forward_progress = np.clip(forward_progress, 0.0, self.max_forward_step)
        r_forward = self.w_forward * forward_progress

        # Stay upright & tall
        height_bonus = np.clip(
            z - self.desired_min_height, 0.0, self.max_height_for_bonus - self.desired_min_height
        )
        posture_penalty = pitch ** 2  # rad^2
        r_posture = self.w_height * height_bonus - self.w_posture * posture_penalty

        # ---- Contacts & hopping ----
        left_contact, right_contact = self._get_foot_contacts()
        any_contact = left_contact or right_contact
        no_contact = not any_contact

        just_takeoff = self.prev_any_contact and no_contact
        just_landing = (not self.prev_any_contact) and any_contact  # currently unused, but could be rewarded
        self.prev_any_contact = any_contact

        # Reward flight phases (both feet off ground)
        r_flight = self.w_flight * float(no_contact)

        # Bonus at takeoff proportional to positive vertical velocity
        if just_takeoff:
            pos_vz = np.clip(vz, 0.0, 4.0)
            r_takeoff = self.w_takeoff * pos_vz
        else:
            r_takeoff = 0.0

        # Small penalty for being in stance (encourages rhythmic bouncing)
        r_stance = -self.w_stance * float(any_contact)

        r_hop = r_flight + r_takeoff + r_stance

        # ---- Leg synchronization ----
        # Encourage both feet to be in the same contact state (hopping, not walking)
        contact_sync = 1.0 if left_contact == right_contact else -1.0
        r_contact_sync = self.w_contact_sync * contact_sync

        # Symmetric muscle activations (left vs right)
        ctrl = self.data.ctrl.copy()  # in [0, 1]

        # hip/knee/ankle: right (0,1,2), left (3,4,5)
        hip_sym = 1.0 - abs(ctrl[0] - ctrl[3])
        knee_sym = 1.0 - abs(ctrl[1] - ctrl[4])
        ankle_sym = 1.0 - abs(ctrl[2] - ctrl[5])

        sym_mean = (hip_sym + knee_sym + ankle_sym) / 3.0
        r_action_sym = self.w_action_sym * sym_mean

        # ---- Energy & smoothness penalties ----
        energy_cost = self.w_energy * float(np.sum(ctrl ** 2))

        if self.prev_ctrl is not None:
            smooth_cost = self.w_smooth * float(np.sum((ctrl - self.prev_ctrl) ** 2))
        else:
            smooth_cost = 0.0
        self.prev_ctrl = ctrl.copy()

        # Total reward
        reward = (
            r_forward
            + r_posture
            + r_hop
            + r_contact_sync
            + r_action_sym
            - energy_cost
            - smooth_cost
        )

        # Track height for debugging
        self.prev_height = z
        self.max_height_achieved = max(self.max_height_achieved, z)

        return float(reward)

    # ---------- TERMINATION ----------

    def _is_terminated(self):
        if not self.terminate_on_fall:
            return False

        # Use torso world height, not qpos[1]
        height = float(self.data.xpos[self.torso_body_id, 2])
        pitch = float(self.data.qpos[2])

        # Consider fallen if too low or too tilted
        is_fallen = height < 0.7          # now 0.7 is meaningful (torso at ~1.3 when upright)
        is_unbalanced = abs(pitch) > 1.0  # ~57 degrees

        return bool(is_fallen or is_unbalanced)

    # ---------- RENDER / CLOSE ----------

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
