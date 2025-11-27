from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from RL_env import MuscleWalkerEnv


def make_env(rank, xml_path):
    def _init():
        return MuscleWalkerEnv(xml_path=xml_path, frame_skip=10)
    return _init


if __name__ == "__main__":
    xml_path = "assets/walker/muscle_new.xml"

    num_envs = 8
    test_env = MuscleWalkerEnv(xml_path=xml_path)
    check_env(test_env, warn=True)

    env = SubprocVecEnv([make_env(i, xml_path) for i in range(num_envs)])
    env = VecMonitor(env)
    
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048 // num_envs,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    model.learn(total_timesteps=1000000)
    model.save("ppo_muscle_walker")