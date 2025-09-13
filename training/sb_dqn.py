import gymnasium
import numpy as np
from stable_baselines3 import DQN
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

# 3-player catan on a "Mini" map (7 tiles) until 6 points.
env = gymnasium.make(
    "catanatron/Catanatron-v0",
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
        ],
        "reward_function": my_reward_function,
        "representation": "vector",
    },
)

eval_env = env

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

env = ActionMasker(env, mask_fn)
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
# Init Environment and Model
# env = gymnasium.make("catanatron/Catanatron-v0")
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking
# model = DQN("MultiInputPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=1_000_000)
model.save("models/DQN/stable_version")
loaded_model = MaskablePPO.load("models/DQN/stable_version", env=eval_env)
obs, _ = eval_env.reset()
for _ in range(1000):
    action, _ = loaded_model.predict(obs, action_masks=mask_fn(eval_env))
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
    print(reward)
    if done:
        obs, _ = eval_env.reset()