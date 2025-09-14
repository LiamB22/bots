import numpy as np
import gymnasium

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.monitor import Monitor

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

def my_reward_function(game, p0_color):
    
    winning_color = game.winning_color()
    if winning_color is not None:  # Game ended
        if p0_color == winning_color:
            return 100
        else:
            return -100
    
    state = game.state
    reward = 0
    colour_index = state.color_to_index[p0_color]
    key = f"P{colour_index}_"
    played_dev_card = state.player_state[f"{key}HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]

    if played_dev_card:
        reward += 5

    return reward

def make_envs(my_reward_function, mask_fn, representation):

    # 3-player catan on a "Mini" map (7 tiles) until 6 points.
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
            WeightedRandomPlayer(Color.WHITE)
        ],
        "reward_function": my_reward_function,
        "representation": representation,
    }
    
    env = gymnasium.make("catanatron/Catanatron-v0",config=config)
    eval_env = gymnasium.make("catanatron/Catanatron-v0",config=config)

    # Init Environment and Model
    env = Monitor(ActionMasker(env, mask_fn))
    eval_env = Monitor(ActionMasker(eval_env, mask_fn))

    return env, eval_env