import numpy as np
import gymnasium

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.monitor import Monitor

import config

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
    current_vp = state.player_state[f"{key}ACTUAL_VICTORY_POINTS"]
    longest_road_length = state.player_state[f"{key}LONGEST_ROAD_LENGTH"]
    reward += (-0.1)/(current_vp + 1)
    reward += (-0.1)/(longest_road_length + 1)
    
    if played_dev_card:
        reward += 1

    return reward

def make_envs(mask_fn, representation):

    # 3-player catan on a "Mini" map (7 tiles) until 6 points.
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
            # WeightedRandomPlayer(Color.WHITE)
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

def evaluate(eval_env, model, num_episodes=config.episodes):
    """
    Evaluate the model over multiple episodes and return performance metrics.
    
    Args:
        eval_env: The evaluation environment
        model: The trained model to evaluate
        num_episodes: Number of episodes to run for evaluation
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    total_rewards = []
    wins = 0
    losses = 0
    
    for episode in range(num_episodes):
        observation, info = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action mask for current state
            action_mask = mask_fn(eval_env)
            
            # Predict action with masking enabled
            action, _ = model.predict(observation, action_masks=action_mask, deterministic=False)
            
            # Take the action
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Check if game ended with a win/loss
            if done:
                if reward > 0:  # Win
                    wins += 1
                elif reward < 0:  # Loss
                    losses += 1
        
        total_rewards.append(episode_reward)
    
    eval_env.close()
    
    # Calculate evaluation metrics
    metrics = {
        'total_episodes': num_episodes,
        'win_rate': wins / num_episodes if num_episodes > 0 else 0,
        'loss_rate': losses / num_episodes if num_episodes > 0 else 0,
        'avg_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        'min_reward': min(total_rewards) if total_rewards else 0,
        'max_reward': max(total_rewards) if total_rewards else 0,
        'total_wins': wins,
        'total_losses': losses
    }
    
    return metrics