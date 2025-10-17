import numpy as np
import gymnasium

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.value import ValueFunctionPlayer
import catanatron.gym

from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.monitor import Monitor

from tqdm import tqdm

import config

def linear_schedule(progress_remaining):
    initial_lr = 3e-4
    final_lr = 3e-6
    return initial_lr + (1 - progress_remaining) * (final_lr - initial_lr)

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

def sparse_reward_function(game, p0_color):

    rewards = config.rewards
    winning_color = game.winning_color()
    if winning_color is not None:  # Game ended
        if p0_color == winning_color:
            return rewards["win"]
        else:
            return rewards["lose"]
    else:
        return rewards["none"]
        
def my_reward_function(game, p0_color):
    
    rewards = config.rewards
    starting_cities = 4
    starting_settlements = 4
    starting_roads = 15

    winning_color = game.winning_color()
    if winning_color is not None:  # Game ended
        if p0_color == winning_color:
            return rewards["win"]
        else:
            return rewards["lose"]
    
    state = game.state
    colour_index = state.color_to_index[p0_color]
    key = f"P{colour_index}_"

    played_dev_card = state.player_state[f"{key}HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]
    current_vp = state.player_state[f"{key}ACTUAL_VICTORY_POINTS"]
    longest_road_length = state.player_state[f"{key}LONGEST_ROAD_LENGTH"]
    has_longest_road = state.player_state[f"{key}HAS_ROAD"]
    has_largest_army = state.player_state[f"{key}HAS_ARMY"]
    roads_left = state.player_state[f"{key}ROADS_AVAILABLE"]
    settlements_left = state.player_state[f"{key}SETTLEMENTS_AVAILABLE"]
    cities_left = state.player_state[f"{key}CITIES_AVAILABLE"]

    # negative rewards
    # reward = (rewards["s_negative"])/(current_vp + 1) + \
    #     (rewards["s_negative"])/(longest_road_length + 1) + \
    #         rewards["s_negative"]*roads_left + \
    #             rewards["s_negative"]*settlements_left + \
    #                 rewards["s_negative"]*cities_left

    # positive rewards
    reward =rewards["current_vp"]*current_vp + \
            rewards["longest_road_length"]*longest_road_length + \
            rewards["roads_left"]*(starting_roads - roads_left) + \
            rewards["settlements_left"]*(starting_settlements - settlements_left) + \
            rewards["cities_left"]*(starting_cities - cities_left) + \
            rewards["move_penalty"]
    
    if played_dev_card:
        reward += rewards["played_dev_card"]
    if has_largest_army:
        reward += rewards["has_largest_army"]
    if has_longest_road:
        reward += rewards["has_longest_road"]

    return reward

def make_envs():

    num_enemies = config.num_enemies
    enemy_type = config.enemy_type
    map_type = config.map_type
    vps_to_win = config.vps_to_win
    representation = config.representation
    reward_functions = config.reward_functions
    reward = config.reward
    if reward == reward_functions[0]:
        reward_function = my_reward_function
    else:
        reward_function = sparse_reward_function

    enemy_list = get_enemy_list(num_enemies) # gets a list of different players with num_enemies of each type
    enemies = enemy_list[enemy_type] # gets the list of correct enemies according to the number taken from config
    # 3-player catan on a "Mini" map (7 tiles) until 6 points.
    configuration={
        "map_type": map_type,
        "vps_to_win": vps_to_win,
        "enemies": enemies,
        "reward_function": reward_function,
        "representation": representation,
    }
    
    env = gymnasium.make("catanatron/Catanatron-v0",config=configuration)
    eval_env = gymnasium.make("catanatron/Catanatron-v0",config=configuration)

    # Init Environment and Model
    env = Monitor(ActionMasker(env, mask_fn))
    eval_env = Monitor(ActionMasker(eval_env, mask_fn))

    return env, eval_env
    
def evaluate(eval_env, model, num_episodes=config.eval_episodes):
    
    total_rewards = []
    wins = 0
    losses = 0
    
    for episode in tqdm(range(num_episodes)):
        observation, info = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action mask for current state
            action_mask = mask_fn(eval_env)
            
            # Predict action with masking enabled
            action, _ = model.predict(observation, action_masks=action_mask, deterministic=True)
            
            # Take the action
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Check if game ended with a win/loss
            if done:
                winning_colour = eval_env.unwrapped.game.winning_color()
                if winning_colour == Color.BLUE:
                    wins += 1
                else:
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
    
    print("Evaluation Results:")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Wins: {metrics['total_wins']}, Losses: {metrics['total_losses']}")
    print(f"Min/Max Reward: {metrics['min_reward']:.2f}/{metrics['max_reward']:.2f}")

def get_enemy_list(num_enemies):
    players_1 = [
        [
            WeightedRandomPlayer(Color.RED)
        ],
        [
            AlphaBetaPlayer(Color.RED)
        ],
        [
            MCTSPlayer(Color.RED)
        ],
        [
            ValueFunctionPlayer(Color.RED)
        ],
        [
            AlphaBetaPlayer(Color.RED)
        ],
    ]
    players_2 = [
        [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE)
        ],
        [
            AlphaBetaPlayer(Color.RED),
            AlphaBetaPlayer(Color.ORANGE)
        ],
        [
            MCTSPlayer(Color.RED),
            MCTSPlayer(Color.ORANGE)
        ],
        [
            ValueFunctionPlayer(Color.RED),
            ValueFunctionPlayer(Color.ORANGE)
        ],
        [
            WeightedRandomPlayer(Color.RED),
            MCTSPlayer(Color.ORANGE)
        ],
    ]
    players_3 = [
        [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
            WeightedRandomPlayer(Color.WHITE)
        ],
        [
            AlphaBetaPlayer(Color.RED),
            AlphaBetaPlayer(Color.ORANGE),
            AlphaBetaPlayer(Color.WHITE)
        ],
        [
            MCTSPlayer(Color.RED),
            MCTSPlayer(Color.ORANGE),
            MCTSPlayer(Color.WHITE)
        ],
        [
            ValueFunctionPlayer(Color.RED),
            ValueFunctionPlayer(Color.ORANGE),
            ValueFunctionPlayer(Color.WHITE)
        ],
        [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
            ValueFunctionPlayer(Color.WHITE)
        ],
    ]
    if num_enemies == 1:
        return players_1
    elif num_enemies == 2:
        return players_2
    elif num_enemies == 3:
        return players_3
    else:
        return []