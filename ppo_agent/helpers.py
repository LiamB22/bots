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
from pathlib import Path

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
    return rewards["none"]

def alt_dense_reward_function(game, p0_color):

    reward_file_path = Path("bots/ppo_agent/reward.txt")
    try:
        reward_file_empty = (not reward_file_path.exists()) or reward_file_path.stat().st_size == 0
    except OSError:
        reward_file_empty = True

    rewards = config.rewards

    winning_color = game.winning_color()
    if winning_color is not None:  # Game ended
        if not reward_file_empty:
            try:
                # opening with "w" truncates the file, effectively clearing it
                with reward_file_path.open("w"):
                    pass
            except OSError:
                pass

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

    # reward_file_empty is True if the file is missing or empty, False otherwise
    if not reward_file_empty:
        # Read previous values
        prev = {}
        try:
            with reward_file_path.open("r") as rf:
                for line in rf:
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # parse booleans and ints
                    if v in ("True", "False"):
                        prev[k] = (v == "True")
                    else:
                        try:
                            prev[k] = int(v)
                        except ValueError:
                            try:
                                prev[k] = float(v)
                            except ValueError:
                                prev[k] = v
        except OSError:
            prev = {}
        # print("I AM THE PREVIOUS THINGY", prev)
        # helper to get previous numeric or boolean defaulting to current value
        prev_played_dev_card = prev.get("played_dev_card", played_dev_card)
        prev_current_vp = int(prev.get("current_vp", current_vp))
        prev_longest_road_length = int(prev.get("longest_road_length", longest_road_length))
        prev_has_longest_road = prev.get("has_longest_road", has_longest_road)
        prev_has_largest_army = prev.get("has_largest_army", has_largest_army)
        prev_roads_left = int(prev.get("roads_left", roads_left))
        prev_settlements_left = int(prev.get("settlements_left", settlements_left))
        prev_cities_left = int(prev.get("cities_left", cities_left))

        # compute positive deltas (only reward increases)
        delta_vp = max(0, current_vp - prev_current_vp)
        delta_longest = max(0, longest_road_length - prev_longest_road_length)
        # roads/settlements/cities: a decrease in "left" means pieces were placed -> positive change
        delta_roads_built = max(0, prev_roads_left - roads_left)
        delta_settlements_built = max(0, prev_settlements_left - settlements_left)
        delta_cities_built = max(0, prev_cities_left - cities_left)

        # booleans: only True if they flipped from False -> True
        gained_longest = (not prev_has_longest_road) and has_longest_road
        gained_army = (not prev_has_largest_army) and has_largest_army
        played_dev_card_gain = (not prev_played_dev_card) and played_dev_card

        # Convert deltas back into the variables used in the later reward formula so only increases are rewarded.
        current_vp = delta_vp
        longest_road_length = delta_longest
        
        roads_left = delta_roads_built
        settlements_left = delta_settlements_built
        cities_left = delta_cities_built

        has_longest_road = gained_longest
        has_largest_army = gained_army
        played_dev_card = played_dev_card_gain

        # update stored file with current absolute values for next comparison
        try:
            with reward_file_path.open("w") as rf:
                rf.write(f"played_dev_card: {state.player_state[f'{key}HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN']}\n")
                rf.write(f"current_vp: {state.player_state[f'{key}ACTUAL_VICTORY_POINTS']}\n")
                rf.write(f"longest_road_length: {state.player_state[f'{key}LONGEST_ROAD_LENGTH']}\n")
                rf.write(f"has_longest_road: {state.player_state[f'{key}HAS_ROAD']}\n")
                rf.write(f"has_largest_army: {state.player_state[f'{key}HAS_ARMY']}\n")
                rf.write(f"roads_left: {state.player_state[f'{key}ROADS_AVAILABLE']}\n")
                rf.write(f"settlements_left: {state.player_state[f'{key}SETTLEMENTS_AVAILABLE']}\n")
                rf.write(f"cities_left: {state.player_state[f'{key}CITIES_AVAILABLE']}\n")
        except OSError:
            pass
    else:
        try:
            with reward_file_path.open("w") as rf:
                rf.write(f"played_dev_card: {played_dev_card}\n")
                rf.write(f"current_vp: {current_vp}\n")
                rf.write(f"longest_road_length: {longest_road_length}\n")
                rf.write(f"has_longest_road: {has_longest_road}\n")
                rf.write(f"has_largest_army: {has_largest_army}\n")
                rf.write(f"roads_left: {roads_left}\n")
                rf.write(f"settlements_left: {settlements_left}\n")
                rf.write(f"cities_left: {cities_left}\n")
        except OSError:
            pass
        pass
    
    reward = (
        rewards["current_vp"] * current_vp
        + rewards["longest_road_length"] * longest_road_length
        + rewards["roads_left"] * (roads_left)
        + rewards["settlements_left"] * (settlements_left)
        + rewards["cities_left"] * (cities_left)
        + rewards["move_penalty"]
    )

    if played_dev_card:
        reward += rewards["played_dev_card"]
    if has_largest_army:
        reward += rewards["has_largest_army"]
    if has_longest_road:
        reward += rewards["has_longest_road"]

    # print a detailed breakdown of the reward components to the terminal
    components = {}
    components["current_vp"] = rewards["current_vp"] * current_vp
    components["longest_road_length"] = rewards["longest_road_length"] * longest_road_length
    components["roads_built"] = rewards["roads_left"] * (roads_left)
    components["settlements_built"] = rewards["settlements_left"] * (settlements_left)
    components["cities_built"] = rewards["cities_left"] * (cities_left)
    components["move_penalty"] = rewards["move_penalty"]

    if played_dev_card:
        components["played_dev_card"] = rewards["played_dev_card"]
    if has_largest_army:
        components["has_largest_army"] = rewards["has_largest_army"]
    if has_longest_road:
        components["has_longest_road"] = rewards["has_longest_road"]

    # Print weights and computed contributions
    # print("Reward weights:", {k: rewards.get(k, None) for k in ["current_vp", "longest_road_length", "roads_left", "settlements_left", "cities_left", "move_penalty", "played_dev_card", "has_largest_army", "has_longest_road"]})
    # print("Reward breakdown:")
    # for name, val in components.items():
    #     print(f"  {name}: {val}")
    # total_printed = sum(components.values())
    # print(f"Total reward (computed from components): {total_printed}")

    return reward
    
def dense_reward_function(game, p0_color):
    
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
        reward_function = dense_reward_function
    elif reward == reward_functions[1]:
        reward_function = alt_dense_reward_function
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