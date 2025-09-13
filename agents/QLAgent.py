import random
import gymnasium
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

def my_reward_function(game, p0_color):
    state = game.state
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is not None:
        return -100

    colour_index = state.color_to_index[p0_color]
    key = f"P{colour_index}_"
    current_vp = state.player_state[f"{key}ACTUAL_VICTORY_POINTS"]
    return current_vp * 0.1

# from state.py ===========================
# These will be prefixed by P0_, P1_, ...
# Create Player State blueprint
# PLAYER_INITIAL_STATE = {
#     "VICTORY_POINTS": 0,
#     "ROADS_AVAILABLE": 15,
#     "SETTLEMENTS_AVAILABLE": 5,
#     "CITIES_AVAILABLE": 4,
#     "HAS_ROAD": False,
#     "HAS_ARMY": False,
#     "HAS_ROLLED": False,
#     "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN": False,
#     # de-normalized features (for performance since we think they are good features)
#     "ACTUAL_VICTORY_POINTS": 0,
#     "LONGEST_ROAD_LENGTH": 0,
#     "KNIGHT_OWNED_AT_START": False,
#     "MONOPOLY_OWNED_AT_START": False,
#     "YEAR_OF_PLENTY_OWNED_AT_START": False,
#     "ROAD_BUILDING_OWNED_AT_START": False,
# }
# =========================================

def train_loop(env, save_path):

    start_state = env.unwrapped.game.state # starting state of the game - check state.py
    
    observation, info = env.reset() # first observation
    
    N = len(start_state.players) # number of players
    numeric_features = observation['numeric'] # gets a list of numeric features from the observation
    board_features = observation['board'] # gets a list of tensors from the board features of the observation

    numeric_state_space_size = numeric_features.shape[0] # numeric shape changes with number of players 14*N + 20

    # tensor shapes
    num_tensors = len(board_features) # channels - changes with number of player 2*N + 12
    tensor_dim_1 = len(board_features[0]) # width - 21
    tensor_dim_2 = len(board_features[0][0]) # height - 11

    for _ in range(1000):
        # your agent here (this takes random actions)
        action = random.choice(info["valid_actions"])

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            observation, info = env.reset()
    env.close()

def eval_loop(env, save_path):

    observation, info = env.reset()

    for _ in range(1000):
        # your agent here (this takes random actions)
        action = random.choice(info["valid_actions"])

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            observation, info = env.reset()
    env.close()

def main():

    file_name = "attempt"
    save_path = f"bots/models/attempt/{file_name}"

    # 3-player catan on a "Mini" map (7 tiles) until 6 points.
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "enemies": [
                WeightedRandomPlayer(Color.RED),
                WeightedRandomPlayer(Color.ORANGE),
                WeightedRandomPlayer(Color.WHITE)
            ],
            "reward_function": my_reward_function,
            "representation": "mixed",
        },
    )

    train_loop(env, save_path)
    # eval_loop(env, save_path)

if __name__ == "__main__":
    main()