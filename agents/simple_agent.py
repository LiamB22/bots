import random
import gymnasium
import numpy as np
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym
import catanatron.gym.board_tensor_features as btf
import catanatron.gym.envs.catanatron_env as ce


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
        "reward_function": ce.simple_reward,
        "representation": "mixed",
    },
)

def simple_agent(valid_actions, observation):
    """
    Prioritise actions
    """
    for action in valid_actions:
        if 93 <= action <= 146:  # BUILD_SETTLEMENT actions
            return action
    
    for action in valid_actions:
        if 21 <= action <= 92:  # BUILD_ROAD actions
            return action
    
    for action in valid_actions:
        if 147 <= action <= 200:  # BUILD_CITY actions
            return action
    
    for action in valid_actions:
        if action == 201:  # BUY_DEVELOPMENT_CARD
            return action
    
    return random.choice(valid_actions)

def train_loop(func):
    # ==========================================================================
    num_games = 1
    wins = 0
    for _ in range(num_games):
        
        observation, info = env.reset()
        reward = 0
        epochs = 0
        done = False

        while not done:
            action = func(info["valid_actions"], observation)
            
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            epochs += 1

        if reward > 0:
            wins +=1

        print(f"Reward: {reward}")

    print(f"{wins} wins out of {num_games} games which gives a win rate of {(wins/num_games)*100}%")   
    # ==========================================================================

    env.close()

def main():
    train_loop(simple_agent)

if __name__ == "__main__":
    main()