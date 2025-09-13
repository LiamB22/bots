import random
import gymnasium
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from sb3_contrib.common.wrappers import ActionMasker
import catanatron.gym
import numpy as np

def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])

def main():

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

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    
    # observation is a dict with board and numeric
    observation, info = env.reset()
    
    numeric_features = observation['numeric']
    # board tensor
    board_features = observation['board']
    
    # gets the size of the space
    state_space_size = env.observation_space['numeric'].shape[0]
    action_space_size = env.action_space.n
    q_table = np.zeros((state_space_size,action_space_size))

    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 0.1
    all_rewards = []

    num_episodes = 100
    model_name = f"DQN_model_{num_episodes}"
    save_path = f"bots/models/DQN/{model_name}"

    for episode in range(num_episodes):
        observation, info = env.reset()
        observation = observation['numeric']
        done = False
        curr_rewards = 0
        
        while not done:
            exploration_threshold = random.uniform(0,1)
            if exploration_threshold > epsilon:
                action = np.argmax(q_table[observation, :])
            else:
                action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)
            new_observation = new_observation['numeric']
            
            q_table[observation, action] = q_table[observation, action] + \
                learning_rate*(reward + discount_factor*np.max(q_table[new_observation,:]) - q_table[observation, action])

            observation = new_observation
            curr_rewards += reward

            done = terminated or truncated

        all_rewards.append(curr_rewards)
            
    print(all_rewards)
    env.close()

if __name__ == "__main__":
    main()