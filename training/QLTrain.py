import random
import gymnasium
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from sb3_contrib.common.wrappers import ActionMasker
import catanatron.gym
import numpy as np

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
    return current_vp * 0.1  # Fixed: initialize reward variable

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])

def discretize_observation(observation, bins=10):
    """Convert continuous observation to a discrete integer index"""
    # Use hashing to create a unique integer from the observation
    return hash(tuple(np.round(observation, decimals=3)))

def save_model(q_table, filename):
    import pickle
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    import pickle
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model file {filename} not found, starting fresh")
        return {}

def evaluate_agent(env, q_table, num_episodes=10):
    """Evaluate the trained agent without exploration"""
    total_rewards = 0
    wins = 0
    
    print(f"\n=== Evaluating Agent for {num_episodes} episodes ===")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        numeric_observation = observation['numeric']
        state_index = discretize_observation(numeric_observation)
        done = False
        episode_reward = 0
        
        while not done:
            # Get Q-values for current state
            if state_index in q_table:
                q_values = q_table[state_index]
            else:
                # If state not in Q-table, use random valid action
                q_values = np.zeros(env.action_space.n)
            
            # Get action mask and choose best valid action
            action_mask = mask_fn(env)
            masked_q = np.where(action_mask, q_values, -np.inf)
            action = np.argmax(masked_q)
            
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_numeric = new_observation['numeric']
            new_state_index = discretize_observation(new_numeric)
            
            state_index = new_state_index
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards += episode_reward
        if reward > 0:  # Positive final reward indicates win
            wins += 1
            
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    win_rate = wins / num_episodes * 100
    avg_reward = total_rewards / num_episodes
    print(f"\nEvaluation Results:")
    print(f"Win Rate = {win_rate:.1f}% ({wins}/{num_episodes})")
    print(f"Average Reward = {avg_reward:.2f}")
    return win_rate, avg_reward

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

    # Get action space size
    action_space_size = env.action_space.n
    
    # Use a dictionary-based Q-table instead of numpy array
    # This handles the large state space efficiently
    q_table = {}  # state_index -> array of action values

    def get_q_values(state_index, action_size):
        """Get Q-values for a state, initialize if not exists"""
        if state_index not in q_table:
            # Initialize with small random values
            q_table[state_index] = np.random.uniform(low=-0.1, high=0.1, size=action_size)
        return q_table[state_index]

    learning_rate = 0.1
    discount_factor = 0.99
    initial_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995
    all_rewards = []

    num_episodes = 100
    model_name = f"DQN_model_{num_episodes}"
    save_path = f"bots/models/DQN/{model_name}"

    for episode in range(num_episodes):
        observation, info = env.reset()
        numeric_observation = observation['numeric']
        state_index = discretize_observation(numeric_observation)
        done = False
        curr_rewards = 0
        
        while not done:
            # Get Q-values for current state
            q_values = get_q_values(state_index, action_space_size)
            
            # Get action mask for current state
            action_mask = mask_fn(env)
            
            current_epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** episode))
            
            exploration_threshold = random.uniform(0, 1)
            if exploration_threshold > current_epsilon:
                # Exploitation: choose best valid action using mask
                masked_q = np.where(action_mask, q_values, -np.inf)
                action = np.argmax(masked_q)
            else:
                # Exploration: choose random valid action using mask
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    # Fallback if no valid actions (shouldn't happen with proper masking)
                    action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)
            new_numeric = new_observation['numeric']
            new_state_index = discretize_observation(new_numeric)
            
            # Get Q-values for next state
            next_q_values = get_q_values(new_state_index, action_space_size)
            
            # Get action mask for next state (for max calculation)
            next_action_mask = mask_fn(env)
            
            # Q-learning update with action masking for next state
            current_q = q_values[action]
            
            # Use masking for max Q-value calculation in next state
            if next_action_mask is not None and np.any(next_action_mask):
                best_next_q = np.max(np.where(next_action_mask, next_q_values, -np.inf))
            else:
                best_next_q = np.max(next_q_values)
            
            new_q = current_q + learning_rate * (reward + discount_factor * best_next_q - current_q)
            q_table[state_index][action] = new_q

            state_index = new_state_index
            curr_rewards += reward
            done = terminated or truncated

        all_rewards.append(curr_rewards)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {curr_rewards}, Unique States: {len(q_table)}")
            
    print(f"Final rewards: {all_rewards}")
    print(f"Total unique states visited: {len(q_table)}")
    
    # Save the Q-table
    import pickle
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(f"{save_path}.pkl", 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Model saved to {save_path}.pkl")
    
    env.close()


    model_path = "bots/models/DQN/DQN_model_100.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        q_table = model_data['q_table']
        print(f"Loaded model trained for {model_data['performance']['episodes_trained']} episodes")
    except:
        print("Error loading model")
        return
    
    # Create environment
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
            "representation": "mixed",
        },
    )
    env = ActionMasker(env, mask_fn)
    
    # Evaluate
    win_rate, avg_reward = evaluate_agent(env, q_table, num_episodes=10)
    
    env.close()

if __name__ == "__main__":
    main()