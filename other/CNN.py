import random
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def my_reward_function(game, p0_color):
    
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 1
    elif winning_color is not None:
        return -1

    state = game.state
    reward = 0
    colour_index = state.color_to_index[p0_color]
    key = f"P{colour_index}_"
    current_vp = state.player_state[f"{key}ACTUAL_VICTORY_POINTS"]
    longest_road_length = state.player_state[f"{key}LONGEST_ROAD_LENGTH"]

    if current_vp > 0:
        reward += 0.1
    else:
        reward += -0.1
    if longest_road_length > 0:
        reward += 0.1
    else:
        reward += -0.1
    
    return reward

# also look at catanatron_env.py and state.py

# P0_WHEATS_IN_HAND, P0_WOODS_IN_HAND, ...
    # P0_ROAD_BUILDINGS_IN_HAND, P0_KNIGHT_IN_HAND, ..., P0_VPS_IN_HAND
    # P0_ROAD_BUILDINGS_PLAYABLE, P0_KNIGHT_PLAYABLE, ...
    # P0_ROAD_BUILDINGS_PLAYED, P0_KNIGHT_PLAYED, ...

    # P1_ROAD_BUILDINGS_PLAYED, P1_KNIGHT_PLAYED, ...

def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])

class CatanatronCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # Get the actual shape from observation space
        board_shape = observation_space['board'].shape
        input_channels = board_shape[0]
        height = board_shape[1]
        width = board_shape[2]
        
        print(f"Board shape: {board_shape}")
        
        super(CatanatronCNN, self).__init__(observation_space, features_dim)
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            dummy = self.conv_layers(dummy)
            flattened_size = dummy.view(1, -1).size(1)
            print(f"Flattened size: {flattened_size}")
        
        self.linear = nn.Linear(flattened_size, features_dim)
    
    def forward(self, observations):
        # Extract board tensor
        board_tensor = observations['board']
        
        # Debug: print shape
        # print(f"Input board shape: {board_tensor.shape}")
        
        # Feature extraction
        x = self.conv_layers(board_tensor)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        
        return x

def make_env():
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
    env = ActionMasker(env, mask_fn)
    return env

def train_sb3_agent(save_path):
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Policy configuration with your custom CNN
    policy_kwargs = dict(
        features_extractor_class=CatanatronCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    # Create the Maskable PPO agent
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="models/CNN/catanatron_tensorboard/",
    )
    
    # Add callback to monitor rewards
    from stable_baselines3.common.callbacks import EvalCallback
    
    # Create eval callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path + "_best",
        log_path=save_path + "_logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Train the agent with callback
    model.learn(
        total_timesteps=500000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()

def train_sb3_agent_continue(save_path):
    # Create vectorized environment FIRST
    env = DummyVecEnv([make_env])
    
    # Load the existing model and set its environment
    model = MaskablePPO.load("bots/models/PPOCNN/PPOCNN", env=env)
    
    # Add callback to monitor rewards
    from stable_baselines3.common.callbacks import EvalCallback
    
    # Create eval callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path + "_best",
        log_path=save_path + "_logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Continue training
    model.learn(
        total_timesteps=500000,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False  # Don't reset the timestep counter
    )
    
    # Save the continued model
    model.save(save_path + "_continued")
    print(f"Continued model saved to {save_path}_continued")
    
    env.close()

def train_loop(env, save_path):
    # Debug: Check observation structure
    observation, info = env.reset()
    print(f"Observation keys: {list(observation.keys())}")
    print(f"Numeric features shape: {observation['numeric'].shape}")
    print(f"Board features shape: {observation['board'].shape}")
    print(f"Board features type: {type(observation['board'])}")
    
    # If board is a list, convert to numpy array for debugging
    if isinstance(observation['board'], list):
        board_array = np.array(observation['board'])
        print(f"Converted board array shape: {board_array.shape}")
    
    # Simple random agent for testing
    for episode in range(10):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = random.choice(info["valid_actions"])
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        print(f"Episode {episode}, Reward: {episode_reward}")

def eval_loop_gym(env, save_path):
    try:
        model = MaskablePPO.load(save_path)
        
        total_reward = 0
        num_episodes = 5
        
        for episode in range(num_episodes):
            observation, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Convert observation to the format SB3 expects
                # SB3 expects a dictionary with the same structure as the observation space
                obs_dict = {
                    'numeric': np.array(observation['numeric']).reshape(1, -1),
                    'board': np.array(observation['board']).reshape(1, *observation['board'].shape)
                }
                
                # Get action mask
                action_masks = mask_fn(env)
                
                # Predict action
                action, _states = model.predict(obs_dict, action_masks=action_masks)
                
                # Take step
                observation, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                episode_reward += reward
            
            print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward}")
            total_reward += episode_reward
        
        print(f"Average Evaluation Reward: {total_reward / num_episodes}")
        
    except Exception as e:
        print(f"Could not load model for evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    file_name = "PPOCNN"
    save_path = f"bots/models/PPOCNN/{file_name}"
    
    # First, let's debug the environment
    print("=== Debugging Environment ===")
    env = make_env()
    train_loop(env, save_path)
    env.close()
    
    # Then train with SB3
    print("\n=== Training with SB3 ===")
    train_sb3_agent(save_path)
    
    # print("\n=== Continuing Training with SB3 ===")
    # train_sb3_agent_continue(save_path)  # Continue training

    # Evaluate using the proper method
    print("\n=== Evaluating ===")
    eval_env = make_env()
    eval_loop_gym(eval_env, save_path)  # Use the Gym API version
    eval_env.close()

if __name__ == "__main__":
    main()