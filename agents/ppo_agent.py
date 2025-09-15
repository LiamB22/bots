from catanatron import Player
from catanatron.cli import register_cli_player
import numpy as np
import os
import sys
import torch

# Add the necessary paths to import your modules
current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from helpers.helpers import mask_fn, my_reward_function
from networks.networks import CNN, COMBINED

class PPOPolicyPlayer(Player):
    def __init__(self, color, model_path=None, representation="vector", deterministic=True):
        super().__init__(color)
        self.representation = representation
        self.deterministic = deterministic
        self.action_mapping = {}  # Will store mapping between actions and indices
        self.reverse_action_mapping = {}  # Reverse mapping
        
        # Load the model if path is provided
        if model_path and os.path.exists(model_path + ".zip"):
            # Create a minimal env for loading and to get action space info
            from helpers.helpers import make_envs
            env, _ = make_envs(my_reward_function, mask_fn, representation)
            self.model = MaskablePPO.load(model_path, env=env)
            print(f"Loaded model from {model_path}")
            
            # Initialize action mapping using the environment
            self.initialize_action_mapping(env)
        else:
            self.model = None
            print("No model loaded - using random fallback")
    
    def initialize_action_mapping(self, env):
        """Initialize the mapping between action objects and environment indices"""
        # We need to simulate the environment to build the action mapping
        # This is a bit hacky but necessary to get the correct mapping
        try:
            obs = env.reset()
            done = False
            max_actions_to_sample = 1000  # Limit to avoid infinite loops
            
            for i in range(max_actions_to_sample):
                if done:
                    obs = env.reset()
                    done = False
                
                # Get valid actions from the environment
                valid_actions = env.unwrapped.get_valid_actions()
                
                # For each valid action, get the corresponding action object
                for action_idx in valid_actions:
                    if action_idx not in self.reverse_action_mapping:
                        # Get the action object from the environment
                        action_obj = env.unwrapped.index_to_action(action_idx)
                        if action_obj is not None:
                            # Create a hashable key for the action
                            action_key = self._action_to_key(action_obj)
                            self.action_mapping[action_key] = action_idx
                            self.reverse_action_mapping[action_idx] = action_obj
                
                # Take a random action to progress the environment
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                
        except Exception as e:
            print(f"Warning: Could not fully initialize action mapping: {e}")
    
    def _action_to_key(self, action):
        """Convert an Action object to a hashable key"""
        # Use a tuple representation that captures all important aspects
        return (
            str(action.color),
            str(action.action_type),
            str(action.value) if action.value is not None else "None"
        )
    
    def _find_matching_action(self, target_action, playable_actions):
        """Find the playable action that matches the target action"""
        target_key = self._action_to_key(target_action)
        
        for playable_action in playable_actions:
            playable_key = self._action_to_key(playable_action)
            if playable_key == target_key:
                return playable_action
        
        return None
    
    def get_observation_from_game(self, game, current_player_color):
        """Convert game state to the appropriate observation format"""
        observation = game.state.to_observation(current_player_color)
        
        if self.representation == "mixed":
            # Mixed representation: dictionary with 'board' and 'numeric'
            return {
                'board': np.array(observation['board']),
                'numeric': np.array(observation['numeric'])
            }
        else:  # vector representation
            # Vector representation: single numpy array
            if 'vector' in observation:
                return np.array(observation['vector'])
            else:
                # Convert dict to vector if needed
                return np.concatenate([np.array(v).flatten() for v in observation.values()])
    
    def get_action_masks(self, playable_actions):
        """Get action masks for the current playable actions"""
        if self.model is None:
            return None
        
        mask = np.zeros(self.model.action_space.n, dtype=np.float32)
        
        for action in playable_actions:
            action_key = self._action_to_key(action)
            if action_key in self.action_mapping:
                action_idx = self.action_mapping[action_key]
                if action_idx < len(mask):
                    mask[action_idx] = 1
        
        return mask
    
    def decide(self, game, playable_actions):
        """Use the trained PPO model to choose an action"""
        
        if self.model is None or len(self.action_mapping) == 0:
            # Fallback to random if no model loaded or no action mapping
            return random.choice(playable_actions)
        
        try:
            # Get observation from current game state
            observation = self.get_observation_from_game(game, self.color)
            
            # Get action masks
            action_masks = self.get_action_masks(playable_actions)
            
            if action_masks is None or np.sum(action_masks) == 0:
                # No valid actions in our mapping, fallback to random
                return random.choice(playable_actions)
            
            # Process observation for model input
            if self.representation == "mixed" and isinstance(observation, dict):
                # Convert to tensors and add batch dimension
                processed_obs = {
                    'board': torch.tensor(observation['board']).unsqueeze(0).float(),
                    'numeric': torch.tensor(observation['numeric']).unsqueeze(0).float()
                }
            else:
                processed_obs = torch.tensor(observation).unsqueeze(0).float()
            
            # Predict action using the model
            action_idx, _ = self.model.predict(
                processed_obs, 
                action_masks=action_masks, 
                deterministic=self.deterministic
            )
            
            # Convert action index back to catanatron Action
            if action_idx in self.reverse_action_mapping:
                target_action = self.reverse_action_mapping[action_idx]
                # Find the matching action in playable_actions
                chosen_action = self._find_matching_action(target_action, playable_actions)
                if chosen_action is not None:
                    return chosen_action
            
            # If mapping fails, fallback to random
            return random.choice(playable_actions)
            
        except Exception as e:
            print(f"Error in PPO decision: {e}")
            # Fallback to random action
            return random.choice(playable_actions)

# Factory function to create players with specific models
def create_ppo_player(model_name="mlp_ppo", representation="vector", deterministic=True):
    """Factory function to create PPO players with different models"""
    model_dir = "bots/models/PPO"
    model_path = f"{model_dir}/{model_name}/best_model"
    
    def player_factory(color):
        return PPOPolicyPlayer(color, model_path, representation, deterministic)
    
    return player_factory

# Register different PPO player variants
register_cli_player("PPO_MLP", create_ppo_player("mlp_ppo", "vector"))
register_cli_player("PPO_CNN", create_ppo_player("cnn_ppo", "mixed"))
register_cli_player("PPO_COMBINED", create_ppo_player("combined_ppo", "mixed"))

# Also register the base class for direct use
register_cli_player("PPO", PPOPolicyPlayer)

# Add random import
import random