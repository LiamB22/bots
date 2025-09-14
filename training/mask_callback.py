import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os

class MaskableEvalCallback(BaseCallback):
    """
    Custom evaluation callback for MaskablePPO that handles action masking.
    """
    
    def __init__(
        self,
        eval_env,
        mask_fn,
        best_model_save_path=None,
        log_path=None,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.mask_fn = mask_fn
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        
        self.best_mean_reward = -np.inf
        self.eval_results = []  # Store evaluation results
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = self.evaluate_policy()
            
            if len(episode_rewards) == 0:
                return True
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {np.std(episode_lengths):.2f}")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose >= 1:
                    print(f"New best mean reward: {self.best_mean_reward:.2f}")
                
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/std_reward", float(std_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            
            # Store results
            self.eval_results.append({
                'timesteps': self.num_timesteps,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'episode_lengths': episode_lengths,
                'episode_rewards': episode_rewards,
            })
            
        return True
    
    def evaluate_policy(self):
        """Evaluate the policy with proper action masking"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            # Reset environment - handle tuple return (obs, info)
            reset_result = self.eval_env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]  # Get just the observation
            else:
                obs = reset_result
                
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                # Get action masks
                action_masks = self.mask_fn(self.eval_env)
                
                # Handle dictionary observations for "mixed" representation
                if isinstance(obs, dict):
                    # For mixed representation, ensure proper format
                    model_obs = {
                        'board': np.array(obs['board']),
                        'numeric': np.array(obs['numeric'])
                    }
                else:
                    model_obs = obs
                
                # Predict with action masks
                action, _ = self.model.predict(
                    model_obs, 
                    action_masks=action_masks, 
                    deterministic=self.deterministic
                )
                
                # Take action
                step_result = self.eval_env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, info, _ = step_result  # Handle 5-tuple return
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if self.verbose >= 2:
                print(f"Evaluation episode {episode + 1}: reward={episode_reward:.2f}")
        
        return episode_rewards, episode_lengths
    
    def get_eval_results(self):
        """Return all evaluation results"""
        return self.eval_results