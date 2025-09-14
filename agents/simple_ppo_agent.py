import random
import gymnasium
import os
import sys

from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

from helpers.helpers import my_reward_function, mask_fn, make_envs
from networks.networks import CNN, COMBINED

def main():

    train_timesteps = 500_000
    eval_episodes = 1
    eval_timesteps = train_timesteps/100000
    deterministic = False
    representation = "mixed"
    # mixed or vector
    
    env, eval_env = make_envs(my_reward_function, mask_fn, representation)

    valid_actions = env.unwrapped.get_valid_actions()
    # print(f"Total valid actions: {len(valid_actions)}")
    # print("First 10 valid actions:", valid_actions[:10])
    # print("Observation space:", env.observation_space)
    # print("Board shape:", env.observation_space['board'].shape)
    # print("Numeric shape:", env.observation_space['numeric'].shape)

    # choose from possible models: mlp_ppo, cnn_ppo, combination_ppo
    if representation == "vector":

        model_name = "mlp_ppo"

        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
    else: # must be mixed then

        # model_name = "cnn_ppo"
        model_name = "combined_ppo"

        if model_name == "cnn_ppo":
            policy_kwargs = dict(
                features_extractor_class=CNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
        else: # must be combined then
            policy_kwargs = dict(
                features_extractor_class=COMBINED,
                features_extractor_kwargs=dict(
                    features_dim=512,
                    cnn_features_dim=256,  # Size for CNN branch
                    mlp_features_dim=256   # Size for MLP branch
                ),
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
            
    model_dir = "bots/models/PPO"
    best_model_name = "best_model"
    save_path = f"{model_dir}/{model_name}"
    load_path = f"{save_path}/{best_model_name}"
    os.makedirs(save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_timesteps,
        deterministic=deterministic,
        render=False,
        n_eval_episodes=eval_episodes,
        verbose=2
    )

    # Create the Maskable PPO agent
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096, # experience before update
        batch_size=128, # size of minibatches creates n_steps/batch_size mini-batches
        n_epochs=15, # number of times we use n_steps
        gamma=0.995, # discount factor
        gae_lambda=0.95, # generalising advantage estimation
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01, # encourages exploration by adding entropy bonus to the loss
        vf_coef=0.5, # weight of value function loss relative to policy loss
        max_grad_norm=0.5, # clips gradient to prevent exploding gradient problem
    )

    # Learn and Load the best saved model
    class DebugCallback(BaseCallback):
        def _on_step(self):
            if self.n_calls % 1000 == 0:
                print(f"Step {self.n_calls}: Action={self.locals['actions']}, Reward={self.locals['rewards']}, Valid actions={self.locals['infos']}")
            return True

    # Use multiple callbacks
    callback = CallbackList([eval_callback, DebugCallback()])
    model.learn(total_timesteps=train_timesteps, callback=callback)
    # model.learn(total_timesteps=train_timesteps, callback=eval_callback)
    best_model = MaskablePPO.load(load_path, env=env)

    print(model.policy)

if __name__ == "__main__":
    main()