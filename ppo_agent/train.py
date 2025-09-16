import os
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

import config
from helpers import make_envs, evaluate
from networks import CNN, COMBINED

def main():
    
    train_timesteps = config.train_timesteps
    model_name = config.model_name
    model_names = config.model_names
    save_path = config.save_path
    representation = config.representation
    representations = config.representations
    
    env, eval_env = make_envs()

    policy_kwargs_list = {
        "vector":[dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )],
        "mixed":[dict(
            features_extractor_class=CNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        ),
        dict(
            features_extractor_class=COMBINED,
            features_extractor_kwargs=dict(
                features_dim=512,
                cnn_features_dim=256,  # Size for CNN branch
                mlp_features_dim=256   # Size for MLP branch
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )]
    }

    # choose from possible models: mlp_ppo, cnn_ppo, combination_ppo
    if representation == representations[0]:
        policy_kwargs = policy_kwargs_list[representation][0]
    else:
        if model_name == model_names[0]:
            policy_kwargs = policy_kwargs_list[representation][0]
        else:
            policy_kwargs = policy_kwargs_list[representation][1]
            
    # Create the Maskable PPO agent
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096, # experience before update
        batch_size=128, # size of minibatches creates n_steps/batch_size mini-batches
        n_epochs=15, # number of times we use n_steps (num games)
        gamma=0.995, # discount factor
        gae_lambda=0.95, # generalising advantage estimation
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01, # encourages exploration by adding entropy bonus to the loss
        vf_coef=0.5, # weight of value function loss relative to policy loss
        max_grad_norm=0.5, # clips gradient to prevent exploding gradient problem
    )

    model.learn(total_timesteps=train_timesteps)
    model.save(save_path)
    print("Best model saved successfully")
    print(model.policy)
    best_model_path = save_path
    if os.path.exists(best_model_path):
        best_model = MaskablePPO.load(best_model_path, env=env)
        print("Best model loaded successfully")
        evaluate(eval_env, best_model)
    else:
        print("No best model found")

if __name__ == "__main__":
    main()