import os
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

import config
from helpers import mask_fn, make_envs, evaluate
from networks import CNN, COMBINED

def main():
    
    train_timesteps = config.train_timesteps
    representation = config.representation
    model_name = config.model_name
    
    env, eval_env = make_envs(mask_fn, representation)

    # choose from possible models: mlp_ppo, cnn_ppo, combination_ppo
    if representation == config.representations[0]:

        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )

    else:
        
        if model_name == config.model_names[1]:
            policy_kwargs = dict(
                features_extractor_class=CNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
        elif model_name == config.model_names[2]: # must be combined then
            policy_kwargs = dict(
                features_extractor_class=COMBINED,
                features_extractor_kwargs=dict(
                    features_dim=512,
                    cnn_features_dim=256,  # Size for CNN branch
                    mlp_features_dim=256   # Size for MLP branch
                ),
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
        else:
            policy_kwargs = dict(
                features_extractor_class=CNN,
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=dict(pi=[256, 128], vf=[256, 128])
            )
            
    model_dir = config.model_dir
    best_model_name = config.best_model_name
    save_path = f"{model_dir}/{model_name}"
    load_path = f"{save_path}/{best_model_name}"
    os.makedirs(save_path, exist_ok=True)

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

    model.learn(total_timesteps=train_timesteps)
    model.save(f"{save_path}/{best_model_name}")
    
    best_model_path = load_path + ".zip"
    if os.path.exists(best_model_path):
        best_model = MaskablePPO.load(load_path, env=env)
        print("Best model loaded successfully")
    else:
        print("No best model found")

    # In your main() function, replace the evaluation part:
    info = evaluate(eval_env, best_model)
    print("Evaluation Results:")
    print(f"Win Rate: {info['win_rate']:.2%}")
    print(f"Average Reward: {info['avg_reward']:.2f}")
    print(f"Wins: {info['total_wins']}, Losses: {info['total_losses']}")
    print(f"Min/Max Reward: {info['min_reward']:.2f}/{info['max_reward']:.2f}")
    print(model.policy)

if __name__ == "__main__":
    main()