import os
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

import config
from helpers import make_envs, evaluate, linear_schedule
from networks import COMBINED

def main():
    
    train_timesteps = config.train_timesteps
    model_name = config.model_name
    model_names = config.model_names
    save_path = config.save_path
    representation = config.representation
    representations = config.representations
    log_dir = config.log_dir
    train_model = config.train_model
    evaluate_model = config.evaluate_model
    show_model_policy = config.show_model_policy
    train_further = config.train_further
    best_model_path = config.best_models_path
    use_best_model = config.use_best_model
    
    env, eval_env = make_envs()

    policy_kwargs_list = {
        "vector": [dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )],
        "mixed": [dict(
            features_extractor_class=COMBINED,
            features_extractor_kwargs=dict(
                features_dim=512,
                cnn_features_dim=256,
                mlp_features_dim=256
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )]
    }

    # choose from possible models: mlp_ppo, cnn_ppo, combination_ppo
    policy_kwargs = policy_kwargs_list[representation][0]
    
    if train_further:
        model_path = save_path
        if os.path.exists(model_path):
            model = MaskablePPO.load(model_path, env=env)
            print("Model loaded successfully. Training model further.")
        else:
            print("No best model found")
    else:    
        # Create the Maskable PPO agent
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,
            # learning_rate=linear_schedule,
            n_steps=2048, # experience before update
            batch_size=128, # size of minibatches creates n_steps/batch_size mini-batches
            n_epochs=15, # number of times we use n_steps (num games)
            gamma=0.995, # discount factor
            gae_lambda=0.95, # generalising advantage estimation
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01, # encourages exploration by adding entropy bonus to the loss
            vf_coef=0.5, # weight of value function loss relative to policy loss
            max_grad_norm=0.1, # clips gradient to prevent exploding gradient problem
            tensorboard_log=log_dir, # log training for later viewing
        )
    if train_model:
        print(f"Training {model_name}")
        model.learn(total_timesteps=train_timesteps)
        model.save(save_path)
        print("Best model saved successfully")
    if show_model_policy:
        print(model.policy)
    if evaluate_model:
        if use_best_model:
            model_path = best_model_path
        else:
            model_path = save_path
        if os.path.exists(model_path):
            best_model = MaskablePPO.load(model_path, env=env)
            print("Best model loaded successfully")
            evaluate(eval_env, best_model)
        else:
            print("No best model found")

if __name__ == "__main__":
    main()