import os
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

current_dir = os.path.dirname(__file__)
bots_root = os.path.join(current_dir, '..')
sys.path.append(bots_root)

from helpers.helpers import my_reward_function, mask_fn, make_envs
from networks.networks import CNN, COMBINED
from training.mask_callback import MaskableEvalCallback
from testing import test_output

def main():

    train_timesteps = 1_000_000
    eval_episodes = 100
    eval_timesteps = train_timesteps/100
    deterministic = False
    representation = "mixed"
    # mixed or vector
    
    env, eval_env = make_envs(my_reward_function, mask_fn, representation)

    test_output(env, eval_env)

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

    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        mask_fn=mask_fn,  # Your mask function
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_timesteps,
        n_eval_episodes=eval_episodes,
        deterministic=deterministic,
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

    # Use multiple callbacks
    # callback = CallbackList([eval_callback, DebugCallback()])
    # model.learn(total_timesteps=train_timesteps, callback=callback)
    model.learn(total_timesteps=train_timesteps, callback=eval_callback)
    eval_results = eval_callback.get_eval_results()
    print(f"Completed {len(eval_results)} evaluation sessions")
    
    best_model_path = load_path + ".zip"
    if os.path.exists(best_model_path):
        best_model = MaskablePPO.load(load_path, env=env)
        print("Best model loaded successfully")
    else:
        print("No best model found")

    print(model.policy)

if __name__ == "__main__":
    main()