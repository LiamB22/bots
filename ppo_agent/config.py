rep_and_models = {
    "vector":["mlp_ppo"], # vector only models. model 0, 1, 2.. etc
    "mixed":["cnn_ppo", "combined_ppo"] # mixed only models. model 0, 1, 2.. etc
    }
map_types = {0:"base", 1:"mini", 2:"tournament"}
enemy_types = {"W":0,"AB":1,"MCTS":2,"VF":3,"MIX":4} # AB uses a default depth of 2
rewards = {"win":100, "lose":-100, "l_positive":1, "s_negative":-0.01, "none":0}
representations = list(rep_and_models.keys())
representation = representations[0] #change representation
model_names = rep_and_models[representation]
model_name = model_names[0] #change model
enemy_type = enemy_types["VF"] #change enemy
map_type = map_types[1] #change map
vps_to_win = 6 #chane number of victory points required to win
num_enemies = 2 #change the number of enemy bots
num_players = num_enemies + 1
train_timesteps = 100_000 #change how long to train agent for
eval_episodes = 1_00 #change how many episodes the agent is evaluated for
train_model = True #boolean for whether the model is trained
evaluate_model = True #boolean if evaluated
show_model_policy = True #boolean for model policy
complete_name = f"{model_name}_{num_players}_{train_timesteps}"
model_policy = "PPO"
log_dir = f"bots/logs/{model_policy}/{model_name}/{complete_name}"
save_path = f"bots/models/{model_policy}/{model_name}/{complete_name}.zip"

# view training with tensorboard --logdir bots/logs/PPO/mlp_ppo/{model} 
# e.g. tensorboard --logdir bots/logs/PPO/mlp_ppo/mlp_ppo_3_100000