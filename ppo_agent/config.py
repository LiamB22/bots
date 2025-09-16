representations = {
    0:"vector", 
    1:"mixed"
    }
model_names = {
    0:"mlp_ppo", # vector only models
    1:"cnn_ppo", 2:"combined_ppo" # mixed only models
    }
map_types = {0:"base", 1:"mini", 2:"tournament"}
enemy_types = {"W":0,"AB":1,"MCTS":2,"VF":3,"MIX":4} # AB uses a default depth of 2
representation = representations[1]
model_name = model_names[2] # remember to change representation when changing the model if needed
enemy_type = enemy_types["W"]
map_type = map_types[1]
vps_to_win = 6
num_enemies = 2
num_players = num_enemies + 1
train_timesteps = 500_000
eval_episodes = 1_000
rewards = {"win":100, "lose":-100, "l_positive":2, "s_negative":-0.1, "none":0}
complete_name = f"{model_name}_{train_timesteps}"
model_policy = "PPO"
save_path = f"bots/models/{model_policy}/{model_name}/{complete_name}.zip"