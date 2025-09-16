rep_and_models = {
    "vector":["mlp_ppo"], # vector only models
    "mixed":["cnn_ppo", "combined_ppo"] # mixed only models
    }
map_types = {0:"base", 1:"mini", 2:"tournament"}
enemy_types = {"W":0,"AB":1,"MCTS":2,"VF":3,"MIX":4} # AB uses a default depth of 2
rewards = {"win":100, "lose":-100, "l_positive":2, "s_negative":-0.01, "none":0}
representations = list(rep_and_models.keys())
representation = representations[1] #change representation
model_names = rep_and_models[representation]
model_name = model_names[1] #change model
enemy_type = enemy_types["W"] #change enemy
map_type = map_types[1] #change map
vps_to_win = 6 #chane number of victory points required to win
num_enemies = 2 #change the number of enemy bots
num_players = num_enemies + 1
train_timesteps = 1 #change how long to train agent for
eval_episodes = 1 #change how many episodes the agent is evaluated for
complete_name = f"{model_name}_{train_timesteps}"
model_policy = "PPO"
save_path = f"bots/models/{model_policy}/{model_name}/{complete_name}.zip"