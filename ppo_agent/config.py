rep_and_models = {
    "vector":["mlp_ppo"], # "0" vector only models. model 0, 1, 2.. etc
    "mixed":["combined_ppo", "board_only", "numeric_only"] # "1" mixed only models. model 0, 1, 2.. etc
    }
map_types = {0:"base", 1:"mini", 2:"tournament"}
enemy_types = {"W":0,"AB":1,"MCTS":2,"VF":3,"MIX":4} # AB uses a default depth of 2
rewards_list = [
    # dense rewards
    {"win":100, "lose":-100, "l_positive":1, "s_positive":0.01, "s_negative":-0.01, "move_penalty":-0.5,"none":0, 
    "current_vp":0.2, "longest_road_length":0.005, "roads_left":0.01, "settlements_left":0.01, "cities_left":0.03,
    "played_dev_card":0.1, "has_largest_army":0.2, "has_longest_road":0.2},
    # alt dense rewards
    {"win":100, "lose":-100, "l_positive":1, "s_positive":0.01, "s_negative":-0.01, "move_penalty":-0.1,"none":0, 
    "current_vp":1, "longest_road_length":0.5, "roads_left":0.5, "settlements_left":0.5, "cities_left":0.5,
    "played_dev_card":1, "has_largest_army":0.5, "has_longest_road":0.5}
]
rewards = rewards_list[1]
representations = list(rep_and_models.keys())
representation = representations[1] #change representation
model_names = rep_and_models[representation]
model_name = model_names[2] #change model
enemy_type = enemy_types["W"] #change enemy
map_type = map_types[1] #change map
reward_functions = ["dense_reward_function","alt_dense_reward_function","sparse_reward_function"]
reward = reward_functions[2]
vps_to_win = 6 #chane number of victory points required to win
num_enemies = 3 #change the number of enemy bots
num_players = num_enemies + 1
train_timesteps_list = [125_000, 250_000, 500_000, 1_000_000]
train_timesteps = train_timesteps_list[3] #change how long to train agent for
eval_episodes = 1_000 #change how many episodes the agent is evaluated for
train_model = True #boolean for whether the model is trained
evaluate_model = True #boolean if evaluated
show_model_policy = False #boolean for model policy
train_further = False #train an existing model further
use_best_model = False #whether or not the best model is used to evaluate
complete_name = f"{model_name}_{num_players}_{train_timesteps}"
model_policy = "PPO"
log_dir = f"bots/logs/{model_policy}/{model_name}/{complete_name}"
save_path = f"bots/models/{model_policy}/{model_name}/{complete_name}.zip"
best_models_path = f"bots/best_models/{complete_name}.zip"

# view training with tensorboard --logdir bots/logs/PPO/mlp_ppo/{model} 
# e.g. tensorboard --logdir bots/logs/PPO/mlp_ppo/mlp_ppo_3_100000