train_timesteps = 100_000
representations = ["vector", "mixed"]
representation = representations[0]
model_names = ["mlp_ppo", "cnn_ppo", "combined_ppo"]
model_name = model_names[0]
model_dir = "bots/models/PPO"
best_model_name = "best_model"
episodes = 100
