import random
import gymnasium
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

class DQNCNN(nn.Module):
    def __init__(self, board_shape, numeric_dim, action_dim=290):
        super(DQNCNN, self).__init__()
        
        # Board shape is (20, 21, 11) - we need to treat 20 as channels
        self.conv1 = nn.Conv2d(board_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the flattened size after convolutions
        self._calculate_conv_output_size(board_shape)
        
        # Dense layers for numeric features
        self.numeric_fc1 = nn.Linear(numeric_dim, 128)
        self.numeric_fc2 = nn.Linear(128, 64)
        
        # Combined layers
        combined_size = self.conv_output_size + 64  # CNN output + numeric features
        self.fc1 = nn.Linear(combined_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)  # 290 actions
        
        self.dropout = nn.Dropout(0.2)
        
    def _calculate_conv_output_size(self, board_shape):
        # Mock forward pass to calculate output size
        x = torch.zeros(1, *board_shape)  # Shape: (1, 20, 21, 11)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.conv_output_size = x.view(1, -1).size(1)
        
    def forward(self, board, numeric):
        # Process board with CNN
        # board shape: (batch_size, 20, 21, 11)
        x_board = F.relu(self.conv1(board))
        x_board = F.relu(self.conv2(x_board))
        x_board = F.relu(self.conv3(x_board))
        x_board = x_board.view(x_board.size(0), -1)  # Flatten
        
        # Process numeric features
        x_numeric = F.relu(self.numeric_fc1(numeric))
        x_numeric = self.dropout(x_numeric)
        x_numeric = F.relu(self.numeric_fc2(x_numeric))
        
        # Combine features
        combined = torch.cat([x_board, x_numeric], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, valid_actions):
        self.buffer.append((state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, valid_actions = zip(*batch)
        
        # Convert to tensors
        board_states = torch.FloatTensor(np.array([s['board'] for s in state]))
        numeric_states = torch.FloatTensor(np.array([s['numeric'] for s in state]))
        actions = torch.LongTensor(action)
        rewards = torch.FloatTensor(reward)
        next_board_states = torch.FloatTensor(np.array([s['board'] for s in next_state]))
        next_numeric_states = torch.FloatTensor(np.array([s['numeric'] for s in next_state]))
        dones = torch.FloatTensor(done)
        
        return (board_states, numeric_states), actions, rewards, (next_board_states, next_numeric_states), dones, valid_actions
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, board_shape, numeric_dim, action_dim=290, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, buffer_size=50000, batch_size=64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_shape = board_shape  # (18, 21, 11)
        self.numeric_dim = numeric_dim  # 62
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = DQNCNN(board_shape, numeric_dim, action_dim).to(self.device)
        self.target_net = DQNCNN(board_shape, numeric_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()
        
        self.update_count = 0
        
    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            # Convert state to proper format
            board = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
            numeric = torch.FloatTensor(state['numeric']).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(board, numeric)
            
            # Create mask for valid actions
            mask = torch.full((self.action_dim,), -float('inf'), device=self.device)
            for action in valid_actions:
                if action < self.action_dim:  # Ensure action is within bounds
                    mask[action] = 0
                
            # Apply mask and select best valid action
            masked_q_values = q_values + mask
            return masked_q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self):
        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None:
            return None
        
        (board_states, numeric_states), actions, rewards, (next_board_states, next_numeric_states), dones, valid_actions_list = sample_result
        
        # Move to device
        board_states = board_states.to(self.device)
        numeric_states = numeric_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_board_states = next_board_states.to(self.device)
        next_numeric_states = next_numeric_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.policy_net(board_states, numeric_states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network with valid action masking
        with torch.no_grad():
            next_q_values = self.target_net(next_board_states, next_numeric_states)
            
            # Apply valid action masks
            next_q = torch.zeros_like(next_q_values[:, 0])
            for i in range(self.batch_size):
                valid_next_actions = valid_actions_list[i]
                if valid_next_actions:
                    # Filter valid actions that are within bounds
                    valid_actions_filtered = [a for a in valid_next_actions if a < self.action_dim]
                    if valid_actions_filtered:
                        next_q[i] = next_q_values[i, valid_actions_filtered].max()
                    else:
                        next_q[i] = 0
                else:
                    next_q[i] = 0  # No valid actions (terminal state)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_epsilon()
        self.update_count += 1
        
        # Update target network periodically
        if self.update_count % 1000 == 0:
            self.update_target_network()
            
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class CatanatronDQNBot:
    def __init__(self, model_path=None):
        # Define dimensions based on your observation shape
        board_shape = (18, 21, 11)  # Channels, height, width
        numeric_dim = 62 # From your numeric array length
        
        self.agent = DQNAgent(board_shape, numeric_dim)
        
        if model_path:
            self.agent.load(model_path)
        
        self.current_state = None
        self.last_action = None
        self.current_valid_actions = None
        self.episode_rewards = []
        
    def decide(self, observation, valid_actions):
        # Convert observation to the format your agent expects
        state = {
            'board': observation['board'],
            'numeric': observation['numeric']
        }
        
        # Select action
        action = self.agent.select_action(state, valid_actions)
        
        # Store for training
        self.current_state = state
        self.last_action = action
        self.current_valid_actions = valid_actions
        
        return action
    
    def record_experience(self, next_observation, reward, done):
        if self.current_state is not None:
            next_state = {
                'board': next_observation['board'],
                'numeric': next_observation['numeric']
            }
            
            # Store experience in replay buffer
            self.agent.memory.push(
                self.current_state, 
                self.last_action, 
                reward, 
                next_state, 
                done,
                self.current_valid_actions
            )
            
            # Train the agent
            loss = self.agent.train()
            
            if done:
                self.episode_rewards.append(reward)
                self.current_state = None
                self.current_valid_actions = None
                
            return loss
        return None

# Training function
def train_dqn_bot(num_episodes=1000, save_interval=100):
    # Create environment
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "enemies": [
                WeightedRandomPlayer(Color.RED),
                WeightedRandomPlayer(Color.ORANGE),
            ],
            "reward_function": my_reward_function,
            "representation": "mixed",
        },
    )
    
    bot = CatanatronDQNBot()
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action = bot.decide(observation, info["valid_actions"])
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record experience and train
            loss = bot.record_experience(next_observation, reward, done)
            
            total_reward += reward
            step_count += 1
            observation = next_observation
            
            if done:
                break
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(bot.episode_rewards[-10:]) if len(bot.episode_rewards) >= 10 else total_reward
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Avg Reward (last 10): {avg_reward:.2f}, "
                  f"Epsilon: {bot.agent.epsilon:.4f}, Steps: {step_count}")
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            bot.agent.save(f"catan_dqn_model_episode_{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    env.close()
    return bot

# Evaluation function
def evaluate_bot(model_path, num_games=10):
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "map_type": "MINI",
            "vps_to_win": 6,
            "enemies": [
                WeightedRandomPlayer(Color.RED),
                WeightedRandomPlayer(Color.ORANGE),
            ],
            "reward_function": my_reward_function,
            "representation": "mixed",
        },
    )
    
    bot = CatanatronDQNBot(model_path)
    bot.agent.epsilon = 0.01  # Minimal exploration for evaluation
    
    wins = 0
    total_rewards = []
    
    for game in range(num_games):
        observation, info = env.reset()
        total_reward = 0
        
        while True:
            action = bot.decide(observation, info["valid_actions"])
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                if reward == 100:  # Win
                    wins += 1
                total_rewards.append(total_reward)
                break
    
    win_rate = wins / num_games
    avg_reward = np.mean(total_rewards)
    
    print(f"Evaluation Results: Win Rate: {win_rate:.2f}, Average Reward: {avg_reward:.2f}")
    env.close()
    return win_rate, avg_reward

if __name__ == "__main__":
    # Start training
    trained_bot = train_dqn_bot(num_episodes=1000)
    
    # Save final model
    trained_bot.agent.save("catan_dqn_final_model.pth")
    
    # Evaluate the trained model
    evaluate_bot("catan_dqn_final_model.pth")