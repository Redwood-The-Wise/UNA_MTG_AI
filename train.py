import torch
import numpy as np
from collections import deque
import random
import os
from typing import List, Tuple, Dict
from transformer_model import MTGTransformer
from training_env import MTGTrainingEnv
from game import Phase
from cards import Land, Creature, Color
from config import config, TrainingConfig
from logger import MTGLogger
from validate import validate_agent, evaluate_deck_building, analyze_game_decisions
from utils import get_valid_actions
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model_manager import ModelManager

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, zone_indices: np.ndarray, action: str, 
             reward: float, next_state: np.ndarray, next_zone_indices: np.ndarray, done: bool):
        """Add a new experience to memory."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, zone_indices, action, reward, next_state, next_zone_indices, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class MTGAgent:
    """Agent for playing MTG using the transformer model."""
    def __init__(self, input_channels=12, num_actions=7, d_model=256, nhead=8, num_layers=6, learning_rate=0.001):
        """Initialize the agent with a transformer model."""
        # Set device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = MTGTransformer(
            input_channels=input_channels,
            num_actions=num_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        self.target_model = MTGTransformer(
            input_channels=input_channels,
            num_actions=num_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        # Copy weights from model to target model
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Set exploration parameters
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(10000)
        
    def select_action(self, state: np.ndarray, valid_actions: List[Tuple[str, Dict]], epsilon: float = 0.1) -> Tuple[str, Dict]:
        """Select an action using epsilon-greedy policy."""
        if random.random() < epsilon:
            # Random action
            return random.choice(valid_actions)
            
        # Get action logits from model
        with torch.no_grad():
            # Ensure state is a numpy array and has correct shape
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            # Handle different state shapes
            if len(state.shape) == 2:
                # If state is (120, 12), add batch dimension
                if state.shape[1] == 12:
                    # Add phase information if missing
                    phase_info = np.zeros((state.shape[0], 1))
                    state = np.concatenate([state, phase_info], axis=1)
                state = state.reshape(1, 120, 13)
            elif len(state.shape) == 3:
                # If state is already (1, 120, 13), keep as is
                if state.shape[1:] != (120, 13):
                    raise ValueError(f"Unexpected state shape: {state.shape}")
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Get action logits from target network
            action_logits, _ = self.target_model(state_tensor)
            
            # Get valid action indices
            valid_indices = [self.model.action_space.index(action[0]) for action in valid_actions]
            
            # Create mask for valid actions
            mask = torch.full_like(action_logits, float('-inf'))
            mask[0, valid_indices] = 0
            masked_logits = action_logits + mask
            
            # Get probabilities
            probs = F.softmax(masked_logits, dim=-1)
            
            # Print probabilities for valid actions
            print("\n[Model] Action probabilities:")
            for idx in valid_indices:
                action = self.model.action_space[idx]
                prob = probs[0, idx].item()
                print(f"  {action}: {prob:.3f}")
            
            # Select action with highest probability
            best_action_idx = probs[0].argmax().item()
            
            # Find the corresponding valid action
            for action, params in valid_actions:
                if self.model.action_space.index(action) == best_action_idx:
                    print(f"[Model] Selected action: {action}")
                    return action, params
            
            # If no valid action found, return random action
            return random.choice(valid_actions)
        
    def update(self, batch: List[Tuple], gamma: float) -> float:
        """Update the model using a batch of experiences."""
        # Unpack batch
        states, zone_indices, actions, rewards, next_states, next_zone_indices, dones = zip(*batch)
        
        # Convert to tensors and ensure correct shapes
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Add phase information if missing
        if states.shape[-1] == 12:
            phase_info = np.zeros((states.shape[0], states.shape[1], 1))
            states = np.concatenate([states, phase_info], axis=2)
        if next_states.shape[-1] == 12:
            phase_info = np.zeros((next_states.shape[0], next_states.shape[1], 1))
            next_states = np.concatenate([next_states, phase_info], axis=2)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        zone_indices = torch.LongTensor(np.array(zone_indices)).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_zone_indices = torch.LongTensor(np.array(next_zone_indices)).to(self.device)
        
        # Extract action names from tuples and convert to indices
        actions = torch.tensor([self.model.action_space.index(action[0]) for action in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        
        # Get current Q-values from main network
        current_q_values, current_values = self.model(states, zone_indices)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q_values, next_values = self.target_model(next_states, next_zone_indices)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            
            # Calculate target Q-values using both immediate rewards and next state values
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
            
            # Add value function loss to stabilize training
            value_targets = rewards + (1 - dones) * gamma * next_values
        
        # Compute losses
        q_loss = F.mse_loss(current_q_values, target_q_values)
        value_loss = F.mse_loss(current_values, value_targets)
        
        # Total loss is weighted sum of Q-value and value function losses
        loss = q_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Update the target network with the current network's weights."""
        self.target_model.load_state_dict(self.model.state_dict())

# Training parameters
EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
SAVE_INTERVAL = 100

def train(env: MTGTrainingEnv, num_episodes: int = 1000, checkpoint_frequency: int = 100):
    """Train the models through self-play."""
    best_reward = float('-inf')
    
    # Create progress bar for episodes
    episode_pbar = tqdm(range(num_episodes), desc="Training Episodes")
    
    for episode in episode_pbar:
        state, _ = env.reset()  # Unpack the state tuple
        state = torch.FloatTensor(state)  # Convert to tensor
        total_reward = 0
        done = False
        step = 0
        
        # Create progress bar for steps within episode
        step_pbar = tqdm(total=1000, desc=f"Episode {episode} Steps", leave=False)
        
        while not done and step < 1000:  # Add step limit to prevent infinite loops
            # Get current player's model manager
            current_player = env.game.get_active_player()
            current_model_manager = env.player1_model_manager if current_player == env.game.players[0] else env.player2_model_manager
            
            # Get valid actions
            valid_actions = env._get_valid_actions()
            
            # Select action using model
            action = current_model_manager.select_action(state, valid_actions)
            
            # Take action and get next state
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)  # Convert next state to tensor
            
            # Update total reward
            total_reward += reward
            
            # Move to next state
            state = next_state
            step += 1
            
            # Update step progress bar
            step_pbar.update(1)
            step_pbar.set_postfix({'reward': total_reward})
            
            # Save checkpoint if we've reached a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            current_model_manager.save_checkpoint(episode, total_reward)
            episode_pbar.set_postfix({'best_reward': best_reward})
            print(f"\nNew best reward: {best_reward:.2f} at episode {episode}")
        
        # Save periodic checkpoint
        if episode % checkpoint_frequency == 0:
            current_model_manager.save_checkpoint(episode, total_reward)
            print(f"\nSaved checkpoint at episode {episode} with reward {total_reward:.2f}")
    
        # Close step progress bar
        step_pbar.close()
        
        # Update episode progress bar
        episode_pbar.set_postfix({
            'episode_reward': total_reward,
            'best_reward': best_reward,
            'steps': step
        })
        
        if episode % env.log_frequency == 0:
            env.print_game_stats()
    
    # Close episode progress bar
    episode_pbar.close()

if __name__ == "__main__":
    # Initialize environment
    env = MTGTrainingEnv()
    
    # Train models
    train(env, num_episodes=1000, checkpoint_frequency=100) 