import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

class MTGCNN(nn.Module):
    def __init__(self, input_channels: int = 8, num_actions: int = 6):
        """
        Initialize the CNN model for MTG.
        
        Args:
            input_channels: Number of channels in the input tensor (default: 8)
            num_actions: Number of possible actions (default: 6)
        """
        super(MTGCNN, self).__init__()
        
        # Define action space
        self.action_space = [
            "cast_land",
            "cast_spell",
            "declare_attackers",
            "declare_blockers",
            "end_turn",
            "next_phase"
        ]
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Output layers
        self.action_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(-1, 128 * 5 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Output heads
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = torch.tanh(self.value_head(x))
        
        return action_probs, state_value
    
    def get_action(self, state: np.ndarray, valid_actions: List[str]) -> Tuple[str, Dict]:
        """
        Get the best action for the current state.
        
        Args:
            state: Game state tensor
            valid_actions: List of valid actions in the current state
            
        Returns:
            Tuple of (action_type, action_params)
        """
        self.eval()
        with torch.no_grad():
            # Convert state to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action probabilities and state value
            action_probs, _ = self(state_tensor)
            
            # Mask invalid actions
            action_probs = action_probs.squeeze(0)
            for i, action in enumerate(self.action_space):
                if action not in valid_actions:
                    action_probs[i] = float('-inf')
            
            # Get the best valid action
            best_action_idx = torch.argmax(action_probs).item()
            return self.action_space[best_action_idx]
    
    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))
        
class MTGAgent:
    def __init__(self, model: MTGCNN, learning_rate: float = 0.001):
        """
        Initialize the MTG agent.
        
        Args:
            model: The CNN model
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.action_space = [
            "cast_land",
            "cast_spell",
            "declare_attackers",
            "declare_blockers",
            "end_turn",
            "next_phase"
        ]
        
    def select_action(self, state: np.ndarray, valid_actions: List[str], epsilon: float = 0.1) -> Tuple[str, Dict]:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Game state tensor
            valid_actions: List of valid actions
            epsilon: Exploration rate
            
        Returns:
            Tuple of (action_type, action_params)
        """
        if np.random.random() < epsilon:
            # Random action
            valid_indices = [i for i, action in enumerate(self.action_space) 
                           if action in valid_actions]
            return self.action_space[np.random.choice(valid_indices)]
        else:
            # Best action
            return self.model.get_action(state, valid_actions)
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               rewards: torch.Tensor, next_states: torch.Tensor, 
               dones: torch.Tensor, gamma: float = 0.99):
        """
        Update the model using experience replay.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
        """
        self.model.train()
        
        # Get current Q-values
        current_action_probs, current_values = self.model(states)
        current_q_values = current_values.squeeze(-1)
        
        # Get next Q-values
        with torch.no_grad():
            next_action_probs, next_values = self.model(next_states)
            next_q_values = next_values.squeeze(-1)
        
        # Calculate target Q-values
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # Calculate losses
        value_loss = F.mse_loss(current_q_values, target_q_values)
        policy_loss = -torch.mean(torch.sum(actions * torch.log(current_action_probs + 1e-8), dim=1))
        
        # Total loss
        loss = value_loss + policy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 