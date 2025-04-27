from dataclasses import dataclass
from typing import Dict, Any
import torch

@dataclass
class TrainingConfig:
    # Model parameters
    input_channels: int = 8
    num_actions: int = 5
    hidden_channels: Dict[str, int] = None
    
    # Training parameters
    num_episodes: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Memory parameters
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000
    
    # Validation parameters
    validation_episodes: int = 100
    validation_interval: int = 100
    
    # Saving parameters
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = {
                "conv1": 32,
                "conv2": 64,
                "conv3": 128,
                "fc1": 256,
                "fc2": 128
            }

# Create default configuration
config = TrainingConfig() 