import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from cards import Card, Creature, Land, Color
import random
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model: int, max_len: int = 120):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class CardEmbedding(nn.Module):
    """Embedding layer for card features."""
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

class TransformerEncoder(nn.Module):
    """Transformer encoder for processing game state."""
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        return self.transformer_encoder(x)

class MTGTransformer(nn.Module):
    """Transformer-based model for MTG game state processing."""
    def __init__(self, input_channels=13, num_actions=7, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Model parameters
        self.input_channels = input_channels  # Now 13 features per card
        self.d_model = d_model
        self.num_actions = num_actions  # Now 7 actions including tap_land
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_cards = 120  # Maximum number of cards to handle
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(8)
        print(self.device)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Action space will be set during initialization
        self.action_space = None
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Zone embeddings (8 zones: library, hand, battlefield, graveyard, exile, stack for both players)
        self.zone_embeddings = nn.Parameter(torch.randn(8, d_model))
        
        # Zone-specific attention heads
        self.zone_attention = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.max_cards)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Increased feed-forward dimension
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, state, zone_indices=None):
        """
        Forward pass through the model.
        Args:
            state: Game state tensor of shape (batch_size, num_cards, num_features)
                  or numpy array of same shape
            zone_indices: Tensor of shape (batch_size, num_cards) indicating which zone each card belongs to
        Returns:
            Tuple of (action_logits, value) where:
            - action_logits has shape (batch_size, num_actions)
            - value has shape (batch_size, 1)
        """
        # Convert numpy array to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        
        # Handle different input shapes
        if state.dim() == 4:
            # Shape: (batch_size, num_cards, height, width)
            batch_size, num_cards, height, width = state.shape
            num_features = height * width
            state = state.reshape(batch_size, num_cards, num_features)
        elif state.dim() == 3:
            # Shape: (batch_size, num_cards, num_features)
            batch_size, num_cards, num_features = state.shape
        elif state.dim() == 2:
            # Shape: (num_cards, num_features)
            num_cards, num_features = state.shape
            batch_size = 1
            state = state.unsqueeze(0)
        elif state.dim() == 1:
            # Shape: (num_features,)
            num_features = state.shape[0]
            batch_size = 1
            num_cards = 1
            state = state.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state tensor shape: {state.shape}")
        
        # Project input to model dimension
        x = state.reshape(-1, num_features)  # Flatten to (batch_size * num_cards, num_features)
        x = self.input_proj(x)  # Project to (batch_size * num_cards, d_model)
        x = x.reshape(batch_size, num_cards, -1)  # Reshape back to (batch_size, num_cards, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add zone embeddings if provided
        if zone_indices is not None:
            zone_emb = self.zone_embeddings[zone_indices]  # (batch_size, num_cards, d_model)
            x = x + zone_emb
            
            # Apply zone-specific attention
            zone_attn_output, _ = self.zone_attention(x, x, x)
            x = x + zone_attn_output  # Residual connection
        
        # Create attention mask for padding if needed
        if num_cards < self.max_cards:
            padding_mask = torch.ones(batch_size, num_cards, dtype=torch.bool)
            padding_mask[:, :num_cards] = False
        else:
            padding_mask = None
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global pooling (only over non-padded tokens)
        if padding_mask is not None:
            mask = ~padding_mask.unsqueeze(-1)  # (batch_size, num_cards, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, d_model)
        else:
            x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Get action logits and value
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        return action_logits, value
    
    def select_action(self, state, valid_actions, epsilon=None):
        """
        Select an action using epsilon-greedy strategy.
        Args:
            state: Current game state
            valid_actions: List of valid action strings or tuples
            epsilon: Optional override for exploration rate
        Returns:
            Selected action as a string or tuple
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        # Handle both string and tuple actions
        action_strings = [action[0] if isinstance(action, tuple) else action for action in valid_actions]
        
        # Convert valid actions to indices
        valid_indices = [self.action_space.index(action) for action in action_strings]
        
        # Random action with probability epsilon
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            return action
        
        # Get action logits
        with torch.no_grad():
            logits, _ = self.forward(state)
            
            # Create mask for valid actions
            mask = torch.zeros_like(logits)
            mask[0, valid_indices] = 1.0
            
            # Apply mask to logits
            masked_logits = logits * mask
            
            # Set invalid actions to negative infinity
            masked_logits[mask == 0] = float('-inf')
            
            # Get probabilities
            probs = F.softmax(masked_logits, dim=-1)
            
            # Select action with highest probability
            action_idx = probs[0].argmax().item()
            selected_action = self.action_space[action_idx]
            
            # Find the corresponding tuple action if it exists
            for action in valid_actions:
                if isinstance(action, tuple) and action[0] == selected_action:
                    selected_action = action
                    break
            
        return selected_action
    
    def save(self, path):
        """Save model state to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'action_space': self.action_space,
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path):
        """Load model state from file."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.action_space = checkpoint['action_space']
        self.epsilon = checkpoint['epsilon'] 