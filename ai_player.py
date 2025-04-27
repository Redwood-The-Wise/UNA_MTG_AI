import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from game import Game, Phase
from cards import Card, Player, Color, CardType

class MTGCNN(nn.Module):
    """Convolutional Neural Network for MTG gameplay"""
    def __init__(self):
        super(MTGCNN, self).__init__()
        
        # Input channels: 7 (hand, battlefield, graveyard, exile, stack, mana pool, opponent's battlefield)
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Output layers for different actions
        self.action_head = nn.Linear(128, 4)  # 4 main actions: cast, attack, block, activate ability
        self.target_head = nn.Linear(128, 100)  # Target selection (card index)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        
        # Action and target predictions
        actions = torch.softmax(self.action_head(x), dim=1)
        targets = torch.softmax(self.target_head(x), dim=1)
        
        return actions, targets

class AIPlayer(Player):
    """AI player that uses CNN to make decisions"""
    def __init__(self, name: str = "AI Player", model_path: Optional[str] = None):
        super().__init__(name)  # Initialize the base Player class
        self.model = MTGCNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def get_game_state_tensor(self, game: Game) -> torch.Tensor:
        """Convert game state to tensor for CNN input"""
        # Create 7-channel input tensor (8x8 grid for each channel)
        state = torch.zeros(7, 8, 8)
        
        # Channel 0: Hand
        for i, card in enumerate(self.hand[:8]):
            state[0, i//4, i%4] = self._card_to_value(card)
            
        # Channel 1: Battlefield
        for i, card in enumerate(self.battlefield[:8]):
            state[1, i//4, i%4] = self._card_to_value(card)
            
        # Channel 2: Graveyard
        for i, card in enumerate(self.graveyard[:8]):
            state[2, i//4, i%4] = self._card_to_value(card)
            
        # Channel 3: Exile
        for i, card in enumerate(self.exile[:8]):
            state[3, i//4, i%4] = self._card_to_value(card)
            
        # Channel 4: Stack
        for i, card in enumerate(game.stack[:8]):
            state[4, i//4, i%4] = self._card_to_value(card)
            
        # Channel 5: Mana Pool
        for color in Color:
            state[5, color.value, 0] = self.mana_pool[color]
            
        # Channel 6: Opponent's Battlefield
        opponent = game.players[(game.players.index(self) + 1) % 2]
        for i, card in enumerate(opponent.battlefield[:8]):
            state[6, i//4, i%4] = self._card_to_value(card)
            
        return state.unsqueeze(0)  # Add batch dimension
        
    def _card_to_value(self, card: Card) -> float:
        """Convert card to numerical value for CNN input"""
        if isinstance(card, Creature):
            return (card.power + card.toughness) / 10.0
        elif isinstance(card, Land):
            return 0.5
        else:
            return 0.3
            
    def choose_action(self, game: Game) -> Tuple[str, Optional[Card]]:
        """Choose an action based on current game state"""
        with torch.no_grad():
            state = self.get_game_state_tensor(game)
            actions, targets = self.model(state)
            
            # Get highest probability action
            action_idx = torch.argmax(actions).item()
            action_types = ["cast_land", "cast_spell", "attack", "block", "ability"]
            chosen_action = action_types[action_idx]
            
            # Get target card if needed
            target_idx = torch.argmax(targets).item()
            target_card = None
            
            if chosen_action == "cast_land":
                if target_idx < len(self.hand):
                    target_card = self.hand[target_idx]
            elif chosen_action == "cast_spell":
                if target_idx < len(self.hand):
                    target_card = self.hand[target_idx]
            elif chosen_action == "attack":
                if target_idx < len(self.battlefield):
                    target_card = self.battlefield[target_idx]
            elif chosen_action == "block":
                if target_idx < len(self.battlefield):
                    target_card = self.battlefield[target_idx]
            elif chosen_action == "ability":
                # TODO: Implement ability targeting
                pass
                
            return chosen_action, target_card
            
    def train(self, game_states: List[torch.Tensor], actions: List[int], 
              targets: List[int], epochs: int = 10):
        """Train the CNN on game data"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            total_loss = 0
            for state, action, target in zip(game_states, actions, targets):
                optimizer.zero_grad()
                action_pred, target_pred = self.model(state)
                
                loss = criterion(action_pred, torch.tensor([action])) + \
                       criterion(target_pred, torch.tensor([target]))
                       
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(game_states)}")
            
        self.model.eval()
        
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path) 