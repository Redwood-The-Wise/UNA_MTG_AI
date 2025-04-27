import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from cards import Card, Creature, Land, Color
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class BaseModel(nn.Module):
    """Base class for all models with common training optimizations."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()  # For mixed precision training
        self.optimizer = None
        self.scheduler = None
        
    def to(self, device=None):
        """Override to method to ensure device consistency."""
        if device is None:
            device = self.device
        super().to(device)
        self.device = device
        return self
        
    def setup_optimizer(self, learning_rate: float = 0.001):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
    def train_step(self, loss: torch.Tensor) -> float:
        """Perform a training step with mixed precision and gradient clipping."""
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer first.")
            
        # Ensure loss is on the correct device and dtype
        loss = loss.to(self.device).float()
            
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Scale loss and backpropagate with mixed precision
        with autocast(device_type=self.device.type):
            scaled_loss = loss * 1.0  # Scale factor can be adjusted
            
        # Backward pass with gradient scaling
        self.scaler.scale(scaled_loss).backward()
        
        # Clip gradients to prevent exploding gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return scaled_loss.item()
        
    def update_learning_rate(self, metric: float):
        """Update learning rate based on performance metric."""
        if self.scheduler is not None:
            self.scheduler.step(metric)
            
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure a tensor is on the correct device and dtype."""
        return tensor.to(self.device).float()

class DecisionModel(BaseModel):
    """Main model that decides which action to take."""
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.actual_input_size = 120 * 20
        
        # Enhanced network architecture with batch normalization
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * self.actual_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Reshape input for convolutional layers
        x = x.view(-1, 1, self.actual_input_size)
        
        # Apply convolutional layers with batch normalization and dropout
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
    def select_action(self, state: torch.Tensor, valid_actions: List[str]) -> str:
        """Select an action based on the current state and valid actions."""
        with torch.no_grad():
            # Get action probabilities
            action_probs = self.forward(state)
            
            # Create a mask for valid actions
            action_mask = torch.zeros(self.action_size, device=state.device)
            for action in valid_actions:
                action_idx = self.action_space.index(action)
                action_mask[action_idx] = 1
                
            # Apply mask and softmax
            masked_probs = action_probs * action_mask
            if masked_probs.sum() == 0:
                # If all valid actions have zero probability, use uniform distribution
                masked_probs = action_mask / action_mask.sum()
            else:
                masked_probs = F.softmax(masked_probs, dim=0)
                
            # Select action with highest probability
            action_idx = torch.argmax(masked_probs).item()
            return self.action_space[action_idx]
            
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw action probabilities for all actions."""
        with torch.no_grad():
            return F.softmax(self.forward(state), dim=0)

class CastSpellModel(BaseModel):
    """Model for deciding which spell to cast."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.actual_input_size = 120 * 20
        
        self.network = nn.Sequential(
            nn.Linear(self.actual_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.actual_input_size)
        return self.network(x)
        
    def select_spell(self, state: torch.Tensor, castable_spells: List[Tuple[Card, float]]) -> Optional[Card]:
        """Select a spell to cast based on the current state."""
        if not castable_spells:
            return None
            
        # Get value for each spell
        spell_values = []
        for spell, _ in castable_spells:
            # Create input tensor with spell features
            spell_features = np.array([
                spell.power if isinstance(spell, Creature) else 0,
                spell.toughness if isinstance(spell, Creature) else 0,
                len(spell.parse_mana_cost()),
                sum(spell.parse_mana_cost().values())
            ])
            input_tensor = torch.cat([
                state,
                torch.FloatTensor(spell_features).view(1, -1)
            ], dim=1)
            
            # Get value from model
            value = self.forward(input_tensor).item()
            spell_values.append((spell, value))
            
        # Sort by value and select best spell
        spell_values.sort(key=lambda x: x[1], reverse=True)
        return spell_values[0][0] if spell_values else None

class DeclareAttackersModel(BaseModel):
    """Model for deciding which creatures to attack with."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.actual_input_size = 120 * 20  # Base state tensor dimensions
        self.creature_feature_size = 12  # Additional creature features
        self.total_input_size = self.actual_input_size + self.creature_feature_size
        
        self.network = nn.Sequential(
            nn.Linear(self.total_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened to (batch_size, total_input_size) and float32
        x = x.view(-1, self.total_input_size).float()
        return self.network(x)
        
    def select_attackers(self, state: torch.Tensor, available_creatures: List[Creature], game) -> List[Creature]:
        """Select which creatures to attack with."""
        if not available_creatures:
            return []
            
        # Create state for each creature
        creature_states = []
        for creature in available_creatures:
            creature_state = self._create_creature_state(state, creature, game)
            creature_states.append(creature_state)
                
        # Convert to tensor and ensure all tensors are on the same device and dtype
        creature_states = torch.stack(creature_states).to(self.device).float()
        
        # Get values from network
        with torch.no_grad():
            values = self.forward(creature_states)
        
        # Calculate potential damage and survival rate
        potential_damage = sum(creature.power for creature in available_creatures)
        total_toughness = sum(creature.toughness for creature in available_creatures)
        
        # Get opponent's potential blockers
        opponent_has_blockers = any(not c.tapped and isinstance(c, Creature) 
                                  for c in game.players[1].battlefield)
        
        # Adjust values based on game state
        for i, creature in enumerate(available_creatures):
            # Base value from network
            attack_value = values[i].item()
            
            # Add bonuses based on game state
            if not opponent_has_blockers:
                attack_value += 0.5
            if len(available_creatures) > len([c for c in game.players[1].battlefield 
                                             if isinstance(c, Creature)]):
                attack_value += 0.3
            if creature.power >= 3:
                attack_value += 0.4
                
            values[i] = attack_value
            
        # Select creatures with positive attack value
        selected_creatures = []
        for i, creature in enumerate(available_creatures):
            if values[i] > -0.05 or (not opponent_has_blockers and values[i] > -0.2):
                selected_creatures.append(creature)
                
        return selected_creatures
        
    def _create_creature_state(self, game_state: torch.Tensor, creature: Creature, game) -> torch.Tensor:
        """Create a state representation for a specific creature."""
        # Get the opponent from the game
        current_player = creature.controller
        opponent = None
        for player in game.players:
            if player != current_player:
                opponent = player
                break
        
        # Get potential blockers
        potential_blockers = [c for c in opponent.battlefield 
                            if isinstance(c, Creature) and not c.tapped]
        
        # Calculate average blocker stats
        avg_blocker_power = np.mean([b.power for b in potential_blockers]) if potential_blockers else 0
        avg_blocker_toughness = np.mean([b.toughness for b in potential_blockers]) if potential_blockers else 0
        
        # Calculate potential counter-attack damage
        counter_attack_damage = sum(c.power for c in opponent.battlefield 
                                  if isinstance(c, Creature) and not c.tapped)
        
        # Create creature features on the same device and dtype as game_state
        creature_features = torch.tensor([
            creature.power,
            creature.toughness,
            len(creature.parse_mana_cost()),
            sum(creature.parse_mana_cost().values()),
            len([c for c in current_player.battlefield if isinstance(c, Creature) and not c.tapped]),  # Available attackers
            len(potential_blockers),  # Potential blockers
            avg_blocker_power,  # Average blocker power
            avg_blocker_toughness,  # Average blocker toughness
            float(creature.power > avg_blocker_toughness),  # Can kill average blocker
            float(creature.toughness > avg_blocker_power),  # Can survive average blocker
            float(creature.power == avg_blocker_toughness),  # Can trade with average blocker
            float(creature.power > 0 and not potential_blockers)  # Can deal damage if unblocked
        ], device=game_state.device, dtype=torch.float32)
        
        # Concatenate game state with creature features
        return torch.cat([game_state.view(-1), creature_features])

class DeclareBlockersModel(BaseModel):
    """Model for deciding which creatures to block with."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.actual_input_size = 120 * 20  # Base state tensor dimensions
        self.blocker_feature_size = 16  # Blocker features
        self.attacker_feature_size = 6  # Attacker features
        self.total_input_size = self.actual_input_size + self.blocker_feature_size + self.attacker_feature_size
        
        self.network = nn.Sequential(
            nn.Linear(self.total_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened to (batch_size, total_input_size) and float32
        x = x.view(-1, self.total_input_size).float()
        return self.network(x)
        
    def select_blockers(self, state: torch.Tensor, 
                       potential_blockers: List[Creature],
                       attackers: List[Creature],
                       game) -> Dict[Creature, Creature]:
        """Select which creatures to block with and assign them to attackers."""
        if not potential_blockers or not attackers:
            return {}
            
        with torch.no_grad():
            blocking_assignments = {}
            available_blockers = potential_blockers.copy()
            
            # Sort attackers by power to prioritize blocking stronger attackers
            sorted_attackers = sorted(attackers, key=lambda x: x.power, reverse=True)
            
            for attacker in sorted_attackers:
                if not available_blockers:
                    break
                    
                # Create state for each potential blocker
                blocker_states = []
                for blocker in available_blockers:
                    blocker_state = self._create_blocking_state(state, blocker, attacker, game)
                    blocker_states.append(blocker_state)
                    
                # Convert to tensor and ensure all tensors are on the same device and dtype
                blocker_states = torch.stack(blocker_states).to(self.device).float()
                
                # Get values from network
                values = self.forward(blocker_states)
                
                # Calculate potential outcomes for each blocker
                for i, blocker in enumerate(available_blockers):
                    # Calculate the trade
                    if blocker.toughness > attacker.power:
                        # Blocker survives
                        values[i] += 5.0
                    elif blocker.toughness == attacker.power:
                        # Both die
                        values[i] += 3.0
                    else:
                        # Blocker dies
                        values[i] += 0.5
                    
                    # Additional bonuses for good trades
                    if blocker.power >= attacker.toughness:
                        values[i] += 3.0
                    if blocker.toughness > attacker.power:
                        values[i] += 2.0
                    
                    # Bonus for preventing lethal damage
                    defending_player = blocker.controller
                    if defending_player.life <= attacker.power:
                        values[i] += 8.0
                
                # Select the best blocker
                best_idx = torch.argmax(values).item()
                if values[best_idx] > -0.5:
                    chosen_blocker = available_blockers[best_idx]
                    blocking_assignments[chosen_blocker] = attacker
                    available_blockers.pop(best_idx)
                    
            return blocking_assignments
            
    def _create_blocking_state(self, game_state: torch.Tensor, 
                             blocker: Creature, attacker: Creature,
                             game) -> torch.Tensor:
        """Create a state representation for a specific blocker-attacker pair."""
        # Get the defending player (blocker's controller)
        defending_player = blocker.controller
        attacking_player = attacker.controller
        
        # Calculate combat outcomes
        can_kill = blocker.power >= attacker.toughness
        can_survive = blocker.toughness > attacker.power
        will_trade = blocker.toughness == attacker.power
        
        # Get other potential blockers
        other_blockers = [c for c in defending_player.battlefield 
                         if isinstance(c, Creature) and not c.tapped and c != blocker]
        
        # Get other attackers
        other_attackers = [a for a in game.combat_state.get('attackers', []) 
                          if a != attacker]
        
        # Calculate average stats
        avg_blocker_power = np.mean([b.power for b in other_blockers]) if other_blockers else 0
        avg_blocker_toughness = np.mean([b.toughness for b in other_blockers]) if other_blockers else 0
        avg_attacker_power = np.mean([a.power for a in other_attackers]) if other_attackers else 0
        avg_attacker_toughness = np.mean([a.toughness for a in other_attackers]) if other_attackers else 0
        
        # Create blocker features
        blocker_features = torch.tensor([
            blocker.power,
            blocker.toughness,
            len(blocker.parse_mana_cost()),
            sum(blocker.parse_mana_cost().values()),
            len(other_blockers),  # Available other blockers
            len(other_attackers),  # Other attackers
            avg_blocker_power,  # Average other blocker power
            avg_blocker_toughness,  # Average other blocker toughness
            avg_attacker_power,  # Average other attacker power
            avg_attacker_toughness,  # Average other attacker toughness
            float(can_kill),  # Can kill the attacker
            float(can_survive),  # Can survive the attack
            float(will_trade),  # Will trade with attacker
            float(blocker.power > avg_attacker_toughness),  # Can kill average attacker
            float(blocker.toughness > avg_attacker_power),  # Can survive average attacker
            float(blocker.power == avg_attacker_toughness)  # Can trade with average attacker
        ], device=game_state.device, dtype=torch.float32)
        
        # Create attacker features
        attacker_features = torch.tensor([
            attacker.power,
            attacker.toughness,
            len(attacker.parse_mana_cost()),
            sum(attacker.parse_mana_cost().values()),
            float(attacker.power > avg_blocker_toughness),  # Can kill average blocker
            float(attacker.toughness > avg_blocker_power)  # Can survive average blocker
        ], device=game_state.device, dtype=torch.float32)
        
        # Concatenate all features
        return torch.cat([game_state.view(-1), blocker_features, attacker_features])

class TapLandsModel(BaseModel):
    """Model for selecting which lands to tap."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.actual_input_size = 120 * 20  # Match the state tensor dimensions
        
        self.network = nn.Sequential(
            nn.Linear(self.actual_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened to (batch_size, actual_input_size)
        x = x.view(-1, self.actual_input_size)
        return self.network(x)
        
    def select_lands(self, state: np.ndarray, available_lands: List[Land]) -> List[Land]:
        """Select lands to tap based on the current state."""
        if not available_lands:
            return []
            
        # Convert state to tensor and ensure correct shape
        state_tensor = torch.FloatTensor(state).view(1, -1)
        
        # Get value for each land
        land_values = []
        for land in available_lands:
            # Create input tensor with land features
            land_features = np.array([
                land.tapped,
                land.color.value if hasattr(land.color, 'value') else 0,
                land.mana_value
            ])
            input_tensor = torch.cat([
                state_tensor,
                torch.FloatTensor(land_features).view(1, -1)
            ], dim=1)
            
            # Get value from model
            value = self.forward(input_tensor).item()
            land_values.append((land, value))
            
        # Sort by value and select top lands
        land_values.sort(key=lambda x: x[1], reverse=True)
        return [land for land, _ in land_values]

class SelectLandModel(BaseModel):
    """Model for selecting which land to play."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.actual_input_size = 120 * 20  # Match the state tensor dimensions
        
        self.network = nn.Sequential(
            nn.Linear(self.actual_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is flattened to (batch_size, actual_input_size)
        x = x.view(-1, self.actual_input_size)
        return self.network(x)
        
    def select_land(self, state: torch.Tensor, available_lands: List[Land]) -> Optional[Land]:
        """Select a land to play based on the current state."""
        if not available_lands:
            return None
            
        # Get value for each land
        land_values = []
        for land in available_lands:
            # Create input tensor with land features
            land_features = np.array([
                land.tapped,
                land.color.value if hasattr(land.color, 'value') else 0,
                land.mana_value
            ])
            input_tensor = torch.cat([
                state,
                torch.FloatTensor(land_features).view(1, -1)
            ], dim=1)
            
            # Get value from model
            value = self.forward(input_tensor).item()
            land_values.append((land, value))
            
        # Sort by value and select best land
        land_values.sort(key=lambda x: x[1], reverse=True)
        return land_values[0][0] if land_values else None 

class ModelManager:
    """Manages all models for a player."""
    def __init__(self, state_size: int, action_space: List[str], checkpoint_dir: str = "checkpoints"):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models with optimizers
        self.decision_model = DecisionModel(state_size, self.action_size).to(self.device)
        self.cast_spell_model = CastSpellModel(state_size).to(self.device)
        self.declare_attackers_model = DeclareAttackersModel(state_size).to(self.device)
        self.declare_blockers_model = DeclareBlockersModel(state_size).to(self.device)
        self.tap_lands_model = TapLandsModel(state_size).to(self.device)
        self.select_land_model = SelectLandModel(state_size).to(self.device)
        
        # Setup optimizers for all models
        for model in [self.decision_model, self.cast_spell_model, 
                     self.declare_attackers_model, self.declare_blockers_model,
                     self.tap_lands_model, self.select_land_model]:
            model.setup_optimizer()
        
        # Track best rewards
        self.best_rewards = {
            'decision': float('-inf'),
            'cast_spell': float('-inf'),
            'declare_attackers': float('-inf'),
            'declare_blockers': float('-inf'),
            'tap_lands': float('-inf'),
            'land': float('-inf')
        }
        
        # Load best models if they exist
        self.load_best_models()
        
    def select_action(self, state: torch.Tensor, valid_actions: List[str]) -> str:
        """Select an action based on the current state."""
        with torch.no_grad():
            # Ensure state is on the correct device and flattened
            state = state.to(self.device)
            state = state.view(1, -1)  # Flatten to (1, state_size)
            
            action_probs = self.decision_model(state)
            
            # Create a mask for valid actions on the same device
            action_mask = torch.zeros(self.action_size, device=self.device)
            for action in valid_actions:
                action_idx = self.action_space.index(action)
                action_mask[action_idx] = 1
                
            # Apply mask and select action
            masked_probs = action_probs * action_mask
            if masked_probs.sum() == 0:
                return "next_phase"  # Default action if no valid actions
                
            action_idx = torch.argmax(masked_probs).item()
            return self.action_space[action_idx]
            
    def handle_cast_spell(self, state: torch.Tensor, castable_spells: List[Tuple[Card, float]]) -> Optional[Card]:
        """Select a spell to cast using the cast spell model."""
        # Ensure state is on the correct device
        state = state.to(self.device)
        return self.cast_spell_model.select_spell(state, castable_spells)
        
    def handle_declare_attackers(self, state: torch.Tensor, potential_attackers: List[Creature], game) -> List[Creature]:
        """Select attackers using the declare attackers model."""
        # Ensure state is on the correct device
        state = state.to(self.device)
        return self.declare_attackers_model.select_attackers(state, potential_attackers, game)
        
    def handle_declare_blockers(self, state: torch.Tensor, 
                              potential_blockers: List[Creature],
                              attackers: List[Creature],
                              game) -> Dict[Creature, Creature]:
        """Select blockers using the declare blockers model."""
        # Ensure state is on the correct device
        state = state.to(self.device)
        return self.declare_blockers_model.select_blockers(state, potential_blockers, attackers, game)
        
    def handle_tap_lands(self, state: torch.Tensor, available_lands: List[Land]) -> List[Land]:
        """Select lands to tap using the tap lands model."""
        # Ensure state is on the correct device
        state = state.to(self.device)
        return self.tap_lands_model.select_lands(state, available_lands)
        
    def select_land(self, hand, game):
        """Select a land to cast from hand."""
        # Check if a land has already been played this turn
        current_player = game.get_active_player()
        if current_player.lands_played_this_turn > 0:
            return None
            
        lands = [card for card in hand if isinstance(card, Land)]
        if not lands:
            return None
        
        # Get state tensor and ensure it's on the correct device
        state = self._get_state_tensor()
        state = torch.FloatTensor(state).to(self.device)
        
        # Use the select land model to choose a land
        return self.select_land_model.select_land(state, lands)
        
    def _get_state_tensor(self) -> np.ndarray:
        """Get a state tensor for the current game state."""
        # Create a simple state representation
        # This should be replaced with a proper state representation from the game
        return np.zeros(self.state_size)
        
    def update_models(self, rewards: Dict[str, float]):
        """Update learning rates based on performance."""
        for model_name, reward in rewards.items():
            model = getattr(self, f"{model_name}_model")
            model.update_learning_rate(reward)
            
    def save_checkpoint(self, episode: int, rewards: Dict[str, float]):
        """Save a complete checkpoint of all models."""
        checkpoint = {
            'episode': episode,
            'rewards': rewards
        }
        for model_name in self.best_rewards.keys():
            model = getattr(self, f"{model_name}_model")
            checkpoint[f"{model_name}_state_dict"] = model.state_dict()
            checkpoint[f"{model_name}_optimizer"] = model.optimizer.state_dict()
            checkpoint[f"{model_name}_scheduler"] = model.scheduler.state_dict()
            checkpoint[f"{model_name}_reward"] = self.best_rewards[model_name]
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.pth"))
        
    def load_checkpoint(self, episode: int):
        """Load a specific checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            for model_name in self.best_rewards.keys():
                model = getattr(self, f"{model_name}_model")
                model.load_state_dict(checkpoint[f"{model_name}_state_dict"])
                model.optimizer.load_state_dict(checkpoint[f"{model_name}_optimizer"])
                model.scheduler.load_state_dict(checkpoint[f"{model_name}_scheduler"])
                self.best_rewards[model_name] = checkpoint[f"{model_name}_reward"]
            return checkpoint['episode'], checkpoint['rewards']
        return None, None 