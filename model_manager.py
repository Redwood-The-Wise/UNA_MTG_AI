import torch
from typing import Dict, List, Optional, Tuple
from models import (
    DecisionModel,
    CastSpellModel,
    DeclareAttackersModel,
    DeclareBlockersModel,
    TapLandsModel,
    SelectLandModel
)
import numpy as np
from cards import Card, Creature, Land, Color
import random
import os
import torch.optim as optim
import torch.nn as nn

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
            # Ensure state is on the correct device and dtype
            state = state.to(self.device).float()
            state = state.view(1, -1)  # Flatten to (1, state_size)
            
            action_probs = self.decision_model(state)
            
            # Create a mask for valid actions on the same device and dtype
            action_mask = torch.zeros(self.action_size, device=self.device, dtype=torch.float32)
            for action in valid_actions:
                action_idx = self.action_space.index(action)
                action_mask[action_idx] = 1
                
            # Apply mask and select action
            masked_probs = action_probs * action_mask
            if masked_probs.sum() == 0:
                return "next_phase"  # Default action if no valid actions
                
            action_idx = torch.argmax(masked_probs).item()
            return self.action_space[action_idx]
        
    def handle_cast_spell(self, state: np.ndarray, castable_spells: List[Tuple[Card, float]]) -> Optional[Card]:
        """Select a spell to cast using the cast spell model."""
        # Convert state to tensor and move to correct device
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.cast_spell_model.select_spell(state_tensor, castable_spells)
        
    def handle_declare_attackers(self, state: np.ndarray, potential_attackers: List[Creature], game) -> List[Creature]:
        """Select attackers using the declare attackers model."""
        # Convert state to tensor and move to correct device
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.declare_attackers_model.select_attackers(state_tensor, potential_attackers, game)
        
    def handle_declare_blockers(self, state: np.ndarray, 
                              potential_blockers: List[Creature],
                              attackers: List[Creature],
                              game) -> Dict[Creature, Creature]:
        """Select blockers using the declare blockers model."""
        # Convert state to tensor and move to correct device
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.declare_blockers_model.select_blockers(state_tensor, potential_blockers, attackers, game)
        
    def handle_tap_lands(self, state: np.ndarray, available_lands: List[Land]) -> List[Land]:
        """Select lands to tap using the tap lands model."""
        # Convert state to tensor and move to correct device
        state_tensor = torch.FloatTensor(state).to(self.device)
        return self.tap_lands_model.select_lands(state_tensor, available_lands)
        
    def save_models(self, episode: int, rewards: Dict[str, float]):
        """Save models if they achieve better rewards."""
        for model_name, reward in rewards.items():
            if reward > self.best_rewards[model_name]:
                self.best_rewards[model_name] = reward
                model = getattr(self, f"{model_name}_model")
                torch.save(model.state_dict(), 
                         os.path.join(self.checkpoint_dir, f"{model_name}_best.pth"))
                print(f"Saved new best {model_name} model with reward {reward}")
    
    def load_best_models(self):
        """Load the best models from checkpoints if they exist."""
        try:
            # Load the most recent checkpoint
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_ep")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_ep")[1].split(".")[0]))
                checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load each model's state dict
                for model_name, model in [
                    ('decision_model', self.decision_model),
                    ('cast_spell_model', self.cast_spell_model),
                    ('declare_attackers_model', self.declare_attackers_model),
                    ('declare_blockers_model', self.declare_blockers_model),
                    ('tap_lands_model', self.tap_lands_model),
                    ('select_land_model', self.select_land_model)
                ]:
                    if model_name in checkpoint:
                        try:
                            model.load_state_dict(checkpoint[model_name])
                            print(f"Successfully loaded {model_name}")
                        except Exception as e:
                            print(f"Error loading {model_name}: {e}")
                            print("Initializing with default weights instead")
                
                # Load best rewards if available
                if 'best_rewards' in checkpoint:
                    self.best_rewards = checkpoint['best_rewards']
                    
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with untrained models")
            
    def load_models(self, model_path: str):
        """Load models from a specific path."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            for model_name in self.best_rewards.keys():
                model = getattr(self, f"{model_name}_model")
                model.load_state_dict(checkpoint[f"{model_name}_state_dict"])
                self.best_rewards[model_name] = checkpoint[f"{model_name}_reward"]
    
    def save_checkpoint(self, episode: int, rewards: Dict[str, float]):
        """Save a complete checkpoint of all models."""
        checkpoint = {
            'episode': episode,
            'rewards': rewards
        }
        for model_name in self.best_rewards.keys():
            model = getattr(self, f"{model_name}_model")
            checkpoint[f"{model_name}_state_dict"] = model.state_dict()
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
                self.best_rewards[model_name] = checkpoint[f"{model_name}_reward"]
            return checkpoint['episode'], checkpoint['rewards']
        return None, None

    def select_land(self, hand, game):
        """Select a land to cast from hand."""
        # Check if a land has already been played this turn
        current_player = game.get_active_player()
        if current_player.lands_played_this_turn > 0:
            return None
            
        lands = [card for card in hand if isinstance(card, Land)]
        if not lands:
            return None
        
        # Prioritize lands that match the deck's color
        # For now, just return the first land
        return lands[0]

    def select_spell(self, hand, mana_pool):
        """Select a spell to cast from hand."""
        spells = [card for card in hand if not isinstance(card, Land)]
        if not spells:
            return None
        
        # Filter spells that can be cast with current mana
        castable_spells = []
        for spell in spells:
            mana_cost = spell.parse_mana_cost()
            if all(mana_pool.get(color, 0) >= cost for color, cost in mana_cost.items()):
                castable_spells.append(spell)
        
        if not castable_spells:
            return None
        
        # Prioritize creatures over other spells
        creatures = [spell for spell in castable_spells if isinstance(spell, Creature)]
        if creatures:
            # Choose the creature with highest power/toughness sum
            return max(creatures, key=lambda c: c.power + c.toughness)
        
        # If no creatures, return the first castable spell
        return castable_spells[0]

    def select_attackers(self, battlefield):
        """Select creatures to attack with."""
        # Get untapped creatures
        untapped_creatures = [creature for creature in battlefield 
                             if isinstance(creature, Creature) and not creature.tapped]
        if not untapped_creatures:
            return []
        
        # For now, attack with all untapped creatures
        # This is a simple strategy that will help us see if combat is working
        return untapped_creatures

    def select_blockers(self, battlefield, attackers):
        """Select creatures to block with."""
        # Get untapped creatures
        untapped_creatures = [creature for creature in battlefield 
                             if isinstance(creature, Creature) and not creature.tapped]
        if not untapped_creatures or not attackers:
            return {}
        
        # Get state tensor and move to correct device
        state = self._get_state_tensor()
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Use the declare blockers model to select blockers
        return self.declare_blockers_model.select_blockers(state_tensor, untapped_creatures, attackers)
        
    def _get_state_tensor(self) -> np.ndarray:
        """Get a state tensor for the current game state."""
        # Create a state tensor with the correct size for each model
        state = np.zeros(self.state_size)
        
        # Pad or truncate the state to match each model's expected input size
        for model in [
            self.decision_model,
            self.cast_spell_model,
            self.declare_attackers_model,
            self.declare_blockers_model,
            self.tap_lands_model,
            self.select_land_model
        ]:
            if hasattr(model, 'actual_input_size'):
                if len(state) < model.actual_input_size:
                    # Pad with zeros if state is too small
                    state = np.pad(state, (0, model.actual_input_size - len(state)))
                elif len(state) > model.actual_input_size:
                    # Truncate if state is too large
                    state = state[:model.actual_input_size]
                    
        return state 