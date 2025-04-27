from typing import List, Dict, Tuple, Optional
import numpy as np
from cards import Card, Player, Color, CardType, Creature, Land, Spell
from game import Game, Phase

class GameState:
    """Represents the current state of the game in a format suitable for transformer input."""
    
    # Define zone indices
    ZONES = {
        'current_library': 0,
        'current_hand': 1,
        'current_battlefield': 2,
        'current_graveyard': 3,
        'current_exile': 4,
        'opponent_library': 5,
        'opponent_hand': 6,
        'opponent_battlefield': 7,
    }
    
    def __init__(self, game: Game):
        self.game = game
        self.state_size = 120  # Number of cards in each zone
        self.feature_size = 20  # Increased from 13 to include more features
        
        # Validate game state
        if not hasattr(game, 'players') or not game.players:
            raise ValueError("Game must have players")
            
        if not hasattr(game, 'current_player_index'):
            raise ValueError("Game must have current_player_index")
            
        if game.current_player_index >= len(game.players):
            raise ValueError("Invalid current player index")
            
        # Initialize players
        self.current_player = game.players[game.current_player_index]
        if self.current_player is None:
            raise ValueError("Current player cannot be None")
            
        self.opponent = game.players[(game.current_player_index + 1) % 2]
        if self.opponent is None:
            raise ValueError("Opponent cannot be None")
            
        # Initialize statistics counters
        self.stats = {
            'valid_actions': 0,
            'invalid_actions': 0,
            'action_counts': {
                'Player 1': {
                    'next_phase': 0,
                    'cast_land': 0,
                    'tap_land': 0,
                    'cast_spell': 0,
                    'declare_attackers': 0,
                    'declare_blockers': 0,
                    'end_turn': 0
                },
                'Player 2': {
                    'next_phase': 0,
                    'cast_land': 0,
                    'tap_land': 0,
                    'cast_spell': 0,
                    'declare_attackers': 0,
                    'declare_blockers': 0,
                    'end_turn': 0
                }
            },
            'turn_counts': {
                'Player 1': 0,
                'Player 2': 0
            },
            'current_turn': 0
        }
        
    def get_state_tensor(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get the current game state as a tensor."""
        # Initialize state tensor
        state = np.zeros((self.state_size, self.feature_size))
        zone_indices = {}
        
        # Get current player and opponent
        current_player = self.game.get_active_player()
        opponent = self.game.get_opponent(current_player)
        
        # Add player-specific features
        player_features = np.array([
            current_player.life,
            len(current_player.hand),
            len(current_player.battlefield),
            len(current_player.graveyard),
            len(current_player.exile),
            current_player.lands_played_this_turn,
            sum(1 for c in current_player.battlefield if isinstance(c, Creature)),
            sum(1 for c in current_player.battlefield if isinstance(c, Land)),
            sum(c.power for c in current_player.battlefield if isinstance(c, Creature)),
            sum(c.toughness for c in current_player.battlefield if isinstance(c, Creature)),
            sum(1 for c in current_player.battlefield if isinstance(c, Creature) and not c.tapped),
            sum(1 for c in current_player.battlefield if isinstance(c, Land) and not c.tapped),
            opponent.life,
            len(opponent.hand),
            len(opponent.battlefield),
            len(opponent.graveyard),
            len(opponent.exile),
            sum(1 for c in opponent.battlefield if isinstance(c, Creature)),
            sum(1 for c in opponent.battlefield if isinstance(c, Land)),
            sum(c.power for c in opponent.battlefield if isinstance(c, Creature))
        ])
        
        # Add phase information
        phase_features = np.zeros(20)
        phase_idx = list(Phase).index(self.game.current_phase)
        phase_features[phase_idx] = 1
        
        # Add combat state information
        combat_features = np.zeros(20)
        if self.game.combat_state:
            combat_features[0] = len(self.game.combat_state.get('attackers', []))
            combat_features[1] = len(self.game.combat_state.get('blockers', {}))
            combat_features[2] = sum(c.power for c in self.game.combat_state.get('attackers', []))
            combat_features[3] = sum(c.toughness for c in self.game.combat_state.get('attackers', []))
        
        # Combine all features
        state[0] = player_features
        state[1] = phase_features
        state[2] = combat_features
        
        # Add card-specific features
        for i, card in enumerate(current_player.hand):
            if i >= self.state_size - 3:
                break
            state[i + 3] = self._get_card_features(card)
            zone_indices[f'hand_{i}'] = i + 3
            
        for i, card in enumerate(current_player.battlefield):
            if i >= self.state_size - 3:
                break
            state[i + 3 + len(current_player.hand)] = self._get_card_features(card)
            zone_indices[f'battlefield_{i}'] = i + 3 + len(current_player.hand)
            
        for i, card in enumerate(opponent.battlefield):
            if i >= self.state_size - 3:
                break
            state[i + 3 + len(current_player.hand) + len(current_player.battlefield)] = self._get_card_features(card)
            zone_indices[f'opponent_battlefield_{i}'] = i + 3 + len(current_player.hand) + len(current_player.battlefield)
        
        return state, zone_indices
    
    def _get_card_features(self, card: Card) -> np.ndarray:
        """Get features for a specific card."""
        features = np.zeros(self.feature_size)
        
        if isinstance(card, Creature):
            features[0] = card.power
            features[1] = card.toughness
            features[2] = 1  # Is creature
            features[3] = 1 if not card.tapped else 0
        elif isinstance(card, Land):
            features[5] = 1  # Is land
            features[6] = 1 if not card.tapped else 0
            features[7] = len(card.colors)
            for i, color in enumerate(card.colors):
                features[8 + i] = 1
        else:
            features[12] = 1  # Is spell
            
        # Add mana cost features
        mana_cost = card.parse_mana_cost()
        features[13] = sum(mana_cost.values())
        features[14] = len(mana_cost)
        for i, (color, amount) in enumerate(mana_cost.items()):
            if i < 5:  # Only store first 5 colors
                features[15 + i] = amount
                
        return features
    
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions for current game state."""
        valid_actions = []
        current_phase = self.game.current_phase
        
        # Quick phase-based action checks
        if current_phase == Phase.CLEANUP:
            return ["end_turn"]
        
        if current_phase == Phase.UNTAP:
            # In untap phase, only allow next_phase
            valid_actions.append("next_phase")
            return valid_actions
            
        if current_phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]:
            # Check for land drop - only if we haven't played a land this turn
            if self.current_player.lands_played_this_turn == 0:
                if any(isinstance(card, Land) for card in self.current_player.hand):
                    valid_actions.append("cast_land")
            
            # Check for untapped lands that can be tapped for mana
            if any(isinstance(card, Land) and not card.tapped for card in self.current_player.battlefield):
                valid_actions.append("tap_land")
            
            # Check for spell casting - only if we have enough mana
            if any(not isinstance(card, Land) and self.current_player.can_pay_mana_cost(card.parse_mana_cost()) 
                  for card in self.current_player.hand):
                valid_actions.append("cast_spell")
            
        elif current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
            if any(isinstance(card, Creature) and not card.tapped for card in self.current_player.battlefield):
                valid_actions.append("declare_attackers")
            
        elif current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
            if any(isinstance(card, Creature) and not card.tapped for card in self.opponent.battlefield) and self.game.combat_state['attackers']:
                valid_actions.append("declare_blockers")
        
        # Always allow next_phase unless in cleanup phase
        if current_phase != Phase.CLEANUP:
            valid_actions.append("next_phase")
        
        # If no valid actions were found, allow next_phase (should never happen)
        if not valid_actions:
            valid_actions.append("next_phase")
        
        return valid_actions
    
    def apply_action(self, action):
        """Apply an action to the game state."""
        try:
            # Get current player
            current_player = self.game.players[self.game.current_player_index]
            
            # Check if action is valid for current phase
            if not self._is_action_valid(action):
                self.stats['invalid_actions'] += 1
                print(f"Invalid action: {action} for phase {self.game.current_phase}")
                return False
            
            # Extract action type and parameters
            if isinstance(action, tuple):
                action_type, params = action
            else:
                action_type = action
                params = {}
                
            # Increment action count for the current player
            player_name = current_player.name
            self.stats['action_counts'][player_name][action_type] += 1
            self.stats['valid_actions'] += 1
            
            # Handle the action
            if action_type == "next_phase":
                self._move_to_next_phase()
            elif action_type == "cast_land":
                self._handle_cast_land()
            elif action_type == "tap_land":
                self._handle_tap_land()
            elif action_type == "cast_spell":
                self._handle_cast_spell()
            elif action_type == "declare_attackers":
                self._handle_declare_attackers()
            elif action_type == "declare_blockers":
                self._handle_declare_blockers()
            elif action_type == "end_turn":
                self._handle_end_turn()
                
            return True
            
        except Exception as e:
            print(f"Error applying action {action}: {str(e)}")
            return False
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return any(player.life <= 0 for player in self.game.players)
    
    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game, if any."""
        if not self.is_game_over():
            return None
            
        for player in self.game.players:
            if player.life > 0:
                return player
                
        return None  # Draw game

    def _move_to_next_phase(self):
        """Move to the next phase in the game."""
        try:
            # Get current phase
            current_phase = self.game.current_phase
            
            # Handle cleanup phase separately
            if current_phase == Phase.CLEANUP:
                # Handle cleanup phase
                self._handle_cleanup_phase()
                
                # Switch to next player's turn
                self.game.current_player_index = (self.game.current_player_index + 1) % len(self.game.players)
                self.game.turn_number += 1
                
                # Reset to untap phase
                self.game.current_phase = Phase.UNTAP
                return True
            
            # For all other phases, use the game's next_phase method
            self.game.next_phase()
            
            # If we just moved to cleanup phase, handle it
            if self.game.current_phase == Phase.CLEANUP:
                self._handle_cleanup_phase()
            
            return True
            
        except Exception as e:
            print(f"Error in _move_to_next_phase: {str(e)}")
            # If there's an error, force advance to next phase
            phases = list(Phase.__members__.values())
            current_index = phases.index(self.game.current_phase)
            next_index = (current_index + 1) % len(phases)
            self.game.current_phase = phases[next_index]
            return True

    def _handle_cleanup_phase(self):
        """Handle cleanup phase actions."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Empty mana pools
            for player in self.game.players:
                player.mana_pool = {}
            
            # Reset "until end of turn" effects
            for player in self.game.players:
                for permanent in player.battlefield:
                    if hasattr(permanent, 'until_end_of_turn_effects'):
                        permanent.until_end_of_turn_effects = {}
            
            # Reset combat state
            self.game.combat_state = {
                'attackers': [],
                'blockers': {},
                'damage_assignment': {}
            }
            
            # Reset player-specific turn state
            current_player.lands_played_this_turn = 0
            current_player.mana_pool = {}
            
            # Handle cleanup step
            if not self.game.cleanup_step():
                print("Warning: cleanup_step failed")
            
            return True
            
        except Exception as e:
            print(f"Error in _handle_cleanup_phase: {str(e)}")
            return True  # Always return True to ensure game can progress

    def _handle_end_turn(self):
        """Handle end of turn actions and transition to next player's turn."""
        try:
            # Update turn statistics
            current_player = self.game.players[self.game.current_player_index]
            self.stats['turn_counts'][current_player.name] += 1
            self.stats['current_turn'] += 1
            
            # Switch to next player's turn
            self.game.current_player_index = (self.game.current_player_index + 1) % len(self.game.players)
            self.game.turn_number += 1
            
            # Reset to untap phase
            self.game.current_phase = Phase.UNTAP
            
            # Reset player-specific turn state
            current_player = self.game.players[self.game.current_player_index]
            current_player.lands_played_this_turn = 0
            current_player.mana_pool = {}
            
            return True
            
        except Exception as e:
            print(f"Error in _handle_end_turn: {str(e)}")
            return False

    def _is_action_valid(self, action):
        """Check if an action is valid for the current game state."""
        try:
            # Extract action type and parameters
            if isinstance(action, tuple):
                action_type, params = action
            else:
                action_type = action
                params = {}
            
            # Get current phase
            current_phase = self.game.current_phase
            current_player = self.game.players[self.game.current_player_index]
            
            # Phase-specific validation
            if current_phase == Phase.CLEANUP:
                return action_type == "end_turn"
            
            if current_phase == Phase.UNTAP:
                return action_type == "next_phase"
            
            if current_phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]:
                if action_type == "cast_land":
                    return (current_player.lands_played_this_turn == 0 and 
                           any(isinstance(card, Land) for card in current_player.hand))
                elif action_type == "tap_land":
                    return any(isinstance(card, Land) and not card.tapped 
                             for card in current_player.battlefield)
                elif action_type == "cast_spell":
                    return any(not isinstance(card, Land) and 
                             current_player.can_pay_mana_cost(card.parse_mana_cost()) 
                             for card in current_player.hand)
            
            if current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
                if action_type == "declare_attackers":
                    return any(isinstance(card, Creature) and not card.tapped 
                             for card in current_player.battlefield)
            
            if current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
                if action_type == "declare_blockers":
                    return (any(isinstance(card, Creature) and not card.tapped 
                              for card in self.opponent.battlefield) and 
                            self.game.combat_state['attackers'])
            
            # next_phase is always valid except in cleanup phase
            if action_type == "next_phase":
                return current_phase != Phase.CLEANUP
            
            return False
            
        except Exception as e:
            print(f"Error in _is_action_valid: {str(e)}")
            return False 

    def print_stats(self):
        """Print current game state statistics."""
        print("\nGame Statistics:")
        print(f"Total valid actions: {self.stats['valid_actions']}")
        print(f"Total invalid actions: {self.stats['invalid_actions']}")
        print(f"Turn counts - Player 1: {self.stats['turn_counts']['Player 1']}, Player 2: {self.stats['turn_counts']['Player 2']}")
        print(f"Current turn: {self.stats['current_turn']}")

    def _handle_cast_land(self):
        """Handle casting a land card."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Find a land in hand
            land = next((card for card in current_player.hand if isinstance(card, Land)), None)
            if not land:
                return False
                
            # Move land from hand to battlefield
            current_player.hand.remove(land)
            current_player.battlefield.append(land)
            current_player.lands_played_this_turn += 1
            
            return True
            
        except Exception as e:
            print(f"Error in _handle_cast_land: {str(e)}")
            return False
            
    def _handle_tap_land(self):
        """Handle tapping a land for mana."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Find an untapped land
            land = next((card for card in current_player.battlefield 
                        if isinstance(card, Land) and not card.tapped), None)
            if not land:
                return False
                
            # Tap the land and add mana to pool
            land.tapped = True
            for color in land.colors:
                current_player.mana_pool[color] = current_player.mana_pool.get(color, 0) + 1
                
            return True
            
        except Exception as e:
            print(f"Error in _handle_tap_land: {str(e)}")
            return False
            
    def _handle_cast_spell(self):
        """Handle casting a spell."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Find a castable spell
            spell = next((card for card in current_player.hand 
                         if not isinstance(card, Land) and 
                         current_player.can_pay_mana_cost(card.parse_mana_cost())), None)
            if not spell:
                return False
                
            # Pay mana cost
            mana_cost = spell.parse_mana_cost()
            for color, amount in mana_cost.items():
                current_player.mana_pool[color] -= amount
                
            # Move spell from hand to battlefield
            current_player.hand.remove(spell)
            current_player.battlefield.append(spell)
            
            return True
            
        except Exception as e:
            print(f"Error in _handle_cast_spell: {str(e)}")
            return False
            
    def _handle_declare_attackers(self):
        """Handle declaring attackers."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Find untapped creatures
            attackers = [card for card in current_player.battlefield 
                        if isinstance(card, Creature) and not card.tapped]
            if not attackers:
                return False
                
            # Declare attackers
            self.game.combat_state['attackers'] = attackers
            for attacker in attackers:
                attacker.tapped = True
                
            return True
            
        except Exception as e:
            print(f"Error in _handle_declare_attackers: {str(e)}")
            return False
            
    def _handle_declare_blockers(self):
        """Handle declaring blockers."""
        try:
            current_player = self.game.players[self.game.current_player_index]
            
            # Find untapped creatures
            blockers = [card for card in current_player.battlefield 
                       if isinstance(card, Creature) and not card.tapped]
            if not blockers or not self.game.combat_state['attackers']:
                return False
                
            # Create blocking assignments
            blocking_assignments = {}
            for blocker in blockers:
                if self.game.combat_state['attackers']:
                    attacker = self.game.combat_state['attackers'][0]
                    blocking_assignments[blocker] = attacker
                    
            # Declare blockers
            self.game.combat_state['blockers'] = blocking_assignments
            for blocker in blockers:
                blocker.tapped = True
                
            return True
            
        except Exception as e:
            print(f"Error in _handle_declare_blockers: {str(e)}")
            return False 