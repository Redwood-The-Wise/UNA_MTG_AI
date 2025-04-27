import numpy as np
from typing import Tuple, Dict, List, Optional
from game import Game, Phase
from game_state import GameState
from cards import Player, Card, Creature, Land, Color
import random
from model_manager import ModelManager
from cards import Spell
import torch

class MTGTrainingEnv:
    """Environment for training the MTG agent."""
    def __init__(self, agent=None, log_frequency=100):
        """Initialize the training environment."""
        self.game = Game()
        self.agent = agent  # This will be player2_model
        self.last_opponent_life = 20  # Track opponent's life for damage calculation
        self.log_frequency = log_frequency  # Number of games between logging
        self.game_count = 0  # Track number of games played
        
        # Initialize action counters for both players
        self.player1_actions = {
            'next_phase': 0,
            'cast_land': 0,
            'tap_land': 0,
            'cast_spell': 0,
            'declare_attackers': 0,
            'declare_blockers': 0,
            'end_turn': 0
        }
        self.player2_actions = {
            'next_phase': 0,
            'cast_land': 0,
            'tap_land': 0,
            'cast_spell': 0,
            'declare_attackers': 0,
            'declare_blockers': 0,
            'end_turn': 0
        }
        
        # Initialize combat statistics
        self.combat_stats = {
            'player1_attacks': 0,
            'player1_blocks': 0,
            'player2_attacks': 0,
            'player2_blocks': 0,
            'total_damage_dealt': 0,
            'total_damage_prevented': 0
        }
        
        # Initialize turn counts
        self.turn_counts = {
            'player1': 0,
            'player2': 0
        }
        
        # Initialize action validity counters
        self.valid_actions = 0
        self.invalid_actions = 0
        
        # Create players with different decks
        self.current_player = Player("Player 1")
        self.opponent = Player("Player 2")
        
        # Initialize Player 1's deck (Green Aggro)
        for _ in range(12):
            self.current_player.library.append(Land("Forest", [Color.G]))
        for _ in range(4):
            self.current_player.library.append(Creature("Llanowar Elves", "G", [Color.G], 1, 1))
        for _ in range(4):
            self.current_player.library.append(Creature("Grizzly Bears", "1G", [Color.G], 2, 2))
        for _ in range(4):
            self.current_player.library.append(Creature("Rampant Growth", "2G", [Color.G], 3, 3))
        for _ in range(4):
            self.current_player.library.append(Creature("Giant Spider", "3G", [Color.G], 2, 4))
        for _ in range(4):
            self.current_player.library.append(Creature("Craw Wurm", "4G", [Color.G], 6, 4))
        
        # Initialize Player 2's deck (Red Aggro)
        for _ in range(12):
            self.opponent.library.append(Land("Mountain", [Color.R]))
        for _ in range(4):
            self.opponent.library.append(Creature("Goblin Guide", "R", [Color.R], 2, 2))
        for _ in range(4):
            self.opponent.library.append(Creature("Monastery Swiftspear", "R", [Color.R], 1, 2))
        for _ in range(4):
            self.opponent.library.append(Creature("Lightning Bolt", "R", [Color.R], 3, 1))
        for _ in range(4):
            self.opponent.library.append(Creature("Lava Spike", "R", [Color.R], 3, 1))
        for _ in range(4):
            self.opponent.library.append(Creature("Bonecrusher Giant", "2R", [Color.R], 4, 3))
            
        # Shuffle decks
        random.shuffle(self.current_player.library)
        random.shuffle(self.opponent.library)
        
        # Add players to game
        self.game.add_player(self.current_player)
        self.game.add_player(self.opponent)
        
        # Initialize game
        self.game.start_game()
        
        # Create game state
        self.game_state = GameState(self.game)
        self.action_space = ['next_phase', 'cast_land', 'tap_land', 'cast_spell', 'declare_attackers', 'declare_blockers', 'end_turn']
        self.observation_space = (120, 13)  # Updated to match new feature count
        
        # Initialize separate model managers for each player
        self.player1_model_manager = ModelManager(self.observation_space[0] * self.observation_space[1], self.action_space)
        self.player2_model_manager = ModelManager(self.observation_space[0] * self.observation_space[1], self.action_space)
        
        # Load the trained models if an agent is provided
        if agent:
            self.player2_model_manager.load_models(agent)
            self.player2_model_manager.eval()  # Set to evaluation mode
        
        self.device = self.player2_model_manager.device  # Use the same device as model_manager
        
        # Initialize game statistics
        self.game_stats = {
            'player1_wins': 0,
            'player2_wins': 0,
            'total_turns': 0,
            'total_damage': 0,
            'total_blocks': 0
        }
        
    def reset(self):
        """Reset the environment to initial state."""
        self.game = Game()
        self.current_player = Player("Player 1")
        self.opponent = Player("Player 2")
        
        # Reset action counters
        self.player1_actions = {action: 0 for action in self.action_space}
        self.player2_actions = {action: 0 for action in self.action_space}
        
        # Clear any existing cards
        self.current_player.library.clear()
        self.opponent.library.clear()
        
        # Initialize Player 1's deck (Green Aggro)
        for _ in range(12):
            self.current_player.library.append(Land("Forest", [Color.G]))
        for _ in range(4):
            self.current_player.library.append(Creature("Llanowar Elves", "G", [Color.G], 1, 1))
        for _ in range(4):
            self.current_player.library.append(Creature("Grizzly Bears", "1G", [Color.G], 2, 2))
        for _ in range(4):
            self.current_player.library.append(Creature("Rampant Growth", "2G", [Color.G], 3, 3))
        for _ in range(4):
            self.current_player.library.append(Creature("Giant Spider", "3G", [Color.G], 2, 4))
        for _ in range(4):
            self.current_player.library.append(Creature("Craw Wurm", "4G", [Color.G], 6, 4))
        
        # Initialize Player 2's deck (Red Aggro)
        for _ in range(12):
            self.opponent.library.append(Land("Mountain", [Color.R]))
        for _ in range(4):
            self.opponent.library.append(Creature("Goblin Guide", "R", [Color.R], 2, 2))
        for _ in range(4):
            self.opponent.library.append(Creature("Monastery Swiftspear", "R", [Color.R], 1, 2))
        for _ in range(4):
            self.opponent.library.append(Creature("Lightning Bolt", "R", [Color.R], 3, 1))
        for _ in range(4):
            self.opponent.library.append(Creature("Lava Spike", "R", [Color.R], 3, 1))
        for _ in range(4):
            self.opponent.library.append(Creature("Bonecrusher Giant", "2R", [Color.R], 4, 3))
        
        # Shuffle decks
        random.shuffle(self.current_player.library)
        random.shuffle(self.opponent.library)
        
        # Add players to game
        self.game.add_player(self.current_player)
        self.game.add_player(self.opponent)
        
        # Initialize game
        self.game.start_game()
        
        # Create game state
        self.game_state = GameState(self.game)
        state, zone_indices = self.game_state.get_state_tensor()
        return state, zone_indices
        
    def step(self, action=None):
        """Execute one time step within the environment."""
        # Get current state
        state, _ = self.game_state.get_state_tensor()
        
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Determine which player should act
        current_player = self.game.players[self.game.current_player_index]
        opponent = self.game.players[(self.game.current_player_index + 1) % 2]
        
        # If no action provided, use the appropriate model to select one
        if action is None:
            valid_actions = self._get_valid_actions()
            if current_player.name == "Player 1":
                action = self.player1_model_manager.select_action(state_tensor, valid_actions)
            else:
                action = self.player2_model_manager.select_action(state_tensor, valid_actions)
        
        # Update action counter for the current player
        if current_player.name == "Player 1":
            self.player1_actions[action] += 1
        else:
            self.player2_actions[action] += 1

        # Validate action against current phase
        if not self._is_action_valid_for_phase(action, self.game.current_phase):
            return state, -1.0, False, {"error": f"Invalid action {action} for phase {self.game.current_phase}"}

        # Handle the selected action
        if action == 'next_phase':
            if self.game.current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
                if not self.game.combat_state.get('attackers', []):
                    self.game.next_phase()
                    return state, -1.0, False, {}
            elif self.game.current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
                # Get defending player and check if they have untapped creatures
                defending_player = self.game.get_opponent(current_player)
                untapped_creatures = [creature for creature in defending_player.battlefield 
                                    if isinstance(creature, Creature) and not creature.tapped]
                
                if self.game.combat_state.get('attackers', []) and untapped_creatures:
                    if current_player.name == "Player 1":
                        self.combat_stats['player1_blocks'] += 1
                    else:
                        self.combat_stats['player2_blocks'] += 1
                    self._handle_declare_blockers(untapped_creatures)
                    self.game.next_phase()
                elif not untapped_creatures:
                    self.game.next_phase()
                    return state, -1.0, False, {}
                elif untapped_creatures:
                    # Return negative reward for skipping blocking when possible
                    self.game.next_phase()
                    return state, -0.5, False, {}
            elif self.game.current_phase == Phase.COMBAT_DAMAGE:
                # Store current life totals before damage
                current_player = self.game.players[self.game.current_player_index]
                defending_player = self.game.players[(self.game.current_player_index + 1) % 2]
                current_player_life_before = current_player.life
                defending_player_life_before = defending_player.life
                
                # Calculate life changes
                current_player_life_change = current_player.life - current_player_life_before
                defending_player_life_change = defending_player.life - defending_player_life_before
                
                # Update combat stats
                attackers = self.game.combat_state.get('attackers', [])
                blockers = self.game.combat_state.get('blockers', {})
                
                # Track unblocked damage
                for attacker in attackers:
                    if attacker not in blockers.values():
                        self.combat_stats['total_damage_dealt'] += attacker.power
                
                # Track blocked damage
                for blocker, attacker in blockers.items():
                    self.combat_stats['total_damage_dealt'] += attacker.power
                    print(f"Blocked damage: {attacker.name} deals {attacker.power} to {blocker.name}")
                    self.combat_stats['total_damage_dealt'] += blocker.power
                    print(f"Blocked damage: {blocker.name} deals {blocker.power} to {attacker.name}")
                
            elif self.game.current_phase == Phase.COMBAT_END:
                self.game.combat_state = {'attackers': [], 'blockers': {}}
            self.game.next_phase()
        elif action == 'cast_land':
            self._handle_cast_land(self.current_player.hand[0] if self.current_player.hand else None)
        elif action == 'tap_land':
            tap_value = self.player1_model_manager.tap_lands_model(state_tensor).item() if current_player.name == "Player 1" else self.player2_model_manager.tap_lands_model(state_tensor).item()
            untapped_lands = [card for card in current_player.battlefield if isinstance(card, Land) and not card.tapped]
            if untapped_lands:
                selected_land = max(untapped_lands, key=lambda l: tap_value)
                self._handle_tap_lands([selected_land])
        elif action == 'cast_spell':
            self._handle_cast_spell()
        elif action == 'declare_attackers':
            # Only allow declaring attackers in the correct phase
            if self.game.current_phase != Phase.COMBAT_DECLARE_ATTACKERS:
                return state, -1.0, False, {"error": "Cannot declare attackers in this phase"}
            
            # Get current player and opponent
            current_player = self.game.players[self.game.current_player_index]
            defending_player = self.game.players[(self.game.current_player_index + 1) % 2]
            
            # Get untapped creatures
            untapped_creatures = [creature for creature in current_player.battlefield 
                                 if isinstance(creature, Creature) and not creature.tapped]
            
            if not untapped_creatures:
                return state, -1.0, False, {"error": "No untapped creatures available"}
            
            # Get state for model input
            model_input = self._get_state_for_model()
            model_input = torch.FloatTensor(model_input).unsqueeze(0)
            
            # Get attack value from model
            attack_value = self.player1_model_manager.declare_attackers_model(model_input).item() if current_player.name == "Player 1" else self.player2_model_manager.declare_attackers_model(model_input).item()
            
            opponent_has_blockers = any(isinstance(c, Creature) and not c.tapped for c in defending_player.battlefield)
            
            if attack_value > 0.01 or (not opponent_has_blockers and attack_value > -0.1):
                # Use the model to select attackers
                selected_attackers = self.player1_model_manager.handle_declare_attackers(
                    state, untapped_creatures, self.game
                ) if current_player.name == "Player 1" else self.player2_model_manager.handle_declare_attackers(
                    state, untapped_creatures, self.game
                )
                
                if selected_attackers:
                    self.game.declare_attackers(player=current_player, attackers=selected_attackers)
                else:
                    self.game.next_phase()
            else:
                self.game.next_phase()
                return state, -0.5, False, {}
        elif action == 'declare_blockers':
            # Only allow declaring blockers in the correct phase
            if self.game.current_phase != Phase.COMBAT_DECLARE_BLOCKERS:
                return state, -1.0, False, {"error": "Cannot declare blockers in this phase"}
            
            # Get defending player and check if they have untapped creatures
            defending_player = self.game.get_opponent(current_player)
            untapped_creatures = [creature for creature in defending_player.battlefield 
                                if isinstance(creature, Creature) and not creature.tapped]
            
            if not untapped_creatures:
                return state, -1.0, False, {"error": "No untapped creatures available for blocking"}
            
            # Get attackers
            attackers = self.game.combat_state.get('attackers', [])
            if not attackers:
                return state, -1.0, False, {"error": "No attackers to block"}
            
            # Use model to select blockers
            blocking_assignments = self.player1_model_manager.handle_declare_blockers(
                state, untapped_creatures, attackers, self.game
            ) if defending_player.name == "Player 1" else self.player2_model_manager.handle_declare_blockers(
                state, untapped_creatures, attackers, self.game
            )
            
            if blocking_assignments:
                # Calculate potential damage prevented
                potential_damage = sum(attacker.power for attacker in attackers 
                                     if attacker not in blocking_assignments.values())
                
                # Update combat stats
                self.combat_stats['total_damage_prevented'] += potential_damage
                
                # Update block counter for the defending player
                if defending_player.name == "Player 1":
                    self.combat_stats['player1_blocks'] += len(blocking_assignments)
                else:
                    self.combat_stats['player2_blocks'] += len(blocking_assignments)
                
                # Declare blockers with proper assignments
                self.game.declare_blockers(defending_player, blocking_assignments)
                return state, 0.5, False, {}  # Positive reward for blocking
            else:
                self.game.combat_state['blockers'] = {}
                self.game.next_phase()
                return state, -0.2, False, {}  # Small negative reward for not blocking
        elif action == 'end_turn':
            if self.game.current_phase != Phase.CLEANUP:
                return state, -1.0, False, {"error": "Cannot end turn outside of cleanup phase"}
            self.game.combat_state = {'attackers': [], 'blockers': {}}
            self.game.end_turn()
        
        # Update turn counts
        if action == 'end_turn':
            if current_player.name == "Player 1":
                self.turn_counts['player1'] += 1
            else:
                self.turn_counts['player2'] += 1
        
        # Get next state
        next_state, _ = self.game_state.get_state_tensor()
        
        # Check if game is over
        done = self._is_game_over()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        return next_state, reward, done, {}
        
    def _get_valid_actions(self) -> List[str]:
        """Get list of valid actions for current game state."""
        valid_actions = []
        current_phase = self.game.current_phase
        current_player = self.game.get_active_player()
        opponent = self.game.get_opponent(current_player)
        
        # Quick phase-based action checks
        if current_phase == Phase.CLEANUP:
            return ["end_turn"]
        
        if current_phase == Phase.UNTAP:
            # In untap phase, only allow next_phase
            valid_actions.append("next_phase")
            return valid_actions
            
        if current_phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]:
            # Check for land drop - only if we haven't played a land this turn
            if current_player.lands_played_this_turn == 0:
                if any(isinstance(card, Land) for card in current_player.hand):
                    valid_actions.append("cast_land")
            
            # Check for untapped lands that can be tapped for mana
            if any(isinstance(card, Land) and not card.tapped for card in current_player.battlefield):
                valid_actions.append("tap_land")
            
            # Check for spell casting - only if we have enough mana
            if any(not isinstance(card, Land) and current_player.can_pay_mana_cost(card.parse_mana_cost()) 
                  for card in current_player.hand):
                valid_actions.append("cast_spell")
        
        elif current_phase == Phase.COMBAT_BEGINNING:
            # Always allow next_phase in combat beginning
            valid_actions.append("next_phase")
        
        elif current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
            # Always allow next_phase to choose not to attack
            valid_actions.append("next_phase")
            # Allow declare_attackers if we have untapped creatures
            if any(isinstance(card, Creature) and not card.tapped for card in current_player.battlefield):
                valid_actions.append("declare_attackers")
        
        elif current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
            # Get the defending player (the one who is not the active player)
            defending_player = self.game.get_opponent(current_player)
            
            # Always allow next_phase to choose not to block
            valid_actions.append("next_phase")
            
            # Only allow declare_blockers if:
            # 1. There are attackers
            # 2. The defending player has untapped creatures
            # 3. The current player is the defending player
            if (self.game.combat_state.get('attackers', []) and 
                any(isinstance(card, Creature) and not card.tapped 
                    for card in defending_player.battlefield) and
                current_player == defending_player):  # Added check for defending player
                valid_actions.append("declare_blockers")
                print(f"\nBlocking is available for {defending_player.name}!")
                print(f"Attackers: {len(self.game.combat_state.get('attackers', []))}")
                print(f"Available blockers: {len([c for c in defending_player.battlefield if isinstance(c, Creature) and not c.tapped])}")
                print(f"Current player: {current_player.name}")
                print(f"Defending player: {defending_player.name}")
        
        elif current_phase in [Phase.COMBAT_DAMAGE, Phase.COMBAT_END]:
            # Always allow next_phase in damage and end phases
            valid_actions.append("next_phase")
        
        # Always allow next_phase unless in cleanup phase
        if current_phase != Phase.CLEANUP:
            valid_actions.append("next_phase")
        
        # If no valid actions were found, allow next_phase (should never happen)
        if not valid_actions:
            valid_actions.append("next_phase")
        
        # Remove any duplicate actions
        valid_actions = list(dict.fromkeys(valid_actions))
        
        return valid_actions
        
    def _create_action_space(self) -> List[Tuple[str, Dict]]:
        """Create a list of all possible actions."""
        actions = []
        
        # Phase progression
        actions.append(("next_phase", {}))
        
        # Cast spells
        actions.append(("cast_land", {}))
        actions.append(("cast_spell", {}))
        
        # Combat actions
        actions.append(("declare_attackers", {}))
        actions.append(("declare_blockers", {}))
        
        # End turn
        actions.append(("end_turn", {}))
        
        return actions
        
    def _handle_cast_land(self, land: Land) -> None:
        """Handle casting a land card."""
        current_player = self.game.get_active_player()
        current_model_manager = self.player1_model_manager if current_player == self.game.players[0] else self.player2_model_manager
        
        # Get state tensor
        state, _ = self.get_state_tensor()
        
        # Select land to play
        selected_land = current_model_manager.select_land(current_player.hand, self.game)
        if selected_land:
            current_player.cast_land(selected_land)
            
        # Update rewards
        self._calculate_reward('cast_land')
        
    def _update_rewards(self) -> None:
        """Update the game statistics and rewards."""
        current_player = self.game.get_active_player()
        opponent = self.game.get_opponent(current_player)
        
        # Update game statistics
        self.game_stats['total_turns'] += 1
        self.game_stats['total_damage'] += self.last_opponent_life - opponent.life
        
        # Update win counts if game is over
        if self.game.is_game_over():
            if current_player.life > 0:
                self.game_stats['player1_wins' if current_player == self.game.players[0] else 'player2_wins'] += 1
            else:
                self.game_stats['player2_wins' if current_player == self.game.players[0] else 'player1_wins'] += 1
        
    def _handle_tap_lands(self, lands):
        """Handle tapping lands for mana."""
        current_player = self.game.get_active_player()
        for land in lands:
            if land in current_player.battlefield and isinstance(land, Land) and not land.tapped:
                if land.tap():  # Only add mana if the land was successfully tapped
                    # Add mana for each color the land produces
                    for color in land.colors:
                        current_player.add_mana(color)
                
    def _handle_cast_spell(self):
        """Handle casting a spell."""
        current_player = self.game.get_active_player()
        spell = current_player.hand[0] if current_player.hand else None
        if spell and (isinstance(spell, Spell) or isinstance(spell, Creature)):
            if current_player.can_pay_mana_cost(spell.parse_mana_cost()):
                current_player.cast_spell(spell)
                
    def _handle_declare_blockers(self, blockers):
        """Handle declaring blockers."""
        defending_player = self.game.get_opponent(self.game.get_active_player())
        valid_blockers = [creature for creature in defending_player.battlefield 
                         if isinstance(creature, Creature) and not creature.tapped]
        attackers = self.game.combat_state.get('attackers', [])
        
        if not valid_blockers or not attackers:
            if self.game_count % self.log_frequency == 0:
                print("No valid blockers or attackers available")
            return
        
        # Get state tensor for the model
        state, _ = self.game_state.get_state_tensor()
        
        # Use the appropriate model based on which player is defending
        if defending_player.name == "Player 1":
            blocking_assignments = self.player1_model_manager.handle_declare_blockers(
                state, valid_blockers, attackers, self.game
            )
        else:
            blocking_assignments = self.player2_model_manager.handle_declare_blockers(
                state, valid_blockers, attackers, self.game
            )
        
        if blocking_assignments:
            # Calculate potential damage prevented
            potential_damage = sum(attacker.power for attacker in attackers 
                                 if attacker not in blocking_assignments.values())
            
            # Update combat stats
            self.combat_stats['total_damage_prevented'] += potential_damage
            
            # Update block counter for the defending player
            if defending_player.name == "Player 1":
                self.combat_stats['player1_blocks'] += len(blocking_assignments)
            else:
                self.combat_stats['player2_blocks'] += len(blocking_assignments)
            
            # Declare blockers with proper assignments
            self.game.declare_blockers(defending_player, blocking_assignments)
            
        else:
            if self.game_count % self.log_frequency == 0:
                print(f"No blocking assignments made by {defending_player.name}'s model")
        
    def _handle_end_turn(self):
        """Handle ending the turn."""
        if self.game.current_phase != Phase.CLEANUP:
            raise ValueError("Must complete all phases before ending turn")
            
        self.game.end_turn()
        
    def _calculate_reward(self, action):
        """Calculate reward for the current action."""
        current_player = self.game.players[self.game.current_player_index]
        opponent = self.game.players[(self.game.current_player_index + 1) % 2]
        
        # Base reward for taking any action
        reward = 0.1
        
        # Life change reward (increased)
        life_change = self.last_opponent_life - opponent.life
        if life_change > 0:
            reward += life_change * 3.0
        elif life_change < 0:
            reward += life_change * 3.5
            
        # Action-specific rewards (increased)
        if action == 'cast_land':
            reward += 1.0
            reward += len([c for c in current_player.battlefield if isinstance(c, Land)]) * 0.2
        elif action == 'cast_spell':
            reward += 2.0
            if any(isinstance(c, Creature) for c in current_player.hand):
                reward += 1.0
        elif action == 'declare_attackers':
            attackers = self.game.combat_state.get('attackers', [])
            reward += len(attackers) * 2.0  # Increased from 1.0
            if len(attackers) > 1:
                reward += 1.0  # Increased from 0.5
            if not any(isinstance(c, Creature) and not c.tapped for c in opponent.battlefield):
                reward += 2.0  # Increased from 1.0
            # Add reward for potential damage
            potential_damage = sum(attacker.power for attacker in attackers)
            reward += potential_damage * 1.5  # New reward for potential damage
        elif action == 'declare_blockers':
            # Get blockers from combat_state
            blockers = self.game.combat_state.get('blockers', {})
            reward += len(blockers) * 5.0  # Increased from 3.0
            
            # Calculate potential damage prevented
            potential_damage = 0
            for attacker in self.game.combat_state.get('attackers', []):
                if attacker not in blockers.values():  # If attacker isn't blocked
                    potential_damage += attacker.power
            
            # Reward for preventing damage
            reward += potential_damage * 6.0  # Increased from 4.0
            
            # Additional reward for effective blocking
            for blocker, attacker in blockers.items():
                if blocker.toughness > attacker.power:
                    reward += 6.0  # Increased from 4.0 for surviving blocks
                elif blocker.toughness == attacker.power:
                    reward += 5.0  # Increased from 3.0 for trading evenly
                elif blocker.power >= attacker.toughness:
                    reward += 4.0  # Increased from 2.0 for killing the attacker even if blocker dies
                
                # Bonus for preventing lethal damage
                if current_player.life <= attacker.power:
                    reward += 10.0  # Large bonus for preventing lethal damage
        
        # Game end rewards (increased)
        if self.game.is_game_over():
            if current_player.life <= 0:
                reward = -200
            elif opponent.life <= 0:
                reward = 200
            elif len(current_player.library) == 0:
                reward = -100
            elif len(opponent.library) == 0:
                reward = 100
            elif self.game.turn_number >= 50:
                reward = -50
                
        return reward
        
    def _is_game_over(self) -> bool:
        """Check if the game is over."""
        return (self.current_player.life <= 0 or 
                self.opponent.life <= 0 or 
                len(self.current_player.library) == 0 or 
                len(self.opponent.library) == 0)
        
    def get_state_tensor(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current game state as tensor and zone indices."""
        # Pre-allocate the state tensor with zeros
        state = np.zeros((120, 13), dtype=np.float32)  # Added phase information
        zone_indices = np.zeros(120, dtype=np.int64)
        
        # Get all cards from both players
        all_cards = []
        zone_map = []  # Track which zone each card belongs to
        
        # Add current player's cards with zone tracking
        for card in self.current_player.library:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['current_library'])
        
        for card in self.current_player.hand:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['current_hand'])
        
        for card in self.current_player.battlefield:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['current_battlefield'])
        
        for card in self.current_player.graveyard:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['current_graveyard'])
        
        for card in self.current_player.exile:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['current_exile'])
        
        # Add opponent's cards
        for card in self.opponent.library:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['opponent_library'])
        
        for card in self.opponent.hand:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['opponent_hand'])
        
        for card in self.opponent.battlefield:
            all_cards.append(card)
            zone_map.append(GameState.ZONES['opponent_battlefield'])
        
        # Process all cards
        num_cards = min(len(all_cards), 120)
        for i in range(num_cards):
            card, zone = all_cards[i], zone_map[i]
            # Convert card to feature vector
            features = [
                float(isinstance(card, Creature)),  # is_creature
                float(isinstance(card, Land)),      # is_land
                float(card.power) if isinstance(card, Creature) else 0.0,  # power
                float(card.toughness) if isinstance(card, Creature) else 0.0,  # toughness
                float(card.tapped),                 # is_tapped
                float(card.controller == self.current_player),  # is_controlled_by_current_player
                float(card in self.current_player.hand),  # is_in_hand
                float(card in self.current_player.graveyard),  # is_in_graveyard
                float(card.controller == self.opponent),  # is_opponent_controlled
                float(card in self.opponent.hand or card in self.opponent.graveyard),  # is_opponent_card
                float(card in self.current_player.battlefield),  # is_on_battlefield
                float(card in self.current_player.exile),  # is_exiled
                float(list(Phase.__members__.values()).index(self.game.current_phase)) / len(Phase.__members__)  # Normalized phase information
            ]
            state[i] = features
            zone_indices[i] = zone
        
        # Verify tensor shapes
        assert state.shape == (120, 13), f"Expected state shape (120, 13), got {state.shape}"
        assert zone_indices.shape == (120,), f"Expected zone_indices shape (120,), got {zone_indices.shape}"
        
        return state, zone_indices

    def print_game_stats(self):
        """Print current game statistics."""
        print("\n" + "="*50)
        print("Game Statistics:")
        print(f"Current turn: {self.game.turn_number}")
        print(f"Current phase: {self.game.current_phase}")
        print(f"Current player: {self.game.players[self.game.current_player_index].name}")
        
        # Print player stats
        for player in self.game.players:
            print(f"\n{player.name} Stats:")
            print(f"  Life: {player.life}")
            print(f"  Hand size: {len(player.hand)}")
            print(f"  Battlefield size: {len(player.battlefield)}")
            print(f"  Library size: {len(player.library)}")
            print(f"  Graveyard size: {len(player.graveyard)}")
            print(f"  Lands played this turn: {player.lands_played_this_turn}")
            print(f"  Mana pool: {player.mana_pool}")
            
            # Print creature stats
            creatures = [c for c in player.battlefield if isinstance(c, Creature)]
            print(f"  Creatures on battlefield: {len(creatures)}")
            for creature in creatures:
                print(f"    - {creature.name} (P/T: {creature.power}/{creature.toughness}, Tapped: {creature.tapped})")
        
        # Print action statistics
        print("\nPlayer 1 Actions:")
        for action, count in self.player1_actions.items():
            print(f"  {action}: {count}")
            
        print("\nPlayer 2 Actions:")
        for action, count in self.player2_actions.items():
            print(f"  {action}: {count}")
        
        # Print combat statistics
        print("\nCombat Statistics:")
        print(f"  Player 1 Attacks: {self.combat_stats['player1_attacks']}")
        print(f"  Player 1 Blocks: {self.combat_stats['player1_blocks']}")
        print(f"  Player 2 Attacks: {self.combat_stats['player2_attacks']}")
        print(f"  Player 2 Blocks: {self.combat_stats['player2_blocks']}")
        print(f"  Total Damage Dealt: {self.combat_stats['total_damage_dealt']}")
        print(f"  Total Damage Prevented: {self.combat_stats['total_damage_prevented']}")
        
        # Print turn and action validity stats
        print("\nGame Progress:")
        print(f"  Player 1 Turns: {self.turn_counts['player1']}")
        print(f"  Player 2 Turns: {self.turn_counts['player2']}")
        print(f"  Valid Actions: {self.valid_actions}")
        print(f"  Invalid Actions: {self.invalid_actions}")
        print("="*50 + "\n")

    def _is_action_valid_for_phase(self, action: str, phase: Phase) -> bool:
        """Check if an action is valid for the current phase."""
        if action == 'next_phase':
            return True  # Always valid
        elif action == 'cast_land':
            return phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]
        elif action == 'cast_spell':
            return phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]
        elif action == 'declare_attackers':
            return phase == Phase.COMBAT_DECLARE_ATTACKERS
        elif action == 'declare_blockers':
            return phase == Phase.COMBAT_DECLARE_BLOCKERS
        elif action == 'end_turn':
            return phase == Phase.CLEANUP
        return False

    def _get_state_for_model(self) -> np.ndarray:
        """Get the current game state formatted for model input."""
        # Get current state
        state, _ = self.game_state.get_state_tensor()
        
        # Get current player and opponent
        current_player = self.game.players[self.game.current_player_index]
        opponent = self.game.players[(self.game.current_player_index + 1) % 2]
        
        # Get untapped creatures for current player
        untapped_creatures = [creature for creature in current_player.battlefield 
                             if isinstance(creature, Creature) and not creature.tapped]
        
        # Get potential blockers from opponent
        potential_blockers = [creature for creature in opponent.battlefield 
                             if isinstance(creature, Creature) and not creature.tapped]
        
        # Calculate average blocker stats
        avg_blocker_power = np.mean([b.power for b in potential_blockers]) if potential_blockers else 0
        avg_blocker_toughness = np.mean([b.toughness for b in potential_blockers]) if potential_blockers else 0
        
        # Calculate potential damage if unblocked
        potential_damage = sum(c.power for c in untapped_creatures)
        
        # Prepare input for the model by adding creature features
        creature_features = np.array([
            0,  # power
            0,  # toughness
            0,  # mana_cost_len
            0,  # mana_cost_sum
            len(untapped_creatures),  # available_attackers
            len(potential_blockers),  # potential_blockers
            avg_blocker_power,  # avg_blocker_power
            avg_blocker_toughness,  # avg_blocker_toughness
            0,  # can_kill_avg
            0,  # can_survive_avg
            0,  # can_trade_avg
            0   # can_deal_damage
        ])
        
        # Concatenate state with creature features
        model_input = np.concatenate([state.reshape(-1), creature_features])
        
        return model_input