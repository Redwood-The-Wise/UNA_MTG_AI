from typing import List, Dict, Optional, Tuple
from enum import Enum
from cards import Card, Player, Color, CardType, Creature, Land, Spell
from card_database import CardDatabase
from effects import Effect, EffectType
import numpy as np

class Phase(Enum):
    """Magic: The Gathering game phases"""
    UNTAP = "untap"
    UPKEEP = "upkeep"
    DRAW = "draw"
    MAIN_PHASE_1 = "main_phase_1"
    COMBAT_BEGINNING = "combat_beginning"
    COMBAT_DECLARE_ATTACKERS = "combat_declare_attackers"
    COMBAT_DECLARE_BLOCKERS = "combat_declare_blockers"
    COMBAT_DAMAGE = "combat_damage"
    COMBAT_END = "combat_end"
    MAIN_PHASE_2 = "main_phase_2"
    END_STEP = "end_step"
    CLEANUP = "cleanup"
    
    def __lt__(self, other):
        """Ensure proper phase ordering."""
        phases = [
            Phase.UNTAP,
            Phase.UPKEEP,
            Phase.DRAW,
            Phase.MAIN_PHASE_1,
            Phase.COMBAT_BEGINNING,
            Phase.COMBAT_DECLARE_ATTACKERS,
            Phase.COMBAT_DECLARE_BLOCKERS,
            Phase.COMBAT_DAMAGE,
            Phase.COMBAT_END,
            Phase.MAIN_PHASE_2,
            Phase.END_STEP,
            Phase.CLEANUP
        ]
        return phases.index(self) < phases.index(other)

class Game:
    def __init__(self):
        self.card_database = CardDatabase()
        self.players: List[Player] = []
        self.current_player_index: int = 0
        self.current_phase: Phase = Phase.UNTAP
        self.turn_number: int = 1
        self.stack: List[Effect] = []
        self.combat_state: Dict = {
            'attackers': [],
            'blockers': {},
            'damage_assigned': False
        }
        
    def add_player(self, player: Player):
        """Add a player to the game."""
        self.players.append(player)
        
    def start_game(self):
        """Initialize the game and draw starting hands."""
        if len(self.players) != 2:
            raise ValueError("Game requires exactly 2 players")
            
        # Each player draws 7 cards
        for player in self.players:
            for _ in range(7):
                player.draw_card()
                
    def next_phase(self):
        """Advance to the next phase."""
        try:
            # Validate game state
            if not hasattr(self, 'players') or not self.players:
                print("Warning: No players in game")
                return False
                
            if not hasattr(self, 'current_player_index'):
                print("Warning: No current player index")
                return False
                
            if self.current_player_index >= len(self.players):
                print("Warning: Invalid current player index")
                return False
                
            current_player = self.players[self.current_player_index]
            if current_player is None:
                print("Warning: Current player is None")
                return False
                
            # Get defending player for combat phases
            defending_player = self.players[(self.current_player_index + 1) % 2]
            if defending_player is None:
                print("Warning: Defending player is None")
                return False
                
            # Check battlefield based on phase
            if self.current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
                # During declare blockers, check defending player's battlefield
                if not hasattr(defending_player, 'battlefield'):
                    print(f"Warning: Player {defending_player.name} has no battlefield")
                    return False
                    
                if defending_player.battlefield is None:
                    print(f"Warning: Player {defending_player.name}'s battlefield is None")
                    return False
            else:
                # For other phases, check current player's battlefield
                if not hasattr(current_player, 'battlefield'):
                    print(f"Warning: Player {current_player.name} has no battlefield")
                    return False
                    
                if current_player.battlefield is None:
                    print(f"Warning: Player {current_player.name}'s battlefield is None")
                    return False
            
            phases = [
                Phase.UNTAP,
                Phase.UPKEEP,
                Phase.DRAW,
                Phase.MAIN_PHASE_1,
                Phase.COMBAT_BEGINNING,
                Phase.COMBAT_DECLARE_ATTACKERS,
                Phase.COMBAT_DECLARE_BLOCKERS,
                Phase.COMBAT_DAMAGE,
                Phase.COMBAT_END,
                Phase.MAIN_PHASE_2,
                Phase.END_STEP,
                Phase.CLEANUP
            ]
            
            current_index = phases.index(self.current_phase)
            next_index = (current_index + 1) % len(phases)
            
            # Handle current phase before moving to next
            if self.current_phase == Phase.UNTAP:
                if not self._handle_untap_phase():
                    print("Warning: Failed to handle untap phase")
            elif self.current_phase == Phase.UPKEEP:
                # Handle upkeep triggers here if needed
                pass
            elif self.current_phase == Phase.DRAW:
                if not self._handle_draw_phase():
                    print("Warning: Failed to handle draw phase")
            elif self.current_phase == Phase.CLEANUP:
                if not self._handle_cleanup_phase():
                    print("Warning: Failed to handle cleanup phase")
                # After cleanup, it's the next player's turn
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                self.turn_number += 1
                # Reset to untap phase for the new turn
                self.current_phase = Phase.UNTAP
                return True
            
            # Empty mana pool at end of phase
            if hasattr(current_player, 'empty_mana_pool'):
                current_player.empty_mana_pool()
            
            # Move to next phase
            self.current_phase = phases[next_index]
            
            # Reset combat state when entering cleanup phase
            if self.current_phase == Phase.CLEANUP:
                self.combat_state = {
                    'attackers': [],
                    'blockers': {},
                    'damage_assigned': False
                }
            # Assign damage when entering damage phase
            elif self.current_phase == Phase.COMBAT_DAMAGE:
                self.assign_damage()
                
            return True
                
        except Exception as e:
            print(f"Error in next_phase: {str(e)}")
            # If there's an error, try to recover by moving to the next phase
            try:
                phases = list(Phase.__members__.values())
                current_index = phases.index(self.current_phase)
                next_index = (current_index + 1) % len(phases)
                self.current_phase = phases[next_index]
                return True
            except Exception as e2:
                print(f"Error in next_phase recovery: {str(e2)}")
                return False
        
    def _handle_untap_phase(self):
        """Handle the untap phase."""
        try:
            # Validate game state
            if not hasattr(self, 'players') or not self.players:
                print("Warning: No players in game")
                return False
                
            if not hasattr(self, 'current_player_index'):
                print("Warning: No current player index")
                return False
                
            if self.current_player_index >= len(self.players):
                print("Warning: Invalid current player index")
                return False
                
            current_player = self.players[self.current_player_index]
            if current_player is None:
                print("Warning: Current player is None")
                return False
                
            if not hasattr(current_player, 'battlefield'):
                print(f"Warning: Player {current_player.name} has no battlefield")
                return False
                
            if current_player.battlefield is None:
                print(f"Warning: Player {current_player.name}'s battlefield is None")
                return False
            
            # Untap all permanents
            for card in current_player.battlefield:
                card.tapped = False
                # Reset damage only for creatures
                if isinstance(card, Creature):
                    card.damage = 0
                    
            # Reset lands played this turn
            current_player.lands_played_this_turn = 0
            
            # Empty mana pool
            current_player.mana_pool = {}
            
            return True
            
        except Exception as e:
            print(f"Error in _handle_untap_phase: {str(e)}")
            return False
        
    def _handle_draw_phase(self):
        """Handle the draw phase."""
        try:
            current_player = self.players[self.current_player_index]
            current_player.draw_card()
            return True
        except Exception as e:
            return False
        
    def _handle_cleanup_phase(self):
        """Handle the cleanup phase."""
        try:
            # Ensure we have a valid current player
            if not self.players or self.current_player_index >= len(self.players):
                raise ValueError("Invalid current player index")
            
            current_player = self.players[self.current_player_index]
            
            # Discard down to 7 cards if needed
            while len(current_player.hand) > 7:
                current_player.discard_card()
                
            # Empty mana pool
            current_player.mana_pool = {}
            
            # Reset "until end of turn" effects
            for card in current_player.battlefield:
                if hasattr(card, 'until_end_of_turn_effects'):
                    card.until_end_of_turn_effects = {}
            
            # Call cleanup step
            if not self.cleanup_step():
                print("Warning: cleanup_step failed")
                    
            return True
            
        except Exception as e:
            print(f"Error in _handle_cleanup_phase: {str(e)}")
            return False
        
    def cast_spell(self, player: Player, card: Card):
        """Cast a spell from a player's hand."""
        print(f"\n=== Casting {card.name} ===")
        if card not in player.hand:
            raise ValueError("Card must be in player's hand to cast")
            
        # Handle lands differently (no mana cost)
        if isinstance(card, Land):
            print("Handling land card")
            player.hand.remove(card)
            card.controller = player
            player.battlefield.append(card)
            return
            
        # Check if player has enough mana
        mana_cost = card.parse_mana_cost()
        if not player.can_pay_mana_cost(mana_cost):
            raise ValueError("Player cannot pay mana cost")
            
        # Pay mana cost
        player.pay_mana_cost(mana_cost)
        
        # Move card from hand to stack
        player.hand.remove(card)
        card.controller = player
        
        # Create effect based on card type
        if isinstance(card, Creature):
            print("Handling creature card")
            # Add creature to battlefield first
            player.battlefield.append(card)
            print(f"Added {card.name} to battlefield")
            
            # For creatures, create an effect that triggers enters-the-battlefield abilities
            effect = Effect(
                effect_type=EffectType.ENTERS_BATTLEFIELD,
                description=f"Put {card.name} onto the battlefield",
                source=card,
                conditions=[],  # No conditions for basic creature casting
                resolution=lambda: self._handle_creature_enters_battlefield(player, card)
            )
        else:
            print("Handling non-creature spell")
            # For other spells, create a generic effect
            effect = Effect(
                effect_type=EffectType.SPELL,
                description=f"Cast {card.name}",
                source=card,
                conditions=[],  # No conditions for basic spell casting
                resolution=lambda: player.graveyard.append(card)
            )
            
        print(f"Adding effect to stack: {effect.description}")
        self.stack.append(effect)
        print(f"Stack size before resolution: {len(self.stack)}")
        self.resolve_stack()  # Immediately resolve the effect
        print(f"Stack size after resolution: {len(self.stack)}")
        
    def _handle_creature_enters_battlefield(self, player: Player, creature: Creature):
        """Handle a creature entering the battlefield."""
        print(f"\n=== Handling enters-the-battlefield for {creature.name} ===")
        print(f"Number of abilities: {len(creature.get_abilities())}")
        
        # Set the controller before resolving abilities
        creature.controller = player
        
        # Trigger enters-the-battlefield abilities
        for ability in creature.get_abilities():
            if ability.effect_type == EffectType.ENTERS_BATTLEFIELD:
                print(f"Found enters-the-battlefield ability: {ability.description}")
                # Resolve the ability directly instead of adding it to the stack
                ability.resolve()
                print(f"Resolved ability: {ability.description}")
        
        print(f"Stack size after resolving abilities: {len(self.stack)}")
        
    def resolve_stack(self):
        """Resolve all effects on the stack."""
        print("\n=== Resolving stack ===")
        while self.stack:
            effect = self.stack.pop()
            print(f"Resolving effect: {effect.description}")
            effect.resolve()
        print("Stack resolved")
        
    def declare_attackers(self, player: Player, attackers: List[Card]):
        """Declare attacking creatures."""
        
        if self.current_phase != Phase.COMBAT_DECLARE_ATTACKERS:
            raise ValueError("Not in declare attackers phase")
            
        if player != self.players[self.current_player_index]:
            raise ValueError("Not your turn")
            
        # Validate attackers
        for creature in attackers:
            if creature not in player.battlefield:
                raise ValueError("Creature must be on battlefield")
            if creature.tapped:
                raise ValueError("Tapped creatures cannot attack")
            if not isinstance(creature, Creature):
                raise ValueError("Only creatures can attack")
                
        # Clear previous attackers
        self.combat_state['attackers'] = []
        self.combat_state['blockers'] = {}
        
        # Set new attackers
        for creature in attackers:
            creature.tapped = True
            self.combat_state['attackers'].append(creature)
            
        
    def declare_blockers(self, defending_player: Player, blockers: Dict[Card, Card]):
        """Declare blocking creatures."""
        if self.current_phase != Phase.COMBAT_DECLARE_BLOCKERS:
            raise ValueError("Not in declare blockers phase")
            
        # The defending player should be the one who is not the current player
        if defending_player != self.players[(self.current_player_index + 1) % 2]:
            raise ValueError("Only the defending player can declare blockers")
            
        # Validate blockers
        for blocker, attacker in blockers.items():
            if blocker not in defending_player.battlefield:
                raise ValueError("Blocker must be on battlefield")
            if blocker.tapped:
                raise ValueError("Tapped creatures cannot block")
            if not isinstance(blocker, Creature):
                raise ValueError("Only creatures can block")
            if attacker not in self.combat_state['attackers']:
                raise ValueError("Invalid attacker")
            # Set the blocker's controller
            blocker.controller = defending_player
                
        self.combat_state['blockers'] = blockers
        
    def assign_damage(self):
        """Assign and resolve combat damage."""
        if self.current_phase != Phase.COMBAT_DAMAGE:
            raise ValueError("Not in damage phase")
            
        # Get the defending player
        defending_player = self.players[(self.current_player_index + 1) % 2]
        # Handle unblocked attackers first
        for attacker in self.combat_state['attackers']:
            if attacker not in self.combat_state['blockers'].values():
                # Deal damage to defending player
                defending_player.take_damage(attacker.power)
                
        # Handle blocked attackers
        destroyed_creatures = set()
        
        # First strike damage phase
        for blocker, attacker in list(self.combat_state['blockers'].items()):
            if attacker.first_strike:
                # First strike creatures deal damage first
                attacker.assign_damage(blocker)
                if blocker.is_destroyed():
                    destroyed_creatures.add(blocker)
                    del self.combat_state['blockers'][blocker]
                    
            if blocker.first_strike and blocker in blocker.controller.battlefield:
                blocker.assign_damage(attacker)
                if attacker.is_destroyed():
                    destroyed_creatures.add(attacker)
                    self.combat_state['attackers'].remove(attacker)
                    
        # Remove destroyed creatures from battlefield after first strike
        for creature in destroyed_creatures:
            if creature in creature.controller.battlefield:
                creature.controller.battlefield.remove(creature)
                creature.controller.graveyard.append(creature)
                
        # Regular damage phase
        destroyed_creatures.clear()
        for blocker, attacker in list(self.combat_state['blockers'].items()):
            if not attacker.first_strike:
                attacker.assign_damage(blocker)
                if blocker.is_destroyed():
                    destroyed_creatures.add(blocker)
                    del self.combat_state['blockers'][blocker]
                    
            if blocker in blocker.controller.battlefield and not blocker.first_strike:
                blocker.assign_damage(attacker)
                if attacker.is_destroyed():
                    destroyed_creatures.add(attacker)
                    self.combat_state['attackers'].remove(attacker)
                    
        # Remove destroyed creatures from battlefield after regular damage
        for creature in destroyed_creatures:
            if creature in creature.controller.battlefield:
                creature.controller.battlefield.remove(creature)
                creature.controller.graveyard.append(creature)
                
        # Mark damage as assigned
        self.combat_state['damage_assigned'] = True
        
    def end_turn(self):
        """End the current player's turn."""
        if self.current_phase != Phase.CLEANUP:
            raise ValueError("Must complete all phases before ending turn")
            
        # Switch to next player
        self.current_player_index = (self.current_player_index + 1) % 2
        self.current_phase = Phase.UNTAP
        self.turn_number += 1
        self.combat_state = {
            'attackers': [],
            'blockers': {},
            'damage_assigned': False
        }

    def cast_land(self, land: Land) -> bool:
        """Cast a land card from hand to battlefield."""
        if not isinstance(land, Land):
            return False
            
        current_player = self.players[self.current_player_index]
            
        # Remove land from hand
        current_player.hand.remove(land)
        
        # Add to battlefield
        current_player.battlefield.append(land)
        
        # Set controller
        land.controller = current_player
        
        # Increment lands played this turn
        current_player.lands_played_this_turn += 1
        
        return True

    def tap_land(self, land: Land) -> bool:
        """Tap a land to add mana to the player's mana pool."""
        if not isinstance(land, Land):
            return False
            
        current_player = self.players[self.current_player_index]
            
        # Check if land is already tapped
        if land.tapped:
            return False
            
        # Check if land is controlled by current player
        if land not in current_player.battlefield:
            return False
            
        # Add mana to player's pool
        for color in land.colors:
            current_player.mana_pool[color] = current_player.mana_pool.get(color, 0) + 1
            
        # Tap the land
        land.tapped = True
        
        return True

    def cleanup_step(self):
        """Handle cleanup step actions."""
        try:
            # Ensure we have valid players
            if not self.players:
                raise ValueError("No players in game")
            
            # Untap all permanents
            for player in self.players:
                if not hasattr(player, 'battlefield'):
                    print(f"Warning: Player {player.name} has no battlefield")
                    continue
                    
                for permanent in player.battlefield:
                    if permanent is None:
                        print("Warning: Found None permanent in battlefield")
                        continue
                    permanent.tapped = False
            
            # Handle damage on creatures
            for player in self.players:
                if not hasattr(player, 'battlefield'):
                    print(f"Warning: Player {player.name} has no battlefield")
                    continue
                    
                for creature in [card for card in player.battlefield if isinstance(card, Creature)]:
                    if creature is None:
                        print("Warning: Found None creature in battlefield")
                        continue
                        
                    # Reset damage at end of turn
                    creature.damage = 0
                    
                    # Handle regeneration shield
                    if hasattr(creature, 'regeneration_shield') and creature.regeneration_shield:
                        creature.regeneration_shield = False
                    
                    # Handle "until end of turn" effects
                    if hasattr(creature, 'until_end_of_turn_effects'):
                        creature.until_end_of_turn_effects = {}
            
            # Reset combat state
            self.combat_state = {
                'attackers': [],
                'blockers': {},
                'damage_assignment': {}
            }
            
            return True
            
        except Exception as e:
            print(f"Error in cleanup_step: {str(e)}")
            return False

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Check if any player has lost
        for player in self.players:
            if player.life <= 0:
                return True
                
        # Check if any player has run out of cards
        for player in self.players:
            if len(player.library) == 0:
                return True
                
        # Check if we've reached the maximum turn limit
        if self.turn_number >= 50:
            return True
            
        return False

    def get_active_player(self) -> Player:
        """Get the currently active player."""
        return self.players[self.current_player_index]

    def get_opponent(self, player: Player) -> Player:
        """Get the opponent of the given player."""
        if player not in self.players:
            raise ValueError("Player not in game")
        return self.players[(self.players.index(player) + 1) % 2]

    def get_valid_actions(self) -> List[str]:
        """Get a list of valid actions for the current game state.
        
        Returns:
            List[str]: A list of valid action strings
        """
        # Get current player and opponent
        current_player = self.players[self.current_player_index]
        opponent = self.players[(self.current_player_index + 1) % 2]
        
        # Get valid actions
        valid_actions = []
        if self.current_phase == Phase.CLEANUP:
            valid_actions.append("end_turn")
        elif self.current_phase == Phase.UNTAP:
            valid_actions.append("next_phase")
        elif self.current_phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]:
            if current_player.lands_played_this_turn == 0 and any(isinstance(card, Land) for card in current_player.hand):
                valid_actions.append("cast_land")
            if any(isinstance(card, Land) and not card.tapped for card in current_player.battlefield):
                valid_actions.append("tap_land")
            if any(not isinstance(card, Land) and current_player.can_pay_mana_cost(card.parse_mana_cost()) 
                  for card in current_player.hand):
                valid_actions.append("cast_spell")
            valid_actions.append("next_phase")
        elif self.current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
            if any(isinstance(card, Creature) and not card.tapped for card in current_player.battlefield):
                valid_actions.append("declare_attackers")
            valid_actions.append("next_phase")
        elif self.current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
            if any(isinstance(card, Creature) and not card.tapped for card in opponent.battlefield) and self.combat_state['attackers']:
                valid_actions.append("declare_blockers")
            valid_actions.append("next_phase")
        else:
            valid_actions.append("next_phase")
            
        return valid_actions

    def get_state_tensor(self) -> Tuple[np.ndarray, List[str]]:
        """Get a state tensor representation of the current game state.
        
        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing:
                - The state tensor as a numpy array
                - A list of valid actions for the current state
        """
        # Initialize state tensor with size 2400 (120 * 20)
        state_size = 2400
        state = np.zeros(state_size)
        
        # Get current player and opponent
        current_player = self.players[self.current_player_index]
        opponent = self.players[(self.current_player_index + 1) % 2]
        
        # Encode current phase (one-hot encoding)
        phases = [
            Phase.UNTAP,
            Phase.UPKEEP,
            Phase.DRAW,
            Phase.MAIN_PHASE_1,
            Phase.COMBAT_BEGINNING,
            Phase.COMBAT_DECLARE_ATTACKERS,
            Phase.COMBAT_DECLARE_BLOCKERS,
            Phase.COMBAT_DAMAGE,
            Phase.COMBAT_END,
            Phase.MAIN_PHASE_2,
            Phase.END_STEP,
            Phase.CLEANUP
        ]
        phase_index = phases.index(self.current_phase)
        state[phase_index] = 1
        
        # Encode player life totals (normalized to 0-1)
        state[12] = current_player.life / 20.0
        state[13] = opponent.life / 20.0
        
        # Encode number of cards in hand (normalized to 0-1)
        state[14] = len(current_player.hand) / 7.0
        state[15] = len(opponent.hand) / 7.0
        
        # Encode number of lands on battlefield (normalized to 0-1)
        current_player_lands = sum(1 for card in current_player.battlefield if isinstance(card, Land))
        opponent_lands = sum(1 for card in opponent.battlefield if isinstance(card, Land))
        state[16] = current_player_lands / 10.0
        state[17] = opponent_lands / 10.0
        
        # Encode number of creatures on battlefield (normalized to 0-1)
        current_player_creatures = sum(1 for card in current_player.battlefield if isinstance(card, Creature))
        opponent_creatures = sum(1 for card in opponent.battlefield if isinstance(card, Creature))
        state[18] = current_player_creatures / 10.0
        state[19] = opponent_creatures / 10.0
        
        # Encode mana pool (normalized to 0-1)
        for i, color in enumerate(Color):
            state[20 + i] = current_player.mana_pool.get(color, 0) / 10.0
            
        # Encode card features (up to 20 cards per zone)
        zones = [
            (current_player.hand, 100),  # Start at index 100
            (current_player.battlefield, 500),  # Start at index 500
            (current_player.graveyard, 900),  # Start at index 900
            (opponent.hand, 1300),  # Start at index 1300
            (opponent.battlefield, 1700),  # Start at index 1700
            (opponent.graveyard, 2100)  # Start at index 2100
        ]
        
        for zone, start_idx in zones:
            for i, card in enumerate(zone[:20]):  # Only encode up to 20 cards per zone
                idx = start_idx + (i * 20)
                # Encode card type
                if isinstance(card, Land):
                    state[idx] = 1
                elif isinstance(card, Creature):
                    state[idx + 1] = 1
                    state[idx + 2] = card.power / 10.0
                    state[idx + 3] = card.toughness / 10.0
                elif isinstance(card, Spell):
                    state[idx + 4] = 1
                # Encode card color
                for j, color in enumerate(Color):
                    if color in card.colors:
                        state[idx + 5 + j] = 1
                # Encode mana cost
                mana_cost = card.parse_mana_cost()
                for j, color in enumerate(Color):
                    state[idx + 10 + j] = mana_cost.get(color, 0) / 10.0
                # Encode tapped state
                if hasattr(card, 'tapped'):
                    state[idx + 15] = 1 if card.tapped else 0
                # Encode controller
                state[idx + 16] = 1 if card.controller == current_player else 0
        
        # Get valid actions
        valid_actions = self.get_valid_actions()
            
        return state, valid_actions
