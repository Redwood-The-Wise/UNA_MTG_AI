from enum import Enum
from typing import List, Dict, Optional, TYPE_CHECKING, ForwardRef
from dataclasses import dataclass, field
import random

if TYPE_CHECKING:
    from effects import Effect, EffectType

class Color(Enum):
    """Magic: The Gathering colors"""
    W = "white"
    U = "blue"
    B = "black"
    R = "red"
    G = "green"
    C = "colorless"

class CardType(Enum):
    """Magic: The Gathering card types"""
    CREATURE = "creature"
    LAND = "land"
    INSTANT = "instant"
    SORCERY = "sorcery"
    ARTIFACT = "artifact"
    ENCHANTMENT = "enchantment"
    PLANESWALKER = "planeswalker"
    TRIBAL = "tribal"
    BASIC = "basic"
    LEGENDARY = "legendary"
    SNOW = "snow"
    WORLD = "world"
    SCHEME = "scheme"
    PHENOMENON = "phenomenon"
    PLANE = "plane"
    VANGUARD = "vanguard"
    CONSPIRACY = "conspiracy"
    DUNGEON = "dungeon"

@dataclass
class Ability:
    """Represents a card ability"""
    name: str
    description: str
    effect_type: 'EffectType'
    conditions: List[str] = field(default_factory=list)
    resolution: List[str] = field(default_factory=list)

class Card:
    """Base class for all Magic: The Gathering cards"""
    def __init__(self, name: str, mana_cost: str, colors: List[Color], card_type: CardType):
        self.name = name
        self.mana_cost = mana_cost
        self.colors = colors
        self.card_type = card_type
        self.tapped = False
        self._abilities: List['Effect'] = []
        self.controller = None  # Add controller attribute
        
    def parse_mana_cost(self) -> Dict[str, int]:
        """Parse the mana cost string into a dictionary of colors and amounts."""
        cost_dict = {
            'W': 0,  # White
            'U': 0,  # Blue
            'B': 0,  # Black
            'R': 0,  # Red
            'G': 0,  # Green
            'C': 0   # Colorless
        }
        if not self.mana_cost:
            return cost_dict
            
        # Handle generic mana (numbers)
        generic = 0
        for char in self.mana_cost:
            if char.isdigit():
                generic = int(char)  # Set generic to the digit value
            elif char.upper() in ['W', 'U', 'B', 'R', 'G', 'C']:
                cost_dict[char.upper()] += 1
                
        # Add generic mana to colorless
        if generic > 0:
            cost_dict['C'] = generic  # Set generic mana directly
            
        return cost_dict
        
    def add_ability(self, ability: 'Effect'):
        """Add an ability to the card"""
        self._abilities.append(ability)
        
    def get_abilities(self) -> List['Effect']:
        """Get all abilities of the card"""
        return self._abilities.copy()
        
    def to_dict(self) -> Dict:
        """Convert card to dictionary for JSON storage"""
        return {
            'name': self.name,
            'mana_cost': self.mana_cost,
            'colors': [color.value for color in self.colors],
            'card_type': self.card_type.value,
            'tapped': self.tapped,
            'abilities': [ability.to_dict() for ability in self._abilities]
        }

class Creature(Card):
    """Represents a creature card"""
    def __init__(self, name: str, mana_cost: str, colors: List[Color], power: int, toughness: int):
        super().__init__(name, mana_cost, colors, CardType.CREATURE)
        self.power = power
        self.toughness = toughness
        self.damage = 0
        self.tapped = False
        
        # Creature abilities
        self.flying = False
        self.first_strike = False
        self.deathtouch = False
        self.vigilance = False
        self.haste = False
        
    def take_damage(self, amount: int):
        """Apply damage to the creature."""
        self.damage += amount
        if self.damage >= self.toughness:
            self.destroy()
            
    def destroy(self):
        """Move the creature to its controller's graveyard."""
        if self.controller:
            self.controller.battlefield.remove(self)
            self.controller.graveyard.append(self)
            
    def can_block(self, attacker: 'Creature') -> bool:
        """Check if this creature can block the attacker."""
        if self.tapped:
            return False
        # Flying creatures can only be blocked by creatures with flying
        if hasattr(attacker, 'flying') and attacker.flying and not hasattr(self, 'flying'):
            return False
        return True
        
    def assign_damage(self, target: 'Creature'):
        """Assign damage to another creature."""
        target.damage += self.power
        
    def is_destroyed(self) -> bool:
        """Check if the creature is destroyed."""
        return self.damage >= self.toughness
        
    def to_dict(self) -> Dict:
        """Convert creature to dictionary for JSON storage"""
        data = super().to_dict()
        data.update({
            'power': self.power,
            'toughness': self.toughness,
            'damage': self.damage,
            'tapped': self.tapped,
            'first_strike': self.first_strike,
            'deathtouch': self.deathtouch,
            'flying': self.flying,
            'vigilance': self.vigilance,
            'haste': self.haste
        })
        return data

class Land(Card):
    """A land card that can produce mana."""
    def __init__(self, name: str, colors: List[Color]):
        super().__init__(name, "0", colors, CardType.LAND)
        self.tapped = False
        
    def tap(self) -> None:
        """Tap the land to produce mana."""
        if not self.tapped:
            self.tapped = True
            return True
        return False
        
    def untap(self) -> None:
        """Untap the land."""
        self.tapped = False
        
    def to_dict(self) -> Dict:
        """Convert land to dictionary for JSON storage"""
        return super().to_dict()

class Spell(Card):
    """Represents an instant or sorcery spell"""
    def __init__(self, name: str, mana_cost: str, colors: List[Color], card_type: CardType):
        super().__init__(name, mana_cost, colors, card_type)
        
    def to_dict(self) -> Dict:
        """Convert spell to dictionary for JSON storage"""
        return super().to_dict()

class Player:
    """Represents a player in the game"""
    def __init__(self, name: str):
        """Initialize a player."""
        self.name = name
        self.life = 20
        self.hand = []
        self.library = []
        self.graveyard = []
        self.battlefield = []
        self.exile = []
        self.lands_played_this_turn = 0
        # Initialize mana pool with all colors set to 0
        self.mana_pool = {}
        self.empty_mana_pool()  # Use empty_mana_pool to ensure consistent initialization
        
    def draw_card(self):
        """Draw a card from the library."""
        if not self.library:
            return None
        card = self.library.pop()
        self.hand.append(card)
        return card
        
    def discard_card(self):
        """Discard a card from hand."""
        if self.hand:
            card = self.hand.pop()
            self.graveyard.append(card)
            
    def empty_mana_pool(self):
        """Empty the mana pool at the end of phase."""
        # Initialize or reset all color keys to 0
        self.mana_pool = {
            'W': 0,  # White
            'U': 0,  # Blue
            'B': 0,  # Black
            'R': 0,  # Red
            'G': 0,  # Green
            'C': 0   # Colorless
        }
        
    def add_mana(self, color: Color, amount: int = 1):
        """Add mana to the player's mana pool."""
        # Ensure mana pool is initialized
        if not self.mana_pool:
            self.empty_mana_pool()
        # Convert Color enum to string key
        color_key = color.name
        self.mana_pool[color_key] += amount
        
    def pay_mana_cost(self, cost: Dict[str, int]):
        """Pay mana from the player's mana pool."""
        # First check if we can pay the cost
        if not self.can_pay_mana_cost(cost):
            raise ValueError("Cannot pay mana cost")
            
        # Pay colored mana first
        for color in ['W', 'U', 'B', 'R', 'G']:
            if color in cost and cost[color] > 0:
                self.mana_pool[color] -= cost[color]
                
        # Then pay generic mana (colorless)
        if 'C' in cost and cost['C'] > 0:
            # Can use any color for generic mana
            remaining = cost['C']
            for color in ['W', 'U', 'B', 'R', 'G']:
                if remaining <= 0:
                    break
                available = min(self.mana_pool[color], remaining)
                self.mana_pool[color] -= available
                remaining -= available
                
    def can_pay_mana_cost(self, cost: Dict[str, int]) -> bool:
        """Check if the player can pay a mana cost."""
        # Ensure mana pool is initialized
        if not self.mana_pool:
            self.empty_mana_pool()
            
        # Count available mana from both mana pool and untapped lands
        available_mana = {
            'W': self.mana_pool.get('W', 0),
            'U': self.mana_pool.get('U', 0),
            'B': self.mana_pool.get('B', 0),
            'R': self.mana_pool.get('R', 0),
            'G': self.mana_pool.get('G', 0),
            'C': self.mana_pool.get('C', 0)
        }
        
        # Add mana from untapped lands
        for land in self.battlefield:
            if isinstance(land, Land) and not land.tapped:
                for color in land.colors:
                    color_key = color.name
                    available_mana[color_key] = available_mana.get(color_key, 0) + 1
                    
        # First check colored mana requirements
        for color in ['W', 'U', 'B', 'R', 'G']:
            if color in cost and available_mana.get(color, 0) < cost[color]:
                return False
                
        # Then check if we can pay generic mana with any color
        if 'C' in cost and cost['C'] > 0:
            total_available = sum(available_mana.get(color, 0) for color in ['W', 'U', 'B', 'R', 'G'])
            if total_available < cost['C']:
                return False
                
        return True

    def cast_land(self, land: Land):
        """Cast a land card."""
        if self.lands_played_this_turn >= 1:
            raise ValueError("Can only play one land per turn")
        if not isinstance(land, Land):
            raise ValueError("Can only cast land cards with this method")
        self.hand.remove(land)
        self.battlefield.append(land)
        self.lands_played_this_turn += 1
        
    def cast_spell(self, spell: Card) -> bool:
        """Cast a spell from hand."""
        if spell not in self.hand:
            return False
        cost = spell.parse_mana_cost()
        if not self.can_pay_mana_cost(cost):
            return False
        self.pay_mana_cost(cost)
        self.hand.remove(spell)
        if isinstance(spell, Creature):
            self.battlefield.append(spell)
            spell.controller = self
        else:
            self.graveyard.append(spell)
        return True

    def take_damage(self, amount: int):
        """Take damage and reduce life total."""
        self.life -= amount 