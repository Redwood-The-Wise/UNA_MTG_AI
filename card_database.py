import json
import os
from typing import Dict, List, Optional
from cards import Card, Creature, Land, Spell, Color, CardType
from effects import Effect, EffectType, TriggerType

class CardDatabase:
    def __init__(self):
        self.cards: Dict[str, Card] = {}
        self.card_data_dir = "card_data"
        self._ensure_card_data_dir()
        self._load_cards()
        
    def _ensure_card_data_dir(self):
        """Ensure the card data directory exists."""
        if not os.path.exists(self.card_data_dir):
            os.makedirs(self.card_data_dir)
            
    def _load_cards(self):
        """Load all cards from the card data directory."""
        for filename in os.listdir(self.card_data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.card_data_dir, filename), 'r') as f:
                    card_data = json.load(f)
                    self._create_card_from_data(card_data)
                    
    def _create_card_from_data(self, data: Dict):
        """Create a card object from JSON data."""
        name = data['name']
        mana_cost = data['mana_cost']
        colors = [Color[color] for color in data['colors']]
        card_type = CardType[data['card_type']]
        
        if card_type == CardType.CREATURE:
            card = Creature(name, mana_cost, colors, data['power'], data['toughness'])
        elif card_type == CardType.LAND:
            card = Land(name, colors)
        else:
            card = Spell(name, mana_cost, colors, card_type)
            
        # Add abilities
        for ability_data in data.get('abilities', []):
            effect = Effect(
                effect_type=EffectType[ability_data['effect_type']],
                description=ability_data['description'],
                source=card,
                conditions=[lambda: True],  # TODO: Parse conditions
                resolution=lambda: None,  # TODO: Parse resolution
                trigger_type=TriggerType[ability_data['trigger_type']] if 'trigger_type' in ability_data else None
            )
            card.add_ability(effect)
            
        self.cards[name] = card
        
    def add_card(self, card: Card):
        """Add a card to the database and save it to disk."""
        self.cards[card.name] = card
        self._save_card(card)
        
    def _save_card(self, card: Card):
        """Save a card to disk."""
        filename = self._sanitize_filename(card.name)
        with open(os.path.join(self.card_data_dir, f"{filename}.json"), 'w') as f:
            json.dump(card.to_dict(), f, indent=2)
            
    def _sanitize_filename(self, name: str) -> str:
        """Convert a card name to a valid filename."""
        # Replace special characters with underscores
        sanitized = ''.join(c if c.isalnum() else '_' for c in name)
        # Remove duplicate underscores
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        return sanitized.lower()
        
    def get_card(self, name: str) -> Optional[Card]:
        """Get a card by name."""
        return self.cards.get(name)
        
    def get_cards_by_type(self, card_type: CardType) -> List[Card]:
        """Get all cards of a specific type."""
        return [card for card in self.cards.values() if card.card_type == card_type]
        
    def get_cards_by_color(self, color: Color) -> List[Card]:
        """Get all cards of a specific color."""
        return [card for card in self.cards.values() if color in card.colors]
        
    def get_cards_by_effect_type(self, effect_type: EffectType) -> List[Card]:
        """Get all cards with a specific effect type."""
        return [
            card for card in self.cards.values()
            if any(ability.effect_type == effect_type for ability in card.get_abilities())
        ]

# Sample cards for testing
SAMPLE_CARDS = [
    {
        "name": "Grizzly Bears",
        "mana_cost": "1G",
        "colors": ["G"],
        "card_type": "CREATURE",
        "power": 2,
        "toughness": 2,
        "abilities": [
            {
                "effect_type": "TRIGGERED",
                "description": "When Grizzly Bears enters the battlefield, draw a card",
                "trigger_type": "ENTER_BATTLEFIELD"
            }
        ]
    },
    {
        "name": "Island",
        "mana_cost": "",
        "colors": ["U"],
        "card_type": "LAND",
        "abilities": [
            {
                "effect_type": "ACTIVATED",
                "description": "T: Add U",
                "cost": {"tap": True}
            }
        ]
    },
    {
        "name": "Lightning Bolt",
        "mana_cost": "R",
        "colors": ["R"],
        "card_type": "INSTANT",
        "abilities": [
            {
                "effect_type": "SPELL",
                "description": "Deal 3 damage to any target"
            }
        ]
    },
    {
        "name": "Giant Growth",
        "mana_cost": "G",
        "colors": ["G"],
        "card_type": "INSTANT",
        "abilities": [
            {
                "effect_type": "SPELL",
                "description": "Target creature gets +3/+3 until end of turn"
            }
        ]
    },
    {
        "name": "Mountain",
        "mana_cost": "",
        "colors": ["R"],
        "card_type": "LAND",
        "abilities": [
            {
                "effect_type": "ACTIVATED",
                "description": "T: Add R",
                "cost": {"tap": True}
            }
        ]
    },
    {
        "name": "Forest",
        "mana_cost": "",
        "colors": ["G"],
        "card_type": "LAND",
        "abilities": [
            {
                "effect_type": "ACTIVATED",
                "description": "T: Add G",
                "cost": {"tap": True}
            }
        ]
    },
    {
        "name": "Llanowar Elves",
        "mana_cost": "G",
        "colors": ["G"],
        "card_type": "CREATURE",
        "power": 1,
        "toughness": 1,
        "abilities": [
            {
                "effect_type": "ACTIVATED",
                "description": "T: Add G",
                "cost": {"tap": True}
            }
        ]
    },
    {
        "name": "Shock",
        "mana_cost": "R",
        "colors": ["R"],
        "card_type": "INSTANT",
        "abilities": [
            {
                "effect_type": "SPELL",
                "description": "Deal 2 damage to any target"
            }
        ]
    }
]

def create_sample_database():
    """Create a card database with sample cards."""
    database = CardDatabase()
    for card_data in SAMPLE_CARDS:
        database.add_card(database._create_card_from_data(card_data))
    return database

if __name__ == "__main__":
    # Create and test the database
    database = create_sample_database()
    
    # Test getting cards
    print("\nTesting card retrieval:")
    bears = database.get_card("Grizzly Bears")
    print(f"Found Grizzly Bears: {bears.name} ({bears.card_type.value})")
    
    # Test getting cards by type
    print("\nTesting card type filtering:")
    creatures = database.get_cards_by_type(CardType.CREATURE)
    print(f"Found {len(creatures)} creatures:")
    for creature in creatures:
        print(f"- {creature.name}")
        
    # Test getting cards by color
    print("\nTesting color filtering:")
    green_cards = database.get_cards_by_color(Color.G)
    print(f"Found {len(green_cards)} green cards:")
    for card in green_cards:
        print(f"- {card.name}")
        
    # Test getting cards by effect type
    print("\nTesting effect type filtering:")
    triggered_cards = database.get_cards_by_effect_type(EffectType.TRIGGERED)
    print(f"Found {len(triggered_cards)} cards with triggered abilities:")
    for card in triggered_cards:
        print(f"- {card.name}") 