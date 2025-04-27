from typing import Dict, List, Optional, Any
from cards import Card, CardType, Color, Creature, Land, Spell
from effects import Effect, EffectType, Ability, TriggerType
import re

class CardParser:
    @staticmethod
    def parse_card(card_data: Dict[str, Any]) -> Card:
        """Parse card data into a Card object."""
        # Convert colors
        colors = [Color[color] for color in card_data['colors']]
        if not colors:
            colors = [Color.C]  # Colorless cards
            
        # Convert card type
        card_type = CardType[card_data['card_type']]
        
        # Create the appropriate card type
        if card_type == CardType.CREATURE:
            return CardParser._parse_creature(card_data, colors)
        elif card_type == CardType.LAND:
            return CardParser._parse_land(card_data, colors)
        else:
            return CardParser._parse_spell(card_data, colors, card_type)
    
    @staticmethod
    def _parse_creature(card_data: Dict[str, Any], colors: List[Color]) -> Creature:
        """Parse creature-specific data."""
        creature = Creature(
            name=card_data['name'],
            mana_cost=card_data.get('mana_cost', ''),
            colors=colors,
            power=int(card_data.get('power', 0)),
            toughness=int(card_data.get('toughness', 0))
        )
        
        # Parse abilities if present
        if 'abilities' in card_data:
            for ability_data in card_data['abilities']:
                ability = CardParser._parse_ability(ability_data, creature)
                creature.add_ability(ability)
                
        return creature
    
    @staticmethod
    def _parse_land(card_data: Dict[str, Any], colors: List[Color]) -> Land:
        """Parse land-specific data."""
        land = Land(
            name=card_data['name'],
            colors=colors
        )
        
        # Parse abilities if present
        if 'abilities' in card_data:
            for ability_data in card_data['abilities']:
                ability = CardParser._parse_ability(ability_data, land)
                land.add_ability(ability)
                
        return land
    
    @staticmethod
    def _parse_spell(card_data: Dict[str, Any], colors: List[Color], card_type: CardType) -> Spell:
        """Parse spell-specific data."""
        spell = Spell(
            name=card_data['name'],
            mana_cost=card_data.get('mana_cost', ''),
            colors=colors,
            card_type=card_type
        )
        
        # Parse abilities if present
        if 'abilities' in card_data:
            for ability_data in card_data['abilities']:
                ability = CardParser._parse_ability(ability_data, spell)
                spell.add_ability(ability)
                
        return spell
    
    @staticmethod
    def _parse_ability(ability_data: Dict[str, Any], source: Card) -> Ability:
        """Parse ability data into an Ability object."""
        effect_type = EffectType[ability_data['effect_type'].upper()]
        
        # Parse conditions
        conditions = []
        if 'conditions' in ability_data:
            conditions = CardParser._parse_conditions(ability_data['conditions'])
            
        # Parse resolution
        resolution = CardParser._parse_resolution(ability_data['resolution'])
        
        # Create effect
        effect = Effect(
            effect_type=effect_type,
            description=ability_data['description'],
            source=source,
            conditions=conditions,
            resolution=resolution,
            trigger_type=TriggerType[ability_data.get('trigger_type', 'ACTIVATED').upper()] if 'trigger_type' in ability_data else None
        )
        
        return Ability(ability_data['name'], effect)
    
    @staticmethod
    def _parse_conditions(conditions_data: List[Dict[str, Any]]) -> List[callable]:
        """Parse condition data into callable functions."""
        conditions = []
        for condition in conditions_data:
            # TODO: Implement condition parsing based on condition type
            # This will be expanded as we add more complex conditions
            conditions.append(lambda: True)  # Placeholder
        return conditions
    
    @staticmethod
    def _parse_resolution(resolution_data: str) -> callable:
        """Parse resolution data into a callable function."""
        # TODO: Implement resolution parsing based on resolution type
        # This will be expanded as we add more complex resolutions
        def default_resolution():
            print(f"Resolving: {resolution_data}")
        return default_resolution
    
    @staticmethod
    def parse_oracle_text(oracle_text: str) -> List[Dict[str, Any]]:
        """Parse oracle text into ability data."""
        abilities = []
        
        # Split oracle text into individual abilities
        ability_texts = re.split(r'\.\s*', oracle_text)
        
        for text in ability_texts:
            if not text.strip():
                continue
                
            # Determine ability type
            if text.startswith('{'):
                # Activated ability
                abilities.append({
                    'name': text,
                    'effect_type': 'ACTIVATED',
                    'description': text,
                    'conditions': [],
                    'resolution': 'activated_ability'
                })
            elif text.startswith('When') or text.startswith('Whenever') or text.startswith('At'):
                # Triggered ability
                abilities.append({
                    'name': text,
                    'effect_type': 'TRIGGERED',
                    'description': text,
                    'conditions': [],
                    'resolution': 'triggered_ability',
                    'trigger_type': 'ENTER_BATTLEFIELD'  # Default, should be determined from text
                })
            else:
                # Static ability
                abilities.append({
                    'name': text,
                    'effect_type': 'STATIC',
                    'description': text,
                    'conditions': [],
                    'resolution': 'static_ability'
                })
                
        return abilities 