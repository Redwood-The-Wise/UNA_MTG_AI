from enum import Enum
from typing import List, Dict, Callable, Any, TYPE_CHECKING, ForwardRef
from dataclasses import dataclass

if TYPE_CHECKING:
    from cards import Card, CardType, Color

class TriggerType(Enum):
    ENTER_BATTLEFIELD = "enter_battlefield"
    LEAVE_BATTLEFIELD = "leave_battlefield"
    TAP = "tap"
    UNTAP = "untap"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_RECEIVED = "damage_received"
    SPELL_CAST = "spell_cast"
    ABILITY_ACTIVATED = "ability_activated"

class EffectType(Enum):
    """Types of effects in the game"""
    # Basic effect types
    SPELL = "spell"
    ABILITY = "ability"
    TRIGGERED = "triggered"
    ACTIVATED = "activated"
    STATIC = "static"
    
    # State change effects
    ENTERS_BATTLEFIELD = "enters_battlefield"
    LEAVES_BATTLEFIELD = "leaves_battlefield"
    DAMAGE = "damage"
    HEALING = "healing"
    COUNTER = "counter"
    DESTROY = "destroy"
    EXILE = "exile"
    TAP = "tap"
    UNTAP = "untap"
    
    # Card manipulation effects
    DRAW = "draw"
    DISCARD = "discard"
    SEARCH = "search"
    SHUFFLE = "shuffle"
    COPY = "copy"
    TOKEN = "token"
    
    # Counter effects
    COUNTER_ADD = "counter_add"
    COUNTER_REMOVE = "counter_remove"
    
    # Card modification effects
    MODIFY_POWER = "modify_power"
    MODIFY_TOUGHNESS = "modify_toughness"
    MODIFY_KEYWORD = "modify_keyword"
    MODIFY_COLOR = "modify_color"
    MODIFY_TYPE = "modify_type"
    MODIFY_SUBTYPE = "modify_subtype"
    MODIFY_ABILITY = "modify_ability"
    MODIFY_MANA_COST = "modify_mana_cost"
    MODIFY_CMC = "modify_cmc"
    
    # Player state effects
    MODIFY_LIFE = "modify_life"
    MODIFY_HAND_SIZE = "modify_hand_size"
    MODIFY_MAX_HAND_SIZE = "modify_max_hand_size"
    MODIFY_LIBRARY_SIZE = "modify_library_size"
    MODIFY_GRAVEYARD_SIZE = "modify_graveyard_size"
    MODIFY_EXILE_SIZE = "modify_exile_size"
    MODIFY_STACK_SIZE = "modify_stack_size"
    MODIFY_BATTLEFIELD_SIZE = "modify_battlefield_size"
    
    # Mana effects
    ADD_MANA = "add_mana"
    REMOVE_MANA = "remove_mana"
    MODIFY_MANA_POOL = "modify_mana_pool"
    MODIFY_MANA_BASE = "modify_mana_base"
    MODIFY_MANA_COLOR = "modify_mana_color"
    MODIFY_MANA_TYPE = "modify_mana_type"
    MODIFY_MANA_AMOUNT = "modify_mana_amount"
    MODIFY_MANA_PRODUCTION = "modify_mana_production"
    MODIFY_MANA_CONSUMPTION = "modify_mana_consumption"
    MODIFY_MANA_EFFICIENCY = "modify_mana_efficiency"
    MODIFY_MANA_BALANCE = "modify_mana_balance"
    MODIFY_MANA_SYNERGY = "modify_mana_synergy"
    MODIFY_MANA_ANTISYNERGY = "modify_mana_antisynergy"
    
    # Color-related effects
    MODIFY_COLOR_IDENTITY = "modify_color_identity"
    MODIFY_COLOR_WEIGHT = "modify_color_weight"
    MODIFY_COLOR_BALANCE = "modify_color_balance"
    MODIFY_COLOR_SYNERGY = "modify_color_synergy"
    MODIFY_COLOR_ANTISYNERGY = "modify_color_antisynergy"
    MODIFY_COLOR_IDENTITY_WEIGHT = "modify_color_identity_weight"
    MODIFY_COLOR_IDENTITY_BALANCE = "modify_color_identity_balance"
    MODIFY_COLOR_IDENTITY_SYNERGY = "modify_color_identity_synergy"
    MODIFY_COLOR_IDENTITY_ANTISYNERGY = "modify_color_identity_antisynergy"
    MODIFY_COLOR_IDENTITY_WEIGHT_BALANCE = "modify_color_identity_weight_balance"
    MODIFY_COLOR_IDENTITY_WEIGHT_SYNERGY = "modify_color_identity_weight_synergy"
    MODIFY_COLOR_IDENTITY_WEIGHT_ANTISYNERGY = "modify_color_identity_weight_antisynergy"
    MODIFY_COLOR_IDENTITY_WEIGHT_BALANCE_SYNERGY = "modify_color_identity_weight_balance_synergy"
    MODIFY_COLOR_IDENTITY_WEIGHT_BALANCE_ANTISYNERGY = "modify_color_identity_weight_balance_antisynergy"
    MODIFY_COLOR_IDENTITY_WEIGHT_BALANCE_SYNERGY_ANTISYNERGY = "modify_color_identity_weight_balance_synergy_antisynergy"

@dataclass
class Effect:
    """Represents a game effect"""
    effect_type: EffectType
    description: str
    source: 'Card'
    conditions: List[Callable[[], bool]]
    resolution: Callable[[], Any]
    cost: Dict[str, Any] = None  # For activated abilities
    trigger_type: TriggerType = None  # For triggered abilities

    def can_resolve(self) -> bool:
        """Check if all conditions are met"""
        return all(condition() for condition in self.conditions)
        
    def resolve(self):
        """Execute the effect's resolution"""
        if self.can_resolve():
            self.resolution()
            
    def activate(self):
        """Activate the effect, paying any costs."""
        if self.effect_type == EffectType.ACTIVATED:
            if self.cost:
                if "tap" in self.cost and self.cost["tap"]:
                    if self.source.tapped:
                        raise ValueError("Source is already tapped")
                    self.source.tapped = True
                if "mana" in self.cost:
                    # Handle mana costs
                    for color, amount in self.cost["mana"].items():
                        if not self.source.controller.can_pay_mana_cost({color: amount}):
                            raise ValueError("Cannot pay mana cost")
                        self.source.controller.pay_mana_cost({color: amount})
            self.resolution()  # Call resolution directly
        else:
            raise ValueError("Only activated abilities can be activated")

    def to_dict(self) -> Dict:
        """Convert effect to dictionary for JSON storage"""
        return {
            'effect_type': self.effect_type.value,
            'description': self.description,
            'source': self.source.name if self.source else None,
            'conditions': [str(c) for c in self.conditions],
            'cost': self.cost,
            'trigger_type': self.trigger_type.value if self.trigger_type else None
        }

class Ability:
    def __init__(self, name: str, effect: Effect):
        self.name = name
        self.effect = effect
        self.is_activated = False

    def can_activate(self) -> bool:
        if self.effect.effect_type == EffectType.ACTIVATED:
            if self.effect.cost:
                if "tap" in self.effect.cost and self.effect.cost["tap"]:
                    if self.effect.source.tapped:
                        return False
                if "mana" in self.effect.cost:
                    # Check if player can pay mana cost
                    for color, amount in self.effect.cost["mana"].items():
                        if not self.effect.source.controller.can_pay_mana_cost({color: amount}):
                            return False
        return True

    def activate(self):
        if self.can_activate():
            self.is_activated = True
            self.effect.activate()

class EffectManager:
    def __init__(self):
        self.active_effects: List[Effect] = []
        self.triggered_abilities: List[Effect] = []
        self.replacement_effects: List[Effect] = []

    def add_effect(self, effect: Effect):
        if effect.effect_type == EffectType.STATIC:
            self.active_effects.append(effect)
        elif effect.effect_type == EffectType.TRIGGERED:
            self.triggered_abilities.append(effect)
        elif effect.effect_type == EffectType.REPLACEMENT:
            self.replacement_effects.append(effect)

    def remove_effect(self, effect: Effect):
        if effect in self.active_effects:
            self.active_effects.remove(effect)
        elif effect in self.triggered_abilities:
            self.triggered_abilities.remove(effect)
        elif effect in self.replacement_effects:
            self.replacement_effects.remove(effect)

    def check_triggers(self, trigger_type: TriggerType):
        for effect in self.triggered_abilities:
            if effect.trigger_type == trigger_type:
                if all(condition() for condition in effect.conditions):
                    effect.resolve()

    def apply_replacement_effects(self, event: Any) -> Any:
        for effect in self.replacement_effects:
            if all(condition() for condition in effect.conditions):
                event = effect.resolution(event)
        return event 