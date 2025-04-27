from typing import List, Tuple, Dict
from game import Phase
from cards import Land, Creature
from training_env import MTGTrainingEnv

def get_valid_actions(env: MTGTrainingEnv) -> List[Tuple[str, Dict]]:
    """
    Get list of valid actions for current game state.
    Returns a list of tuples (action_name, action_params).
    """
    valid_actions = []
    current_phase = env.game.current_phase
    
    # Quick phase-based action checks
    if current_phase == Phase.CLEANUP:
        return [("end_turn", {})]
        
    if current_phase in [Phase.MAIN_PHASE_1, Phase.MAIN_PHASE_2]:
        # Check for land drop
        if len([card for card in env.current_player.hand if isinstance(card, Land)]) > 0:
            if len([card for card in env.current_player.battlefield if isinstance(card, Land)]) < 5:
                valid_actions.append(("cast_land", {}))
        
        # Check for spell casting
        if any(not isinstance(card, Land) for card in env.current_player.hand):
            valid_actions.append(("cast_spell", {}))
            
    elif current_phase == Phase.COMBAT_DECLARE_ATTACKERS:
        if any(isinstance(card, Creature) for card in env.current_player.battlefield):
            valid_actions.append(("declare_attackers", {}))
            
    elif current_phase == Phase.COMBAT_DECLARE_BLOCKERS:
        if any(isinstance(card, Creature) for card in env.opponent.battlefield) and env.game.combat_state['attackers']:
            valid_actions.append(("declare_blockers", {}))
    
    # Always allow next_phase except in cleanup
    if current_phase != Phase.CLEANUP:
        valid_actions.append(("next_phase", {}))
    
    return valid_actions 