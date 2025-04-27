import numpy as np
from typing import List, Tuple, Dict, Any
from model import MTGAgent
from training_env import MTGTrainingEnv
from game import Phase
from cards import Land, Creature
from utils import get_valid_actions

def validate_agent(agent: MTGAgent, num_episodes: int = 100) -> Tuple[List[float], float, Dict[str, int]]:
    """
    Validate the agent's performance.
    
    Args:
        agent: The agent to validate
        num_episodes: Number of validation episodes
        
    Returns:
        Tuple of (rewards, win_rate, action_distribution)
    """
    env = MTGTrainingEnv()
    rewards = []
    wins = 0
    action_dist = {}
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = get_valid_actions(env)
            
            # Select action (no exploration during validation)
            action = agent.select_action(state, valid_actions, epsilon=0)
            
            # Update action distribution
            action_type = action[0]
            action_dist[action_type] = action_dist.get(action_type, 0) + 1
            
            # Take action
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
        
        # Check if agent won
        if env.opponent.life <= 0 or len(env.opponent.library) == 0:
            wins += 1
            
    win_rate = wins / num_episodes
    return rewards, win_rate, action_dist

def evaluate_deck_building(agent: MTGAgent, num_games: int = 100) -> Dict[str, float]:
    """
    Evaluate the agent's deck building decisions.
    
    Args:
        agent: The agent to evaluate
        num_games: Number of games to analyze
        
    Returns:
        Dictionary of deck building metrics
    """
    env = MTGTrainingEnv()
    metrics = {
        'avg_mana_curve': 0.0,
        'color_distribution': {},
        'spell_land_ratio': 0.0,
        'creature_spell_ratio': 0.0
    }
    
    for _ in range(num_games):
        env.reset()
        deck = env.current_player.library + env.current_player.hand
        
        # Calculate mana curve
        mana_costs = [card.parse_mana_cost()['total'] for card in deck if hasattr(card, 'parse_mana_cost')]
        metrics['avg_mana_curve'] += np.mean(mana_costs)
        
        # Calculate color distribution
        for card in deck:
            for color in card.colors:
                metrics['color_distribution'][color] = metrics['color_distribution'].get(color, 0) + 1
                
        # Calculate ratios
        num_lands = len([card for card in deck if isinstance(card, Land)])
        num_creatures = len([card for card in deck if isinstance(card, Creature)])
        num_spells = len(deck) - num_lands
        
        metrics['spell_land_ratio'] += num_spells / (num_lands if num_lands > 0 else 1)
        metrics['creature_spell_ratio'] += num_creatures / (num_spells if num_spells > 0 else 1)
        
    # Average metrics
    metrics['avg_mana_curve'] /= num_games
    metrics['spell_land_ratio'] /= num_games
    metrics['creature_spell_ratio'] /= num_games
    
    # Normalize color distribution
    total_colors = sum(metrics['color_distribution'].values())
    metrics['color_distribution'] = {
        color: count / total_colors 
        for color, count in metrics['color_distribution'].items()
    }
    
    return metrics

def analyze_game_decisions(agent: MTGAgent, num_games: int = 10) -> Dict[str, Any]:
    """
    Analyze the agent's in-game decisions.
    
    Args:
        agent: The agent to analyze
        num_games: Number of games to analyze
        
    Returns:
        Dictionary of decision metrics
    """
    env = MTGTrainingEnv()
    metrics = {
        'avg_turns': 0,
        'combat_stats': {
            'attack_rate': 0.0,
            'block_rate': 0.0,
            'successful_attacks': 0,
            'successful_blocks': 0
        },
        'resource_usage': {
            'land_play_rate': 0.0,
            'mana_efficiency': 0.0
        },
        'phase_distribution': {}
    }
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        turns = 0
        combat_opportunities = 0
        attacks_made = 0
        blocks_made = 0
        lands_played = 0
        land_opportunities = 0
        
        while not done:
            valid_actions = get_valid_actions(env)
            action = agent.select_action(state, valid_actions, epsilon=0)
            
            # Track phase distribution
            current_phase = env.game.current_phase
            metrics['phase_distribution'][current_phase] = metrics['phase_distribution'].get(current_phase, 0) + 1
            
            # Track combat decisions
            if "declare_attackers" in valid_actions:
                combat_opportunities += 1
                if action[0] == "declare_attackers":
                    attacks_made += 1
                    
            if "declare_blockers" in valid_actions:
                if action[0] == "declare_blockers":
                    blocks_made += 1
                    
            # Track resource usage
            if "cast_land" in valid_actions:
                land_opportunities += 1
                if action[0] == "cast_land":
                    lands_played += 1
                    
            state, _, done, _ = env.step(action)
            
            if env.game.current_phase == Phase.UNTAP:
                turns += 1
                
        metrics['avg_turns'] += turns
        metrics['combat_stats']['attack_rate'] += attacks_made / max(1, combat_opportunities)
        metrics['combat_stats']['block_rate'] += blocks_made / max(1, combat_opportunities)
        metrics['resource_usage']['land_play_rate'] += lands_played / max(1, land_opportunities)
        
    # Average metrics
    metrics['avg_turns'] /= num_games
    metrics['combat_stats']['attack_rate'] /= num_games
    metrics['combat_stats']['block_rate'] /= num_games
    metrics['resource_usage']['land_play_rate'] /= num_games
    
    # Normalize phase distribution
    total_phases = sum(metrics['phase_distribution'].values())
    metrics['phase_distribution'] = {
        phase: count / total_phases 
        for phase, count in metrics['phase_distribution'].items()
    }
    
    return metrics 