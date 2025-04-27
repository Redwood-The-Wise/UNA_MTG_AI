from training_env import MTGTrainingEnv
import tqdm
import torch

def run_selfplay_stats(num_games=100):
    env = MTGTrainingEnv()
    stats = {
        'player1_wins': 0,
        'player2_wins': 0,
        'draws': 0,
        'total_turns': 0,
    }
    
    # Create progress bar for episodes
    episode_pbar = tqdm.tqdm(range(num_games), desc="Training Episodes")
    
    for episode in episode_pbar:
        state, _ = env.reset()  # Unpack the state tuple
        state = torch.FloatTensor(state)  # Convert to tensor
        total_reward = 0
        done = False
        step = 0
        
        # Create progress bar for steps within episode
        step_pbar = tqdm.tqdm(total=1000, desc=f"Episode {episode} Steps", leave=False)
        
        while not done and step < 10000:  # Add step limit to prevent infinite loops
            # Get current player's model manager
            current_player = env.game.get_active_player()
            current_model_manager = env.player1_model_manager if current_player == env.game.players[0] else env.player2_model_manager
            
            # Get valid actions
            valid_actions = env._get_valid_actions()
            
            # Select action using model
            action = current_model_manager.select_action(state, valid_actions)
            
            # Take action and get next state
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)  # Convert next state to tensor
            
            # Update total reward
            total_reward += reward
            
            # Move to next state
            state = next_state
            step += 1
            step_pbar.update(1)
        stats['total_turns'] += step
        if env.game.players[0].life <= 0:
            stats['player2_wins'] += 1
        elif env.game.players[1].life <= 0:
            stats['player1_wins'] += 1
        else:
            stats['draws'] += 1

    print(f"Player 1 win rate: {stats['player1_wins'] / num_games:.2%}")
    print(f"Player 2 win rate: {stats['player2_wins'] / num_games:.2%}")
    print(f"Draw rate: {stats['draws'] / num_games:.2%}")
    print(f"Average turns per game: {stats['total_turns'] / num_games:.2f}")

if __name__ == "__main__":
    run_selfplay_stats(100) 