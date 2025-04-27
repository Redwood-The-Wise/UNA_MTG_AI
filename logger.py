import os
import json
import time
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class MTGLogger:
    def __init__(self, log_dir: str):
        """Initialize the logger."""
        self.log_dir = log_dir
        self.metrics = {
            'episode_rewards': [],
            'avg_rewards': [],
            'losses': [],
            'epsilons': [],
            'validation_rewards': [],
            'win_rates': [],
            'action_distributions': [],
            'phase_distributions': []
        }
        self.episode_start_time = time.time()
        self.training_start_time = time.time()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
    def log_episode(self, episode: int, episode_reward: float, loss: float, 
                   epsilon: float, action_dist: Dict[str, int], phase_dist: Dict[str, int]):
        """Log metrics for a single episode."""
        # Calculate average reward over last 100 episodes
        self.metrics['episode_rewards'].append(episode_reward)
        avg_reward = np.mean(self.metrics['episode_rewards'][-100:])
        self.metrics['avg_rewards'].append(avg_reward)
        
        # Log other metrics
        self.metrics['losses'].append(loss)
        self.metrics['epsilons'].append(epsilon)
        self.metrics['action_distributions'].append(action_dist)
        self.metrics['phase_distributions'].append(phase_dist)
        
        # Calculate time statistics
        episode_time = time.time() - self.episode_start_time
        total_time = time.time() - self.training_start_time
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Loss: {loss:.4f}")
            print(f"Epsilon: {epsilon:.2f}")
            print(f"Episode Time: {episode_time:.2f}s")
            print(f"Total Training Time: {total_time/3600:.2f}h")
            print("\nAction Distribution:")
            for action, count in action_dist.items():
                print(f"{action}: {count}")
            
        # Reset episode timer
        self.episode_start_time = time.time()
        
    def log_validation(self, episode: int, rewards: List[float], win_rate: float):
        """Log validation results."""
        avg_reward = np.mean(rewards)
        self.metrics['validation_rewards'].append(avg_reward)
        self.metrics['win_rates'].append(win_rate)
        
        print(f"\nValidation Results (Episode {episode + 1})")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        
    def save_metrics(self):
        """Save metrics to files."""
        # Save metrics as JSON
        metrics_file = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f)
            
        # Create plots
        self._plot_rewards()
        self._plot_loss()
        self._plot_win_rate()
        self._plot_action_distribution()
        
    def _plot_rewards(self):
        """Plot episode rewards and validation rewards."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['episode_rewards'], label='Episode Reward', alpha=0.3)
        plt.plot(self.metrics['avg_rewards'], label='Average Reward', linewidth=2)
        if self.metrics['validation_rewards']:
            x_vals = np.arange(0, len(self.metrics['episode_rewards']), 
                             len(self.metrics['episode_rewards']) // len(self.metrics['validation_rewards']))
            plt.plot(x_vals, self.metrics['validation_rewards'], 
                    label='Validation Reward', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training and Validation Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, 'rewards.png'))
        plt.close()
        
    def _plot_loss(self):
        """Plot training loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['losses'])
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, 'loss.png'))
        plt.close()
        
    def _plot_win_rate(self):
        """Plot validation win rate."""
        if self.metrics['win_rates']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['win_rates'])
            plt.xlabel('Validation Round')
            plt.ylabel('Win Rate')
            plt.title('Validation Win Rate')
            plt.grid(True)
            plt.savefig(os.path.join(self.run_dir, 'win_rate.png'))
            plt.close()
            
    def _plot_action_distribution(self):
        """Plot action distribution over time."""
        if not self.metrics['action_distributions']:
            return
            
        # Get all unique actions
        actions = set()
        for dist in self.metrics['action_distributions']:
            actions.update(dist.keys())
        actions = sorted(list(actions))
        
        # Create data for plotting
        data = np.zeros((len(self.metrics['action_distributions']), len(actions)))
        for i, dist in enumerate(self.metrics['action_distributions']):
            for j, action in enumerate(actions):
                data[i, j] = dist.get(action, 0)
                
        # Plot
        plt.figure(figsize=(12, 6))
        for i, action in enumerate(actions):
            plt.plot(data[:, i], label=action)
        plt.xlabel('Episode')
        plt.ylabel('Action Count')
        plt.title('Action Distribution Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'action_distribution.png'))
        plt.close() 