"""
Run Experiment for IR-01 Energy-Aware Robot Grid Exploration
Main entry point for running experiments, logging, and visualization.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional

from controller import ExplorationController


def create_directories():
    for d in ['logs', 'visualizations', 'metrics']:
        os.makedirs(d, exist_ok=True)


def run_single_experiment(seed: int = 20260911, grid_size: int = 30,
                         max_energy: int = 400, verbose: bool = True) -> Dict:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed: {seed}")
        print(f"Grid size: {grid_size}x{grid_size}, Energy: {max_energy}")
        print(f"{'='*60}")
    
    controller = ExplorationController(grid_size=grid_size, max_energy=max_energy)
    
    start_time = time.time()
    results = controller.run_full_exploration(eval_seed=seed)
    end_time = time.time()
    
    results['runtime'] = end_time - start_time
    results['seed'] = seed
    
    if verbose:
        print(f"  Coverage: {results['final_coverage']:.4f} ({results['final_coverage']*100:.2f}%)")
        print(f"  Steps: {results['total_steps']}")
        print(f"  Energy used: {results['energy_used']}/{max_energy}")
        print(f"  Remaining energy: {results['remaining_energy']}")
        print(f"  Energy ratio (Ek): {results['energy_ratio']:.4f}")
        print(f"  Successful return: {results['successful_return']}")
        print(f"  Ck * Ek: {results['final_coverage'] * results['energy_ratio']:.4f}")
        print(f"  Runtime: {results['runtime']:.3f}s")
    
    return results


def run_multiple_experiments(seeds: List[int] = None, num_experiments: int = 10,
                           grid_size: int = 30, max_energy: int = 400) -> List[Dict]:
    if seeds is None:
        seeds = [20260911 + i for i in range(num_experiments)]
    
    results = []
    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment {i+1}/{len(seeds)}...")
        result = run_single_experiment(seed, grid_size, max_energy, verbose=True)
        results.append(result)
    
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    coverages = [r['final_coverage'] for r in results]
    energy_ratios = [r['energy_ratio'] for r in results]
    successful_returns = [r['successful_return'] for r in results]
    cek_values = [r['final_coverage'] * r['energy_ratio'] for r in results]
    runtimes = [r['runtime'] for r in results]
    
    return {
        'num_experiments': len(results),
        'mean_coverage': float(np.mean(coverages)),
        'std_coverage': float(np.std(coverages)),
        'min_coverage': float(np.min(coverages)),
        'max_coverage': float(np.max(coverages)),
        'mean_energy_ratio': float(np.mean(energy_ratios)),
        'successful_return_fraction': float(np.mean(successful_returns)),
        'mean_cek': float(np.mean(cek_values)),
        'mean_runtime': float(np.mean(runtimes)),
    }


def save_logs(results: List[Dict], metrics: Dict, prefix: str = "experiment"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Simplify results for JSON
    serializable = []
    for r in results:
        sr = {
            'seed': int(r['seed']),
            'total_steps': int(r['total_steps']),
            'final_coverage': float(r['final_coverage']),
            'remaining_energy': int(r['remaining_energy']),
            'energy_used': int(r['energy_used']),
            'successful_return': bool(r['successful_return']),
            'cells_revealed': int(r['cells_revealed']),
            'total_reachable': int(r['total_reachable']),
            'energy_ratio': float(r['energy_ratio']),
            'runtime': float(r['runtime']),
            'trajectory_len': len(r['trajectory']),
        }
        serializable.append(sr)
    
    with open(f"logs/{prefix}_results_{timestamp}.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    
    with open(f"metrics/{prefix}_metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nLogs saved to logs/ and metrics/")


def visualize_results(results: List[Dict], metrics: Dict, save_path: str = None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IR-01 Energy-Aware Robot Grid Exploration Results', fontsize=14)
    
    coverages = [r['final_coverage'] for r in results]
    energy_ratios = [r['energy_ratio'] for r in results]
    cek_values = [r['final_coverage'] * r['energy_ratio'] for r in results]
    
    # 1. Coverage distribution
    ax = axes[0, 0]
    ax.hist(coverages, bins=max(5, len(coverages)//2), edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_coverage'], color='r', linestyle='--', 
               label=f"Mean: {metrics['mean_coverage']:.3f}")
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Frequency')
    ax.set_title('Coverage Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Energy ratio distribution
    ax = axes[0, 1]
    ax.hist(energy_ratios, bins=max(5, len(energy_ratios)//2), edgecolor='black', alpha=0.7, color='green')
    ax.axvline(metrics['mean_energy_ratio'], color='r', linestyle='--',
               label=f"Mean: {metrics['mean_energy_ratio']:.3f}")
    ax.set_xlabel('Energy Ratio (Ek)')
    ax.set_ylabel('Frequency')
    ax.set_title('Energy Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Ck * Ek distribution
    ax = axes[0, 2]
    ax.hist(cek_values, bins=max(5, len(cek_values)//2), edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(metrics['mean_cek'], color='r', linestyle='--',
               label=f"Mean: {metrics['mean_cek']:.3f}")
    ax.set_xlabel('Ck * Ek')
    ax.set_ylabel('Frequency')
    ax.set_title('Ck * Ek Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Coverage vs Energy Ratio scatter
    ax = axes[1, 0]
    ax.scatter(energy_ratios, coverages, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Energy Ratio (Ek)')
    ax.set_ylabel('Coverage (Ck)')
    ax.set_title('Coverage vs Energy Ratio')
    ax.grid(True, alpha=0.3)
    
    # 5. Successful return pie chart
    ax = axes[1, 1]
    successful = sum(1 for r in results if r['successful_return'])
    unsuccessful = len(results) - successful
    if successful + unsuccessful > 0:
        ax.pie([successful, unsuccessful], 
               labels=['Successful', 'Unsuccessful'],
               autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'],
               startangle=90)
    ax.set_title(f'Successful Returns ({successful}/{len(results)})')
    
    # 6. Trajectory visualization (first experiment)
    ax = axes[1, 2]
    if results:
        trajectory = results[0]['trajectory']
        traj_x = [t[0] for t in trajectory]
        traj_y = [t[1] for t in trajectory]
        ax.plot(traj_y, traj_x, 'b-', alpha=0.5, linewidth=1)
        ax.plot(traj_y[0], traj_x[0], 'go', markersize=10, label='Start')
        ax.plot(traj_y[-1], traj_x[-1], 'rs', markersize=10, label='End')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_title(f'Trajectory (Exp 1, {len(trajectory)} steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 30)
        ax.set_ylim(30, -1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    plt.close()


def print_summary(metrics: Dict):
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Number of experiments: {metrics['num_experiments']}")
    print(f"Mean Coverage (Ck): {metrics['mean_coverage']:.4f} ({metrics['mean_coverage']*100:.2f}%)")
    print(f"Mean Energy Ratio (Ek): {metrics['mean_energy_ratio']:.4f}")
    print(f"Successful Return Fraction: {metrics['successful_return_fraction']:.4f}")
    print(f"Mean Ck * Ek: {metrics['mean_cek']:.4f}")
    print(f"Mean Runtime: {metrics['mean_runtime']:.3f}s")
    print("="*60)


def main():
    print("IR-01 Energy-Aware Robot Grid Exploration")
    print("Running experiments...")
    
    create_directories()
    
    num_experiments = 10
    results = run_multiple_experiments(num_experiments=num_experiments)
    metrics = compute_metrics(results)
    
    print_summary(metrics)
    save_logs(results, metrics)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualize_results(results, metrics, 
                     save_path=f"visualizations/results_{timestamp}.png")
    
    print("\nAll experiments completed!")
    return results, metrics


if __name__ == "__main__":
    main()
