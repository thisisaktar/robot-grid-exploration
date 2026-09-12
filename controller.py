"""
Controller for IR-01 Energy-Aware Robot Grid Exploration
Online decision controller that uses only revealed information.
"""

from typing import Dict, List, Tuple, Optional
from grid_simulator import GridSimulator
from explorer import GridExplorer


class ExplorationController:
    """
    Online exploration controller following IR-01 rules.
    
    At each step, the controller may use only:
    - Cells revealed so far
    - Current robot position
    - Previous actions
    - Remaining energy
    """
    
    def __init__(self, grid_size: int = 30, max_energy: int = 400,
                 base_pos: Tuple[int, int] = (0, 0)):
        self.grid_size = grid_size
        self.max_energy = max_energy
        self.base_pos = base_pos
        self.simulator = GridSimulator(grid_size=grid_size, seed=20260911)
        self.explorer = GridExplorer(
            grid_size=grid_size, base_pos=base_pos, max_energy=max_energy
        )
        self.total_reachable = 0
        self.step_count = 0
        self.is_done = False
        self.action_history: List[str] = []
    
    def initialize_map(self, eval_seed: Optional[int] = None) -> Dict:
        self.simulator.generate_grid(eval_seed=eval_seed)
        self.total_reachable = self.simulator.total_reachable
        self.explorer.reset()
        
        # Initial sensing at base position
        visible = self.simulator.get_visible_cells(self.base_pos)
        self.explorer.update_revealed(visible)
        
        self.step_count = 0
        self.is_done = False
        self.action_history = []
        
        return {
            'position': self.base_pos,
            'energy': self.max_energy,
            'cells_revealed': len(self.explorer.revealed_cells),
            'total_reachable': self.total_reachable,
            'coverage': self._compute_coverage()
        }
    
    def step(self) -> Dict:
        if self.is_done:
            return self._get_done_result()
        
        # Get action from explorer (uses only revealed info)
        action = self.explorer.get_action()
        
        if action == 'DONE':
            self.is_done = True
            self.action_history.append(action)
            return self._get_done_result()
        
        # Execute action
        old_pos = self.explorer.current_pos
        new_pos = self.explorer.execute_action(action)
        
        # Sense new cells at new position
        visible = self.simulator.get_visible_cells(new_pos)
        self.explorer.update_revealed(visible)
        
        self.step_count += 1
        self.action_history.append(action)
        
        # Check termination
        if self.explorer.energy <= 0:
            self.is_done = True
        
        return {
            'action': action,
            'position': self.explorer.current_pos,
            'energy': self.explorer.energy,
            'cells_revealed': len(self.explorer.revealed_cells),
            'coverage': self._compute_coverage(),
            'step': self.step_count,
            'is_done': self.is_done
        }
    
    def _compute_coverage(self) -> float:
        if self.total_reachable == 0:
            return 0.0
        revealed_free = sum(1 for occ in self.explorer.revealed_cells.values() if occ == 0)
        return min(1.0, revealed_free / self.total_reachable)
    
    def _get_done_result(self) -> Dict:
        successful_return = (self.explorer.current_pos == self.base_pos and 
                           self.explorer.energy > 0)
        return {
            'action': 'DONE',
            'position': self.explorer.current_pos,
            'energy': self.explorer.energy,
            'cells_revealed': len(self.explorer.revealed_cells),
            'coverage': self._compute_coverage(),
            'step': self.step_count,
            'is_done': True,
            'successful_return': successful_return
        }
    
    def run_full_exploration(self, eval_seed: Optional[int] = None) -> Dict:
        self.initialize_map(eval_seed)
        
        max_steps = self.grid_size * self.grid_size * 4
        
        while not self.is_done and self.step_count < max_steps:
            result = self.step()
            if result['is_done']:
                break
        
        final = self._get_done_result()
        energy_used = self.max_energy - final['energy']
        
        return {
            'trajectory': self.explorer.trajectory,
            'total_steps': self.step_count,
            'final_coverage': final['coverage'],
            'remaining_energy': final['energy'],
            'energy_used': energy_used,
            'successful_return': final['successful_return'],
            'cells_revealed': final['cells_revealed'],
            'total_reachable': self.total_reachable,
            'energy_ratio': final['energy'] / self.max_energy,
        }
