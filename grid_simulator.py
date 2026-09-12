"""
Grid Simulator for IR-01 Energy-Aware Robot Grid Exploration
Handles grid generation, obstacle placement, reachability validation, and simulation mechanics.
"""

import numpy as np
from collections import deque
from typing import Set, Tuple, List, Dict, Optional


class GridSimulator:
    """
    Local grid-world simulator following IR-01 development rules.
    
    Grid Properties:
    - Size: 30x30
    - Base position: (0, 0)
    - Obstacle probability: 20%
    - Sensing radius: Manhattan distance <= 2
    """
    
    def __init__(self, grid_size: int = 30, obstacle_prob: float = 0.20, 
                 seed: int = 20260911):
        self.grid_size = grid_size
        self.obstacle_prob = obstacle_prob
        self.base_pos = (0, 0)
        self.seed = seed
        self.grid = None
        self.reachable_cells = None
        self.total_reachable = 0
        self.total_free = 0
        
    def generate_grid(self, eval_seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(eval_seed if eval_seed is not None else self.seed)
        
        while True:
            obstacles = rng.random((self.grid_size, self.grid_size)) < self.obstacle_prob
            grid = obstacles.astype(np.int8)
            grid[0, 0] = 0
            
            reachable = self._compute_reachable(grid)
            
            free_cells = int(np.sum(grid == 0))
            reachable_free = int(np.sum(reachable & (grid == 0)))
            
            if free_cells > 0:
                reachability_ratio = reachable_free / free_cells
                
                if reachability_ratio >= 0.70:
                    self.grid = grid
                    self.reachable_cells = reachable & (grid == 0)
                    self.total_reachable = int(np.sum(self.reachable_cells))
                    self.total_free = free_cells
                    return grid
    
    def _compute_reachable(self, grid: np.ndarray) -> np.ndarray:
        reachable = np.zeros_like(grid, dtype=bool)
        start = self.base_pos
        
        if grid[start[0], start[1]] == 1:
            return reachable
            
        queue = deque([start])
        reachable[start[0], start[1]] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                    not reachable[nx, ny] and grid[nx, ny] == 0):
                    reachable[nx, ny] = True
                    queue.append((nx, ny))
        
        return reachable
    
    def get_visible_cells(self, pos: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        x, y = pos
        visible = {}
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        visible[(nx, ny)] = int(self.grid[nx, ny])
        return visible
    
    def is_reachable_cell(self, pos: Tuple[int, int]) -> bool:
        return bool(self.reachable_cells[pos[0], pos[1]])
    
    def get_grid_info(self) -> dict:
        return {
            'grid_size': self.grid_size,
            'base_pos': self.base_pos,
            'total_reachable': self.total_reachable,
            'total_free': self.total_free,
            'obstacle_prob': self.obstacle_prob
        }
