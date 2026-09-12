"""
Explorer for IR-01 Energy-Aware Robot Grid Exploration
Optimized for Ck x Ek (coverage x remaining energy ratio).
"""

from collections import deque
from typing import Set, Tuple, List, Dict, Optional


class GridExplorer:
    """
    Frontier-based exploration optimized for Ck x Ek.
    
    Strategy:
    1. Conservative energy margins to return with energy remaining
    2. Adaptive exploration phases based on coverage progress
    3. Info-gain scoring for efficient frontier selection
    """
    
    def __init__(self, grid_size: int = 30, base_pos: Tuple[int, int] = (0, 0),
                 max_energy: int = 400):
        self.grid_size = grid_size
        self.base_pos = base_pos
        self.max_energy = max_energy
        
        self.revealed_cells: Dict[Tuple[int, int], int] = {}
        self.current_pos = base_pos
        self.energy = max_energy
        self.trajectory: List[Tuple[int, int]] = [base_pos]
        self.visited_positions: Set[Tuple[int, int]] = {base_pos}
        self.steps = 0
        
        self.DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.DIR_NAMES = {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}
    
    def update_revealed(self, visible_cells: Dict[Tuple[int, int], int]):
        for pos, occ in visible_cells.items():
            self.revealed_cells[pos] = occ
    
    def _in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def _bfs_distances(self, start):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            d = dist[(x, y)]
            for dx, dy in self.DIRS:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in dist or not self._in_bounds(nx, ny):
                    continue
                if next_pos in self.revealed_cells and self.revealed_cells[next_pos] == 1:
                    continue
                dist[next_pos] = d + 1
                queue.append(next_pos)
        return dist
    
    def _bfs_path(self, start, goal):
        if start == goal:
            return [start]
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            (x, y), path = queue.popleft()
            for dx, dy in self.DIRS:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in visited or not self._in_bounds(nx, ny):
                    continue
                if next_pos in self.revealed_cells and self.revealed_cells[next_pos] == 1:
                    continue
                new_path = path + [next_pos]
                if next_pos == goal:
                    return new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return None
    
    def get_frontiers(self):
        frontiers = []
        for pos, occ in self.revealed_cells.items():
            if occ != 0 or pos == self.current_pos:
                continue
            for dx, dy in self.DIRS:
                nx, ny = pos[0] + dx, pos[1] + dy
                if self._in_bounds(nx, ny) and (nx, ny) not in self.revealed_cells:
                    frontiers.append(pos)
                    break
        return frontiers
    
    def _estimate_info_gain(self, pos):
        x, y = pos
        count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny) and (nx, ny) not in self.revealed_cells:
                        count += 1
        return count
    
    def _get_coverage(self):
        total = sum(1 for v in self.revealed_cells.values() if v == 0)
        reachable = self._estimate_reachable_free()
        if reachable == 0:
            return 0
        return min(1.0, total / reachable)
    
    def _estimate_reachable_free(self):
        """Estimate total reachable free cells using BFS from base through revealed + unknown."""
        reachable = set()
        queue = deque([self.base_pos])
        reachable.add(self.base_pos)
        while queue:
            x, y = queue.popleft()
            for dx, dy in self.DIRS:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                if next_pos in reachable or not self._in_bounds(nx, ny):
                    continue
                if next_pos in self.revealed_cells and self.revealed_cells[next_pos] == 1:
                    continue
                reachable.add(next_pos)
                queue.append(next_pos)
        return len(reachable)
    
    def _energy_margin(self):
        return 15
    
    def get_action(self) -> str:
        frontiers = self.get_frontiers()
        
        dist_from_robot = self._bfs_distances(self.current_pos)
        
        dist_to_base = dist_from_robot.get(self.base_pos)
        if dist_to_base is None:
            dist_to_base = self.grid_size * 2
        
        margin = self._energy_margin()
        
        if self.energy <= dist_to_base + margin:
            if self.current_pos != self.base_pos:
                path = self._bfs_path(self.current_pos, self.base_pos)
                if path and len(path) > 1:
                    dx = path[1][0] - path[0][0]
                    dy = path[1][1] - path[0][1]
                    return self.DIR_NAMES[(dx, dy)]
            return 'DONE'
        
        best = None
        best_score = -1
        
        for f in frontiers:
            d_to_f = dist_from_robot.get(f)
            if d_to_f is None or d_to_f == 0:
                continue
            
            total_needed = d_to_f + dist_to_base + margin
            if total_needed > self.energy:
                continue
            
            info_gain = self._estimate_info_gain(f)
            if info_gain == 0:
                continue
            
            score = info_gain / max(1, d_to_f)
            
            if f not in self.visited_positions:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best = f
        
        if best is not None:
            path = self._bfs_path(self.current_pos, best)
            if path and len(path) > 1:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                return self.DIR_NAMES[(dx, dy)]
        
        if self.current_pos != self.base_pos:
            path = self._bfs_path(self.current_pos, self.base_pos)
            if path and len(path) > 1:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                return self.DIR_NAMES[(dx, dy)]
        
        return 'DONE'
    
    def execute_action(self, action: str) -> Tuple[int, int]:
        if action == 'DONE':
            return self.current_pos
        
        DIR_MAP = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        dx, dy = DIR_MAP[action]
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        if not self._in_bounds(new_pos[0], new_pos[1]):
            self.energy -= 1
            self.steps += 1
            return self.current_pos
        
        if new_pos in self.revealed_cells and self.revealed_cells[new_pos] == 1:
            self.energy -= 1
            self.steps += 1
            return self.current_pos
        
        self.current_pos = new_pos
        self.energy -= 1
        self.steps += 1
        self.trajectory.append(new_pos)
        self.visited_positions.add(new_pos)
        return new_pos
    
    def reset(self):
        self.revealed_cells = {}
        self.current_pos = self.base_pos
        self.energy = self.max_energy
        self.trajectory = [self.base_pos]
        self.visited_positions = {self.base_pos}
        self.steps = 0
