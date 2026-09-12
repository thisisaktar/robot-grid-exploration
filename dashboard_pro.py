"""
Professional Dashboard for IR-01 Energy-Aware Robot Grid Exploration
Backend server with simulation engine.
"""
import os, json, time, threading, numpy as np
from flask import Flask, render_template, jsonify, request
from collections import deque

app = Flask(__name__, template_folder='templates')

class SimState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        with self.lock:
            self.revealed, self.position, self.trajectory = {}, [0,0], []
            self.coverage, self.step, self.energy = 0.0, 0, 400
            self.max_energy, self.total_reachable, self.grid_size = 400, 0, 30
            self.is_running, self.is_done, self.stop_requested = False, False, False
    def update(self, **kw):
        with self.lock:
            for k,v in kw.items(): setattr(self, k, v)
    def get_dict(self):
        with self.lock:
            return dict(revealed=dict(self.revealed), position=list(self.position),
                trajectory=[list(t) for t in self.trajectory], coverage=self.coverage,
                step=self.step, energy=self.energy, max_energy=self.max_energy,
                total_reachable=self.total_reachable, grid_size=self.grid_size,
                is_running=self.is_running, is_done=self.is_done)

sim_state = SimState()

class GridSim:
    def __init__(self, size=30, prob=0.2, seed=20260911):
        self.size, self.prob, self.seed, self.grid, self.total_reachable = size, prob, seed, None, 0
    def generate(self):
        rng = np.random.default_rng(self.seed)
        while True:
            g = (rng.random((self.size,self.size)) < self.prob).astype(int)
            g[0,0] = 0
            r = self._reach(g)
            free = int(np.sum(g==0)); rf = int(np.sum(r & (g==0)))
            if free > 0 and rf/free >= 0.70:
                self.grid = g; self.total_reachable = rf; return
    def _reach(self, g):
        r = np.zeros_like(g, dtype=bool); q = deque([(0,0)]); r[0,0] = True
        while q:
            x,y = q.popleft()
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx,ny = x+dx,y+dy
                if 0<=nx<self.size and 0<=ny<self.size and not r[nx,ny] and g[nx,ny]==0:
                    r[nx,ny] = True; q.append((nx,ny))
        return r

def run_sim(params):
    gs = GridSim(params.get('grid_size',30), params.get('obstacle',0.2), params.get('seed',20260911))
    gs.generate()
    sz = gs.size; energy = params.get('energy',400); sensing = params.get('sensing',2)
    speed = params.get('speed',0.05); pos = (0,0); revealed = {}; traj = [pos]
    visited = {pos}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    dn = {(-1,0):'UP',(1,0):'DOWN',(0,-1):'LEFT',(0,1):'RIGHT'}
    def ib(x,y): return 0<=x<sz and 0<=y<sz

    def bfs_dist(start):
        dist = {start: 0}; q = deque([start])
        while q:
            x,y = q.popleft(); d = dist[(x,y)]
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy; np_ = (nx,ny)
                if np_ in dist or not ib(nx,ny): continue
                if np_ in revealed and revealed[np_] == 1: continue
                dist[np_] = d+1; q.append(np_)
        return dist

    def bfs_path(start, goal):
        if start == goal: return [start]
        q = deque([(start, [start])]); v = {start}
        while q:
            (x,y), p = q.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy; np_ = (nx,ny)
                if np_ in v or not ib(nx,ny): continue
                if np_ in revealed and revealed[np_] == 1: continue
                if np_ == goal: return p + [np_]
                v.add(np_); q.append((np_, p + [np_]))
        return None

    def sense(p):
        for dx in range(-sensing,sensing+1):
            for dy in range(-sensing,sensing+1):
                if abs(dx)+abs(dy)<=sensing:
                    nx,ny=p[0]+dx,p[1]+dy
                    if ib(nx,ny) and (nx,ny) not in revealed: revealed[(nx,ny)]=int(gs.grid[nx,ny])

    def get_fr():
        fr=set()
        for p,occ in revealed.items():
            if occ!=0 or p==pos: continue
            for dx,dy in dirs:
                if ib(p[0]+dx,p[1]+dy) and (p[0]+dx,p[1]+dy) not in revealed: fr.add(p); break
        return list(fr)

    def ig(p):
        c=0
        for dx in range(-2,3):
            for dy in range(-2,3):
                if abs(dx)+abs(dy)<=2 and ib(p[0]+dx,p[1]+dy) and (p[0]+dx,p[1]+dy) not in revealed: c+=1
        return c

    sense(pos)
    sim_state.update(revealed={f"{k[0]},{k[1]}":v for k,v in revealed.items()},
        position=list(pos), trajectory=[list(t) for t in traj], coverage=0, step=0,
        energy=energy, max_energy=energy, total_reachable=gs.total_reachable,
        grid_size=sz, is_running=True, is_done=False)
    for step in range(sz*sz*4):
        if energy<=0 or sim_state.stop_requested: break

        dist_robot = bfs_dist(pos)
        dist_to_base = dist_robot.get((0,0))
        if dist_to_base is None:
            dist_to_base = sz * 2

        action = None

        if energy <= dist_to_base + 15:
            if pos != (0,0):
                path = bfs_path(pos, (0,0))
                if path and len(path) > 1:
                    action = dn.get((path[1][0]-pos[0], path[1][1]-pos[1]))
            if action is None:
                break
        else:
            fr = get_fr(); best, bs = None, -1
            for f in fr:
                d_to_f = dist_robot.get(f)
                if d_to_f is None or d_to_f == 0: continue
                if d_to_f + dist_to_base + 15 > energy: continue
                s = ig(f) / max(1, d_to_f)
                if f not in visited: s *= 1.2
                if s > bs: bs, best = s, f
            if best:
                path = bfs_path(pos, best)
                if path and len(path) > 1:
                    action = dn.get((path[1][0]-pos[0], path[1][1]-pos[1]))

        if action is None:
            if pos != (0,0):
                path = bfs_path(pos, (0,0))
                if path and len(path) > 1:
                    action = dn.get((path[1][0]-pos[0], path[1][1]-pos[1]))
            if action is None:
                break
        DM={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
        ddx,ddy=DM[action]; np_=(pos[0]+ddx,pos[1]+ddy)
        if ib(np_[0],np_[1]) and revealed.get(np_)!=1: pos=np_; energy-=1; traj.append(pos); visited.add(pos); sense(pos)
        else: energy-=1
        fr_c=sum(1 for v in revealed.values() if v==0)
        cov=min(1.0,fr_c/gs.total_reachable) if gs.total_reachable>0 else 0
        sim_state.update(revealed={f"{k[0]},{k[1]}":v for k,v in revealed.items()},
            position=list(pos), trajectory=[list(t) for t in traj],
            coverage=cov, step=step+1, energy=energy)
        time.sleep(speed)
    fr_c=sum(1 for v in revealed.values() if v==0)
    cov=min(1.0,fr_c/gs.total_reachable) if gs.total_reachable>0 else 0
    sim_state.update(coverage=cov, is_running=False, is_done=True, stop_requested=False)

@app.route('/')
def index(): return render_template('dashboard.html')

@app.route('/api/state')
def api_state(): return jsonify(sim_state.get_dict())

@app.route('/api/start', methods=['POST'])
def api_start():
    params = request.json; sim_state.reset(); sim_state.update(is_running=True)
    threading.Thread(target=run_sim, args=(params,), daemon=True).start()
    return jsonify({'ok':True})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    sim_state.stop_requested = True; return jsonify({'ok':True})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    sim_state.reset(); return jsonify({'ok':True})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("Dashboard: http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)
