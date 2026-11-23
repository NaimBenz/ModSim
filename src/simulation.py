import numpy as np
from circuit_template import VEH_LENGTH, STRATEGIES, place_vehicles, step, gaps

"""simulate.py produit les données et critères.py les analyse. Le code est séparé pour permettre de tester différentes stratégies de conduite sans avoir à réécrire la simulation."""

# 1. Stratégie IDM (car following model)

def idm(gap, v, dv, p):
    """Intelligent Driver Model"""
    a = p['a']  # maximum acceleration
    b = p['b']  # comfortable deceleration
    v0 = p['v0']  # desired speed
    s0 = p['s0']  # minimum gap
    T = p['T']  # safe time headway, parameter that controls how much distance the vehicle wants to keep from the vehicle in front

    #s*, derired dynamical distance
    s_star = s0 + np.maximum(0, v * T + (v * dv) / (2 * np.sqrt(a * b)))
    gap = np.maximum(gap, 0.1)  # avoid division by zero
    return a * (1 - (v / v0) - (s_star / gap) ** 2)

STRATEGIES["idm"] = idm

IDM = {"strat": "idm",
       "a": 1.0,  # maximum acceleration (m/s^2)
       "b": 1.5,  # comfortable deceleration (m/s^2)
       "v0": 30.0,  # desired speed (m/s)
       "s0": 2.0,  # minimum gap (m)
       "T": 1.5,
       "delta": 4}  # safe time headway (s)

# 2. Place N vehicules on 1 lane

def int_single_lane(circuit, N, layout="uniform", seed=0, jitter=0.0):
    veh = place_vehicles(circuit, N, layout=layout, seed=seed)#calls f from circuit_template.py
    keep = (veh["direction"] == 1) & (veh["lane"]==0)#keep only one line and one direction
    s = np.sort(veh["s"][keep])[::-1].copy()
    if jitter > 0:
        rng = np.random.default_rng(seed)
        s += rng.uniform(-jitter, jitter, N)
    v = np.zeros(N)
    if np.any(gaps(s, circuit) < 0):
        raise ValueError("Vehicles are overlapping")

    return s, v

# 3. Simulation loop

def simulate(s0, v0, circuit, params, t_end=600.0, dt=0.1, algo="rk4"):
    """Step from circuit_template.py to simulate the dynamics of vehicles on a circuit. Step forgets about the past, so we need to keep track of the history of the simulation in this function."""
    K = int(round(t_end / dt))
    N = len(s0)
    S = np.empty((K+1, N))
    V = np.empty((K+1, N))
    A = np.empty((K+1, N))
    S[0], V[0] = s0, v0
    collisions = []
    s, v = np.array(s0, float), np.array(v0, float)

    for k in range(K):
        s, v, a, impacts = step(s, v, dt, circuit, params, algo)
        S[K+1], V[K+1], A[K+1] = s, v, a #keep all the history of the sim
        for n, dv in impacts:
            collisions.append(((k+1)*dt, n, dv))
    
    return {"t": np.arange(K+1)*dt, "s": S, "v": V, "a": A, "collisions": collisions, "L": circuit["L"], "dt": dt}



