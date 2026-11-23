import numpy as np
from circuit_template import B_MAX

def indice(sim,t):
    """Converts a time t into an index in the simulation data."""
    return int(round(t / sim["dt"]))

def debit(sim, t_w):
    """Q = 3600/ (L(t_end - t_w)) * sum of distances traveled"""
    k_w = indice(sim, t_w)
    distance = sim["s"][-1] - sim["s"][k_w]#distancce of all vehicles 
    duree = sim["t"][-1] - sim["t"][k_w]
    return 3600 * distance.sum() / (sim["L"] * duree)

def collisions(sim):
    """C: Returns the number of collisions in the simulation"""
    vitesses = np.array([dv for (_, _, dv) in sim["collisions"]])
    return {"C": len(vitesses),
            "dv_moy": vitesses.mean() if len(vitesses) > 0 else 0.0,
            "dv_max": vitesses.max() if len(vitesses) > 0 else 0.0}

def deceleration(sim, t_w):

    a = sim["a"][indice(sim, t_w):]#accelerations in the time window
    dec = np.maximum(0.0, -a)#only decelerations
    return {"D_max": dec.max(),
            "D_99": np.percentile(dec, 99),
            "frac_sat": np.mean(a <= -B_MAX+1e-6)}#fraction of time spent at saturation deceleration