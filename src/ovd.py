import numpy as np
from circuit_template import STRATEGIES


def ovd(gap, v, dv, p):
    """Optimal velocity difference"""
    a, v0, hc, w = p["a"], p["v0"], p["hc"], p["w"]
    V = v0 / 2 * (np.tanh((gap - hc) / w) + 1)
    V = np.maximum(V, 0.1)  # avoid division by 0
    # TODO: add a lambda*dv for breaking
    return a * (1 - (v / V) ** 2)


STRATEGIES["ovd"] = ovd
OVD = {"strat": "ovd", "a": 2.0, "v0": 30.0, "hc": 25.0, "w": 10.0}
