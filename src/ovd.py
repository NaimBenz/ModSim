import numpy as np
from circuit_template import STRATEGIES


def build_jacobian(lam: float, V: float, DV: float, a: float, N: int):
    alpha = (2 * a * DV) / V
    beta = (-2 * a) / V - lam

    A_ss = np.zeros((N, N))  # Partie supérieure gauche de jacobienne
    A_sv = np.zeros((N, N))  # Partie superieur droite de jacobienne
    A_vs = np.zeros((N, N))  # Partie inférieure gauche de jacobienne
    A_vv = np.zeros((N, N))  # Partie inférieure droite de jacobienne

    for n in range(N):
        # Equation de l'écart
        A_sv[n, n] = -1
        A_sv[n, (n - 1) % N] = 1

        # Equation de la vitesse
        A_vs[n, n] = alpha
        A_vv[n, n] = beta
        A_vv[n, (n - 1) % N] = lam

    # Assemblage des quatres parties
    A = np.block([[A_ss, A_sv], [A_vs, A_vv]])

    return A


def ovd(gap, v, dv, p):
    """Optimal velocity difference"""
    a, v0, hc, w, lam = p["a"], p["v0"], p["hc"], p["w"], p["lam"]
    V = v0 / 2 * (np.tanh((gap - hc) / w) + 1)
    V = np.maximum(V, 0.1)  # Éviter la division par 0
    DV = v0 / 2 * w * (1 - np.tanh((gap - hc) / w) ** 2)
    return (
        a * (1 - (v / V) ** 2) + lam * dv
    )  # J'ai ajouter la prise en compte de la vitesse relative au modele ovd


STRATEGIES["ovd"] = ovd
# Pour l'instant, lambda = 0.5 mais je dois faire des calculs pour trouver par quoi je multiplie la prise
# en compte de la vitesse relative
OVD = {"strat": "ovd", "a": 2.0, "v0": 30.0, "hc": 25.0, "w": 10.0, "lam": 0.5}
