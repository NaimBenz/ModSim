import numpy as np
from circuit_template import STRATEGIES, VEH_LENGTH


def equilibrium_values(L, N, p):
    v0 = p["v0"]
    hc = p["hc"]
    w = p["w"]

    # Ecart homogène
    b = L / N - VEH_LENGTH

    # Vitesse optimale à l'équilibre
    Vb = v0 / 2 * (np.tanh((b - hc) / w) + 1)

    # On tient compte du maximum à 0.1 dans ovd()
    if Vb <= 0.1:
        Vb = 0.1
        Vpb = 0.0
    else:
        Vpb = v0 / (2 * w) * (1 - np.tanh((b - hc) / w) ** 2)
    return b, Vb, Vpb


def build_jacobian(lam: float, V: float, Vp: float, a: float, N: int):
    alpha = (2 * a * Vp) / V
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
    return (
        a * (1 - (v / V) ** 2) + lam * dv
    )  # J'ai ajouter la prise en compte de la vitesse relative au modele ovd


STRATEGIES["ovd"] = ovd
# Pour les valeurs de a, v0, hc, w, j'ai mis lam = 1.4 car le graphique montre que
# la fonction est stable de 0.67 à 5 avec un lambda optimal en 4.5
# TODO: A tester plusieurs valeur de lambda pour le rapport
OVD = {"strat": "ovd", "a": 3.0, "v0": 30.0, "hc": 10.0, "w": 5.0, "lam": 1.4}
