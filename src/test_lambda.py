import numpy as np
import matplotlib.pyplot as plt
from ovd import build_jacobian, equilibrium_values, OVD

# Paramètres du circuit pour l'anneau de Sugiyama
# TODO: A changer parce qu'ici c'est juste un placeholder
L = 230.0
N = 22

# Calcul de l'équilibre : b (écart), Vb (vitesse), Vpb (dérivée V'(b))
b, Vb, Vpb = equilibrium_values(L, N, OVD)
a = OVD["a"]


def max_re(lam, Vb, Vpb, a, N):
    """Retourne la plus grande partie réelle des valeurs propres non nulles."""
    A = build_jacobian(lam, Vb, Vpb, a, N)
    ev = np.linalg.eigvals(A)
    # On retire la valeur propre nulle liée à la conservation de la taille du circuit
    ev_non_nulles = ev[np.abs(ev) > 1e-8]
    return np.max(ev_non_nulles.real)


def find_lambda_c(Vb, Vpb, a, N, lo=0.0, hi=5.0, tol=1e-4):
    """Trouve le seuil critique lambda_c par dichotomie."""
    if max_re(lo, Vb, Vpb, a, N) < 0:
        return 0.0  # Déja stable sans terme en lambda
    if max_re(hi, Vb, Vpb, a, N) >= 0:  # Les 2 bornes sont instables
        raise ValueError(f"Toujours instable à lambda = {hi}. Augmente la borne 'hi'.")

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if max_re(mid, Vb, Vpb, a, N) >= 0:
            lo = mid
        else:
            hi = mid
    return hi


# Seuil pour lequel lambda sépare la stabilité et l'instabilité
lam_c = find_lambda_c(Vb, Vpb, a, N)
print(f"Seuil numérique (valeurs propres) : lambda_c    = {lam_c:.4f}")

# On essaye tout les valeurs entre 0 et 5 pour lambda afin de tester la stabilité
lambdas = np.arange(0.0, 5.1, 0.1)
max_reelles = [max_re(lam, Vb, Vpb, a, N) for lam in lambdas]

# Graphique qui analyse la stabilité locale
plt.plot(
    lambdas, max_reelles, marker="o", markersize=4, label=r"$\max\,\mathrm{Re}(\mu)$"
)
plt.axhline(0, color="black", linestyle="dashed")
plt.axvline(lam_c, color="red", linestyle="dotted", label=rf"$\lambda_c = {lam_c:.3f}$")
plt.xlabel(r"$\lambda\ (\mathrm{s}^{-1})$")
plt.ylabel("Plus grande partie réelle")
plt.title(f"Stabilité du modèle OVD (L = {L} m, N = {N})")
plt.legend()
plt.grid()
plt.show()
