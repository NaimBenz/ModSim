import numpy as np
from ovd import build_jacobian, equilibrium_values, OVD
import matplotlib.pyplot as plt


# Paramètres du circuit pour Anneau de Sugiyama
# TODO: A changer parce qu'ici c'est juste en placeHolder
L = 300.0
N = 22

"""
Ecart à l'équilibre -> b
Vitesse à l'équilibre -> Vb
Dérivée de la vitesse optimale -> Vpb
"""

# Calcul de l'équilibre
b, Vb, Vpb = equilibrium_values(L, N, OVD)

# Valeurs de lambda à tester
lambdas = np.arange(0.0, 5.1, 0.1)

lambda_stable = []
max_reelles = []

for lam in lambdas:
    A = build_jacobian(lam, Vb, Vpb, OVD["a"], N)

    valeurs_propres = np.linalg.eigvals(A)

    # On retire la valeur propre nulle structurelle
    valeurs_non_nulles = valeurs_propres[np.abs(valeurs_propres) > 1e-8]

    max_reelle = np.max(valeurs_non_nulles.real)

    if max_reelle < -1e-8:
        print(f"lambda = {lam:.1f} : stable , {max_reelle}")
        lambda_stable.append(lam)
        max_reelles.append(max_reelle)
    else:
        print(f"lambda = {lam:.1f} : instable ")

"""
Ce graphe montre seulement que la stabilité augmente plus le
paramètre lambda augmente. 
Comme l'OVD avec l'extension + lambda*dv à été fait pour l'appliquer 
au scénario de l'anneau de Sugiyama, il va falloir à ce moment déterminer le meilleur lambda
qui garanti la meilleure stabilité.
"""

plt.plot(lambda_stable, max_reelles, marker="o")
plt.axhline(0, linestyle="dashed")
plt.xlabel("Lambda")
plt.ylabel("Plus grande partie réelle")
plt.title("Stabilité du modèle OVD-DV en fonction de lambda")
plt.grid()
plt.show()
