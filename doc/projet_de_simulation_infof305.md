# Projet de simulation INFOF305 2026-27
## Stratégies de suivi de véhicules sur circuit fermé : débit, sécurité, confort et stabilité

**Auteurs :** Gianluca Bontempi, Pascal Tribel  
**Année académique :** 2026-2027  

---

## 1. Contexte et motivation

Les modèles microscopiques de trafic dits de *car-following* décrivent le comportement de chaque conducteur par une équation différentielle qui relie son accélération à la situation qu'il perçoit devant lui. Pour un véhicule $n \in [1, N]$ avec $N$ le nombre de véhicules, de position $x_n$ et de vitesse $v_n = \dot{x}_n$, on écrit :

$$\dot{v}_n = f(s_n, v_n, \Delta v_n)$$
$$s_n = x_{n-1} - x_n - l, \quad \Delta v_n = v_{n-1} - v_n \quad (1)$$

où $s_n$ est la distance pare-chocs à pare-chocs au véhicule précédent (le leader $n-1$), $l$ la longueur des véhicules et $\Delta v_n$ la vitesse relative (positive si le leader s'éloigne).

La fonction $f$ encode une stratégie de conduite. Le choix de $f$ détermine des propriétés macroscopiques qui peuvent présenter des conflits : un conducteur qui suit de très près son leader augmente le débit mais aussi le risque de collision ; un conducteur très réactif amortit les perturbations mais freine brutalement.

Le système est dit à **l'équilibre homogène** lorsque la distance de chaque véhicule à son leader est identique (on la note $s_e$), et que la déviation standard de la distribution de la vitesse et de la vitesse relative des véhicules est nulle.

### Objectif du projet
Concevoir au moins trois fonctions $f(s, v, \Delta v)$, les implémenter dans le simulateur fourni et les comparer, par simulation et par analyse de ces simulations, selon quatre critères :
* débit,
* nombre de collisions,
* décélération maximale,
* distance à l'équilibre.

Plusieurs fonctions sont disponibles dans la littérature : si vous utilisez ou faites référence à ces fonctions, vous devez les citer correctement et rendre clair l'aspect original de votre approche.

---

## 2. Cadre de travail : le gabarit `circuit_template.py`

Le fichier `circuit_template.py` vous est fourni. Il contient :
* la classe `Circuit`, qui construit un circuit fermé lisse (une spline périodique) à partir de formes prédéfinies (`circle`, `stadium`), et qui convertit l'abscisse curviligne $s \in [0, L]$ en coordonnées $(x, y)$ ;
* la fonction `place_vehicles`, qui place $N$ véhicules selon une disposition `uniform`, `random` ou `platoon` ;
* la fonction `gaps`, qui calcule les distances $s_n$ sur le circuit périodique ;
* la fonction `plot_state`, qui dessine le circuit et les véhicules (éventuellement colorés selon leur vitesse) ;
* deux fonctions à compléter : `acceleration(gap, v, dv, params)` et `step(s, v, dt, circuit, params)`.

D'autres fonctions utilitaires y sont définies : vous pouvez bien entendu vous y référer.

Chaque véhicule étant décrit par son abscisse le long de la courbe, le modèle reste unidimensionnel : dans le template, la géométrie du circuit n'intervient dans la dynamique que par sa longueur $L$.

On vous fournit aussi un fichier Jupyter Notebook `Template.ipynb` dans lequel vous rédigerez le code de vos expériences.

---

## 3. Travail demandé

### 3.1 Conception des stratégies

Vous proposerez et implémenterez au moins trois fonctions $f$ de la forme (1) motivées par les critères suivants :
* Maximiser le débit sous contrainte de sécurité ;
* Minimiser les freinages brusques ;
* Amortir les ondes de type *stop-and-go*.

Pour chaque fonction, vous préciserez la signification physique de chaque paramètre, son unité et la valeur retenue (après expérimentation).

#### Règles communes (pour une comparaison équitable)
* $f$ n'utilise que $(s, v, \Delta v)$ du véhicule et de son leader direct, et elle est déterministe.
* Les paramètres de $f$ sont fixés une fois pour toutes : il est interdit de les ajuster scénario par scénario.
* Limites physiques imposées par le simulateur : $\dot{v} \in [-b_{max}, a_{max}]$ avec $b_{max} = 9\text{ m/s}^2$ et $a_{max} = 3\text{ m/s}^2$ ; vitesse $0 \le v \le v_{lim} = 30\text{ m/s}$.
* Intégration numérique par des méthodes vues aux TPs.
* **Collision :** lorsque $s_n < 0$, l'événement est comptabilisé, le suiveur est replacé à distance nulle derrière son leader et prend la vitesse de celui-ci ; la simulation continue.

### 3.2 Critères d'évaluation des stratégies

Les grandeurs suivantes sont calculées sur une fenêtre de mesure $[t_w, t_{end}]$.

* **Débit $Q$ (à maximiser) :** Nombre de véhicules franchissant une section par heure, estimé à partir des distances parcourues (positions non repliées modulo $L$) :
  $$Q = \frac{3600}{L(t_{end} - t_w)} \sum_{n=1}^{N} [x_n(t_{end}) - x_n(t_w)] \quad [\text{veh/h}].$$

* **Collisions $C$ (à minimiser) :** Nombre d'événements de collision sur toute la simulation. On rapportera aussi la vitesse relative à l'impact, indicateur de gravité.

* **Décélération maximale $D_{max}$ (à minimiser) :** $D_{max} = \max_{n,t}(0, -\dot{v}_n(t))$, décélération effectivement appliquée (après saturation). On rapportera également le $99^\text{e}$ centile $D_{99}$, plus robuste, ainsi que la fraction du temps passé en saturation $\dot{v} = -b_{max}$.

* **Distance à l'équilibre $E$ (à minimiser) :** Écart normalisé entre l'état du système et l'équilibre homogène :
  $$E(t) = \sqrt{\frac{1}{N} \sum_{n=1}^{N} \left[ \left(\frac{s_n(t) - s_e}{s_e}\right)^2 + \left(\frac{v_n(t) - \overline{v}(t)}{\max(\overline{v}(t), 1)}\right)^2 + \left(\frac{\dot{v}_n(t)}{a_{max}}\right)^2 \right]}$$
  $$s_e = \frac{L}{N} - l$$
  où $\overline{v}(t) = \frac{1}{N} \sum_{n=1}^{N} v_n(t)$. On rapportera la valeur moyenne $\overline{E}$ sur la fenêtre de mesure et le temps de convergence $t_\epsilon$ (premier instant à partir duquel $E(t) < \epsilon$ pour différentes valeurs de $\epsilon$), ou $t_\epsilon = \infty$ si le système ne converge pas (onde *stop-and-go* persistante).

Ces critères sont contradictoires : il n'existe pas de stratégie « meilleure » sur tous les plans. Vous présenterez les résultats sous forme de **front de Pareto** (par exemple $Q$ contre $D_{99}$, en indiquant les stratégies ayant provoqué des collisions) et discuterez les compromis.

### 3.3 Scénarios de simulation

Vous devrez utiliser au moins chacun des scénarios suivants (Table 1). Les résultats stochastiques (S1 et S2) seront moyennés sur au moins 20 réalisations indépendantes (graines aléatoires différentes), avec intervalles de confiance.

| Scénario | Circuit | Description |
| :--- | :--- | :--- |
| **S1** Anneau de Sugiyama | `circle`, $L = 230\text{ m}$ | $N = 22$, disposition `uniform`, perturbation de $\pm 1\text{ m}$ sur les positions initiales ; vitesses initiales nulles. |
| **S2** Balayage en densité | `stadium` | $N$ variant de 10 à la saturation du circuit, disposition `random` ; tracer $Q$, $C$, $D_{99}$ et $\overline{E}$ en fonction de $N$. |
| **S3** Freinage d'urgence | `stadium` | $N = 40$, partant de l'équilibre ; à $t = 200\text{ s}$, le véhicule 1 freine à $b_{max}$ pendant $3\text{ s}$, puis reprend sa stratégie. Mesurer $C$, $D_{max}$ et le temps de retour à l'équilibre. |

*Table 1 : Scénarios communs. Durée $t_{end} = 600\text{ s}$, fenêtre de mesure à partir de $t_w = 300\text{ s}$ (sauf S3).*

---

## 4. Extensions (au choix, valorisées dans l'évaluation)

* **Temps de réaction :** Remplacer (1) par une équation à retard :
  $$\dot{v}_n(t) = f(s_n(t - \tau), v_n(t - \tau), \Delta v_n(t - \tau))$$
  et étudier l'effet de $\tau$ sur les quatre critères.
* **Trafic mixte :** Mélanger plusieurs stratégies sur le même circuit.
* **Optimisation :** Paramétrer une famille de fonctions $f$ et rechercher automatiquement les paramètres situés sur le front de Pareto.
* **Courbure :** Tenir compte de la courbure du circuit pour la limite de la vitesse maximale autorisée.
* **Ajouts personnels :** Une idée nouvelle et originale sera récompensée d'un bonus.

---

## 5. Livrables et évaluation

1. **Code :** template complété, fonctions $f$, calcul des critères, scripts reproduisant toutes les figures du rapport, cellules Markdown complétant le code quand des explications sont nécessaires.
2. **Rapport (4 pages maximum, hors annexes [^1]) :** modélisation, résultats, discussion des compromis, limites. Le rapport doit suivre la structure classique d'un rapport scientifique. N'incluez ni page de garde, ni table des matières — ça n'est pas utile pour un rapport de cette longueur.
3. **Présentation orale potentielle (15 minutes + questions) :** incluant une animation d'au moins un scénario [^2]. Des groupes tirés au hasard pourront être convoqués pour présenter leur travail, en étant prévenus au moins 3 jours à l'avance. Aucun nouveau résultat ne pourra y être présenté.

| Élément évalué | Poids |
| :--- | :---: |
| Qualité, originalité et justification des stratégies proposées | 50 % |
| Analyse multicritère et discussion des compromis | 20 % |
| Qualité et reproductibilité du code | 15 % |
| Qualité de la documentation et du rapport | 15 % |

*Table 2 : Grille d'évaluation indicative. Une extension bien menée peut apporter un bonus.*

### Organisation
Vous formerez des groupes de **3 étudiant·e·s (exactement)** à renseigner avant le **9/11/2026** sur le Google Sheet suivant : [Lien cliquable](https://docs.google.com/spreadsheets/d/1j7AA5e-aEri3d8GHgyEDXvPSOVOck9SvdqWf5fwxvgo/edit?usp=sharing) [^3]. Si vous n'avez pas de groupe, complétez un groupe incomplet dans le tableau. 

Le projet dure un mois : 
* **Début :** 9/11/2026
* **Fin :** 9/12/2026 à 23h59

Pour toute question, vous pouvez contacter `pascal.tribel@ulb.be`.

---

[^1]: Les annexes ne doivent pas apporter d'information indispensable à la compréhension du travail ; elles doivent servir de complément en cas de besoin (par exemple, une démonstration, des résultats complémentaires, des résultats non concluants...).
[^2]: Pensez à `FuncAnimation`.
[^3]: Lien complet : `https://docs.google.com/spreadsheets/d/1j7AA5e-aEri3d8GHgyEDXvPSOVOck9SvdqWf5fwxvgo/edit?usp=sharing`