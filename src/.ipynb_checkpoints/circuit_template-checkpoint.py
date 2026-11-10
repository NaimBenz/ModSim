import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from scipy.interpolate import splev, splprep
import pandas as pd

LANE_WIDTH = 3.5      # largeur d'une voie [m]
MEDIAN_WIDTH = 3.0    # largeur du terre-plein central [m]
N_LANES = 2           # nombre de voies par sens
VEH_LENGTH = 5.0      # longueur d'un véhicule [m]
VEH_WIDTH = 1.9       # largeur d'un véhicule [m]

# demi-largeur totale de la route (de l'axe jusqu'au bord extérieur)
HALF_ROAD = MEDIAN_WIDTH / 2 + N_LANES * LANE_WIDTH


def lane_offset(direction, lane):
    """Décalage latéral (signé) du centre d'une voie par rapport à l'axe du circuit."""
    d = MEDIAN_WIDTH / 2 + (np.asarray(lane) + 0.5) * LANE_WIDTH
    return -np.asarray(direction) * d

class Circuit:
    """Courbe fermée paramétrée par l'abscisse curviligne s dans [0, L)."""
    def __init__(self, xy, smoothing=0.0, n_samples=4000):
        xy = np.asarray(xy, dtype=float)
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]

        tck, _ = splprep([xy[:, 0], xy[:, 1]], s=smoothing * len(xy), per=1)
        u = np.linspace(0, 1, n_samples, endpoint=False)
        dx, dy = splev(u, tck, der=1)
        s_of_u = np.concatenate([[0], np.cumsum(np.hypot(dx, dy)) / n_samples])
        self.L = s_of_u[-1]
        self.s_grid = np.linspace(0, self.L, n_samples, endpoint=False)
        u_grid = np.interp(self.s_grid, s_of_u, np.append(u, 1.0))
        x, y = splev(u_grid, tck)
        dx, dy = splev(u_grid, tck, der=1)
        ddx, ddy = splev(u_grid, tck, der=2)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.heading = np.unwrap(np.arctan2(dy, dx))
        self.kappa = (np.asarray(dx) * ddy - np.asarray(dy) * ddx) / np.hypot(dx, dy) ** 3

    def _lookup(self, table, s):
        """Valeur de 'table' en s (périodique : s et s+L sont le même point)."""
        return np.interp(np.mod(s, self.L), self.s_grid, table, period=self.L)

    def heading_at(self, s):
        """Angle de la tangente en s. On interpole cos et sin séparément
        pour éviter le saut entre +pi et -pi."""
        return np.arctan2(self._lookup(np.sin(self.heading), s), self._lookup(np.cos(self.heading), s))

    def curvature(self, s):
        """Courbure en s (> 0 si le circuit tourne à gauche)."""
        return self._lookup(self.kappa, s)

    def position(self, s, d=0.0):
        """Point (x, y) à l'abscisse s, décalé latéralement de d (gauche > 0)."""
        theta = self.heading_at(s)
        x = self._lookup(self.x, s) - d * np.sin(theta)
        y = self._lookup(self.y, s) + d * np.cos(theta)
        return x, y

    def lane_metric(self, s, d):
        """Longueur d'une voie décalée de d, par unité de longueur de l'axe : 1 - kappa*d."""
        return 1.0 - self.curvature(s) * d

    def min_radius(self):
        """Plus petit rayon de courbure du circuit."""
        return 1.0 / np.abs(self.kappa).max()

def make_circle(length=600.0):
    R = length / (2 * np.pi)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    points = np.c_[R * np.cos(t), R * np.sin(t)]
    return Circuit(points)


def make_stadium(straight=300.0, radius=60.0):
    t = np.linspace(-np.pi / 2, np.pi / 2, 40, endpoint=False)
    xs = np.linspace(-straight / 2, straight / 2, 40, endpoint=False)
    bas = np.c_[xs, -radius * np.ones_like(xs)]
    droite = np.c_[straight / 2 + radius * np.cos(t), radius * np.sin(t)]
    haut = np.c_[-xs, radius * np.ones_like(xs)]
    gauche = np.c_[-straight / 2 - radius * np.cos(t), -radius * np.sin(t)]
    return Circuit(np.vstack([bas, droite, haut, gauche]))


def load_csv(path, smoothing=1.0):
    xy = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))
    return Circuit(xy, smoothing)

def place_vehicles(circuit, n_per_lane, layout="uniform", min_gap=2.0, seed=0):
    rng = np.random.default_rng(seed)
    L = circuit.L
    pas = VEH_LENGTH + min_gap

    if n_per_lane * pas > L:
        raise ValueError("Trop de véhicules pour ce circuit")

    s_all, lane_all, dir_all = [], [], []

    for direction in (+1, -1):
        for lane in range(N_LANES):
            if layout == "uniform":
                s = np.linspace(0, L, n_per_lane, endpoint=False)
                s = s + 0.5 * lane * L / n_per_lane
            elif layout == "random":
                espace_libre = L - n_per_lane * pas
                coupures = np.sort(rng.uniform(0, espace_libre, n_per_lane))
                s = coupures + np.arange(n_per_lane) * pas
                s = s + rng.uniform(0, L)
            elif layout == "platoon":
                s = np.arange(n_per_lane) * pas
            else:
                raise ValueError(f"Layout inconnu : {layout}")
            s_all.append(np.mod(s, L))
            lane_all.append(np.full(n_per_lane, lane))
            dir_all.append(np.full(n_per_lane, direction))
    s = np.concatenate(s_all)
    return {"s": s, "v": np.zeros_like(s), "lane": np.concatenate(lane_all), "direction": np.concatenate(dir_all)}

def neighbours(veh, L, lane=None):
    s, direction = veh["s"], veh["direction"]
    N = len(s)
    lane = veh["lane"] if lane is None else np.asarray(lane)
    dist_avant = np.mod(direction[:, None] * (s[None, :] - s[:, None]), L)
    candidat = (direction[:, None] == direction[None, :]) & (veh["lane"][None, :] == lane[:, None])
    candidat[np.arange(N), np.arange(N)] = False
    devant = np.where(candidat, dist_avant, np.inf)
    derriere = np.where(candidat, np.mod(-dist_avant, L), np.inf)
    leader = devant.argmin(axis=1)
    follower = derriere.argmin(axis=1)
    gap_front = devant.min(axis=1) - VEH_LENGTH
    gap_back = derriere.min(axis=1) - VEH_LENGTH
    vide = ~candidat.any(axis=1)
    leader[vide] = -1
    follower[vide] = -1
    return leader, gap_front, follower, gap_back

def vehicle_colors(veh, color_by, v_max):
    if color_by == "speed":
        norm = plt.Normalize(0, v_max)
        return plt.cm.RdYlGn(norm(veh["v"])), norm
    if color_by == "lane":
        orange, bleu = [0.9, 0.5, 0.1, 1], [0.2, 0.5, 0.9, 1]
        return np.where((veh["lane"] == 0)[:, None], [orange], [bleu]), None
    rouge, bleu = [0.85, 0.2, 0.2, 1], [0.2, 0.3, 0.85, 1]
    return np.where((veh["direction"] > 0)[:, None], [rouge], [bleu]), None

def band_path(circuit, d_in, d_out, k):
    """Anneau fermé entre les courbes décalées de d_in*k et d_out*k."""
    s = circuit.s_grid
    x_ext, y_ext = circuit.position(s, d_out * k)
    x_int, y_int = circuit.position(s, d_in * k)
    verts = np.vstack([np.c_[x_ext, y_ext], [[x_ext[0], y_ext[0]]], np.c_[x_int[::-1], y_int[::-1]], [[x_int[-1], y_int[-1]]]])
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(s) - 1) + [Path.CLOSEPOLY]) * 2
    return Path(verts, codes)


def draw_road(ax, circuit, k):
    s = circuit.s_grid
    for sign in (+1, -1):
        bande = band_path(circuit, sign * MEDIAN_WIDTH / 2, sign * HALF_ROAD, k)
        ax.add_patch(PathPatch(bande, fc="0.55", ec="none", zorder=0))
        ax.plot(*circuit.position(s, sign * HALF_ROAD * k), "w-", lw=1, zorder=1)
        ax.plot(*circuit.position(s, sign * MEDIAN_WIDTH / 2 * k), "w-", lw=1, zorder=1)
        for lane in range(1, N_LANES):
            d = sign * (MEDIAN_WIDTH / 2 + lane * LANE_WIDTH) * k
            ax.plot(*circuit.position(s, d), color="w", ls=(0, (4, 6)), lw=0.8, zorder=1)
    median = band_path(circuit, -MEDIAN_WIDTH / 2, MEDIAN_WIDTH / 2, k)
    ax.add_patch(PathPatch(median, fc="#6a9f58", ec="none", zorder=0))
    arrow_len = min(0.04 * circuit.L, 2.5 * HALF_ROAD * k)
    for direction in (+1, -1):
        d = lane_offset(direction, 0.5) * k
        x0, y0 = circuit.position(0.0, d)
        x1, y1 = circuit.position(direction * arrow_len, d)
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="k", lw=1.5), zorder=5)

def draw_vehicles(ax, circuit, veh, colors, k, label_cars=False):
    s, direction, lane = veh["s"], veh["direction"], veh["lane"]
    s_centre = s - direction * VEH_LENGTH / 2
    xc, yc = circuit.position(s_centre, lane_offset(direction, lane) * k)
    theta = circuit.heading_at(s_centre) + np.where(direction > 0, 0.0, np.pi)
    zoom_veh = max(1.0, k * 0.8)
    coins = 0.5 * zoom_veh * np.array([[-VEH_LENGTH, -VEH_WIDTH],
                                       [ VEH_LENGTH, -VEH_WIDTH],
                                       [ VEH_LENGTH,  VEH_WIDTH],
                                       [-VEH_LENGTH,  VEH_WIDTH]])

    for i in range(len(s)):
        R = np.array([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]),  np.cos(theta[i])]])
        coins_i = coins @ R.T + [xc[i], yc[i]]
        ax.add_patch(Polygon(coins_i, closed=True, fc=colors[i], ec="k", lw=0.5, zorder=3))
        if label_cars:
            ax.annotate(str(i), (xc[i], yc[i]), fontsize=6, ha="center", va="center", zorder=4)

def finish_plot(ax, circuit, veh, color_by, norm, title):
    if color_by == "speed":
        if not hasattr(ax, "_cbar"):
            ax._cbar = ax.figure.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm), ax=ax, shrink=0.5, label="vitesse [m/s]")
    elif color_by == "direction":
        ax.plot([], [], "s", color=[0.85, 0.2, 0.2], mec="k", label="sens +1")
        ax.plot([], [], "s", color=[0.2, 0.3, 0.85], mec="k", label="sens -1")
        ax.legend(loc="upper right", fontsize=8)
    elif color_by == "lane":
        ax.plot([], [], "s", color=[0.9, 0.5, 0.1], mec="k", label="voie 0 (gauche)")
        ax.plot([], [], "s", color=[0.2, 0.5, 0.9], mec="k", label="voie 1 (droite)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"L = {circuit.L:.0f} m, {len(veh['s'])} véhicules, 2 x {N_LANES} voies")


def plot_schematic(circuit, veh, color_by, v_max, ax, title):
    ax.plot(np.append(circuit.x, circuit.x[0]), np.append(circuit.y, circuit.y[0]), color="0.6", lw=7, solid_capstyle="round", zorder=0)
    ax.plot(circuit.x, circuit.y, color="#6a9f58", lw=1, zorder=1)
    s, direction = veh["s"], veh["direction"]
    x, y = circuit.position(s, lane_offset(direction, veh["lane"]))
    angle = np.degrees(circuit.heading_at(s)) + np.where(direction > 0, 0, 180) - 90
    colors, norm = vehicle_colors(veh, color_by, v_max)
    for i in range(len(s)):
        ax.plot(x[i], y[i], marker=(3, 0, angle[i]), ms=7, color=colors[i], mec="k", mew=0.5, zorder=3)
    finish_plot(ax, circuit, veh, color_by, norm, title)
    ax.text(0.01, 0.01, "vue schématique (voies pas à l'échelle) : utiliser zoom pour les voir", transform=ax.transAxes, fontsize=8, color="0.4")
    return ax

def plot_state(circuit, veh, color_by="direction", v_max=30.0, ax=None, title="", label_cars=False, zoom=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 12))
    ax.clear()
    taille = max(np.ptp(circuit.x), np.ptp(circuit.y))
    k_voulu = taille / 250.0
    k_max = 0.8 * circuit.min_radius() / HALF_ROAD
    if zoom:
        k, schematique = 1.0, False
    else:
        k = max(1.0, min(k_voulu, k_max))
        schematique = k_voulu > 2 * k
    if schematique:
        return plot_schematic(circuit, veh, color_by, v_max, ax, title)
    draw_road(ax, circuit, k)
    colors, norm = vehicle_colors(veh, color_by, v_max)
    draw_vehicles(ax, circuit, veh, colors, k, label_cars)
    finish_plot(ax, circuit, veh, color_by, norm, title)
    if k > 1:
        ax.text(0.01, 0.01, f"largeur de la route dessinée x{k:.1f} (visuel uniquement)", transform=ax.transAxes, fontsize=8, color="0.4")
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.autoscale_view()
    return ax


def zoom_box(circuit, s, half_width=60.0):
    """Fenêtre carrée (x0, x1, y0, y1) centrée sur l'abscisse s."""
    x, y = circuit.position(s)
    return (x - half_width, x + half_width, y - half_width, y + half_width)

def print_neighbours_table(veh, L):
    """Tableau des voisins de chaque véhicule (voie propre et voie adjacente)."""
    leader, gap, follower, gap_back = neighbours(veh, L)
    leader_adj, gap_adj, _, _ = neighbours(veh, L, lane=1 - veh["lane"])

    df = pd.DataFrame({
        "sens": veh["direction"],
        "voie": veh["lane"],
        "s [m]": veh["s"],
        "v [m/s]": veh["v"],
        "leader": leader,
        "gap avant [m]": gap,
        "suiveur": follower,
        "gap arrière [m]": gap_back,
        "leader autre voie": leader_adj,
        "gap autre voie [m]": gap_adj,
    })
    df.index.name = "i"

    df["_pos"] = df["sens"] * df["s [m]"]
    df = df.sort_values(["sens", "voie", "_pos"], ascending=[False, True, True]).drop(columns="_pos")
    df = df.replace(np.inf, np.nan)
    cols_gap = ["gap avant [m]", "gap arrière [m]", "gap autre voie [m]"]
    return (df.style.format({"s [m]": "{:.1f}", "v [m/s]": "{:.1f}", **{c: "{:.1f}" for c in cols_gap}}, na_rep="—")
    .background_gradient(subset=cols_gap, cmap="RdYlGn", vmin=0, vmax=60).set_caption("Voisins de chaque véhicule (gap = pare-chocs à pare-chocs)"))