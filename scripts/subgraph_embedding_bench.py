"""Banco de pruebas de embeddings para búsqueda de subgrafos similares.

Compara tres técnicas de representación de subgrafos, todas sin ningún dato de
crimen (respeta la invariante de PROJECT_SPEC §5.1):

  1. graph2vec-fusionado  -- la de producción (WL + PV-DBOW + estructural + jerarquía).
  2. WWL                   -- Wasserstein Weisfeiler-Lehman graph kernel.
                             Togninalli, Ghisu, Llinares-Lopez, Rieck, Borgwardt,
                             "Wasserstein Weisfeiler-Lehman Graph Kernels", NeurIPS 2019
                             (arXiv:1906.01277).
  3. scattering            -- Geometric Scattering Transform sobre grafos.
                             Gao, Wolf, Hirn, "Geometric Scattering for Graph Data
                             Analysis", ICML 2019, PMLR 97:2122-2131.

Para 10 arquetipos de subgrafo (una fila cada uno) calcula, por tecnica, el
top-5 de subgrafos mas parecidos separado en dos matrices: los que comparten
lugar con el objetivo (>=1 nodo en comun) y los que estan en otro sitio
(0 nodos en comun, el criterio SAME_PLACE_JACCARD del pipeline).

Salidas en reports/subgraph_embedding/:
  - <tecnica>.png            figura con las dos matrices (mismo lugar / otro lugar)
  - all_techniques.png       las tres tecnicas juntas
  - tables.json              los datos crudos de cada celda
  - README.md                las tablas 10x6 en texto
"""

from __future__ import annotations

import ast
import json
import logging
import math
from pathlib import Path

import numpy as np
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bench")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "subgraph_embedding"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
WL_ITERS = 2
HIER = {
    "expressway": {"motorway", "motorway_link", "trunk", "trunk_link"},
    "avenue": {"primary", "primary_link"},
    "collector": {"secondary", "secondary_link", "tertiary", "tertiary_link"},
    "street": {"residential", "living_street", "unclassified", "road"},
    "service": {"service", "track", "busway"},
}
HIER_ORDER = ["expressway", "avenue", "collector", "street", "service"]


# --------------------------------------------------------------------------- #
# Carga y reconstruccion de los subgrafos
# --------------------------------------------------------------------------- #
def fold_highway(raw) -> str:
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            pass
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    raw = str(raw)
    for cls, members in HIER.items():
        if raw in members:
            return cls
    return "street"


def _parse_linestring(s):
    """'LINESTRING (lon lat, lon lat, ...)' -> (N,2) array de lon/lat."""
    try:
        inner = s[s.index("(") + 1:s.rindex(")")]
        return np.array([[float(p[0]), float(p[1])]
                         for p in (q.strip().split() for q in inner.split(","))
                         if len(p) == 2], float)
    except Exception:
        return None


def load_corpus():
    hs = json.loads((ROOT / "data/interim/hotspots.json").read_text())["hotspots"]
    sim = json.loads((ROOT / "data/processed/similarity.json").read_text())
    struct_by_id = {h["id"]: np.asarray(h["structural"], float) for h in sim["hotspots"]}
    hier_by_id = {h["id"]: np.asarray(h["hierarchy"], float) for h in sim["hotspots"]}
    struct_dims = sim["dims"]["structural"]

    log.info("Leyendo la red vial (29.832 nodos)...")
    G = nx.read_graphml(ROOT / "data/interim/network.graphml")
    xy = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}   # (lon, lat)

    lat0 = float(np.mean([xy[n][1] for n in xy]))
    kx = math.cos(math.radians(lat0)) * 111_320.0
    ky = 110_540.0

    # --- mapa de la ciudad: aristas de toda la red para el fondo de calles ---
    edge_geom, edge_cls = {}, {}
    eu, ev = [], []
    seen = set()
    for a, b, d in G.edges(data=True):
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        eu.append(xy[a]); ev.append(xy[b])
        cls = fold_highway(d.get("highway"))
        edge_cls[key] = cls
        g_ = _parse_linestring(d["geometry"]) if "geometry" in d else None
        edge_geom[key] = g_
    netmap = {
        "lat0": lat0,
        "EU": np.array(eu, float), "EV": np.array(ev, float),          # (M,2) lon/lat
        "NX": np.array([xy[n][0] for n in G.nodes], float),
        "NY": np.array([xy[n][1] for n in G.nodes], float),
        "geom": edge_geom, "cls": edge_cls,
    }

    def edge_lookup(u, v):
        key = (u, v) if u < v else (v, u)
        data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        length = 0.0
        if data:
            d0 = next(iter(data.values()))
            length = float(d0.get("length", 0.0) or 0.0)
        return edge_cls.get(key, "street"), length, edge_geom.get(key)

    subs = []
    for h in hs:
        nodes = [str(n) for n in h["nodes"]]
        edges = [(str(a), str(b)) for a, b in h["edges"]]
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for a, b in edges:
            cls, length, geom = edge_lookup(a, b)
            if a in xy and b in xy:
                (ax, ay), (bx, by) = xy[a], xy[b]
                length = length or math.hypot((ax - bx) * kx, (ay - by) * ky)
            g.add_edge(a, b, cls=cls, length=length or 1.0, geom=geom)
        # coordenadas locales en metros, centradas (para los descriptores)
        pts = np.array([[xy[n][0] * kx, xy[n][1] * ky] for n in g.nodes if n in xy], float)
        if len(pts):
            pts = pts - pts.mean(0)
        coord, geo = {}, {}
        i = 0
        for n in g.nodes:
            geo[n] = np.array(xy[n], float) if n in xy else np.zeros(2)
            if n in xy:
                coord[n] = pts[i]; i += 1
            else:
                coord[n] = np.zeros(2)
        lons = [xy[n][0] for n in g.nodes if n in xy]
        lats = [xy[n][1] for n in g.nodes if n in xy]
        subs.append({
            "id": h["id"], "month": h["month"], "G": g,
            "nodes": set(nodes), "coord": coord, "geo": geo,
            "centroid": (float(np.mean(lats)), float(np.mean(lons))) if lons else (0.0, 0.0),
            "n_nodes": h["n_nodes"], "n_edges": len(h["edges"]),
            "structural": struct_by_id[h["id"]],
            "hierarchy": hier_by_id[h["id"]],
        })
    return subs, struct_dims, netmap


# --------------------------------------------------------------------------- #
# Tecnica 1: graph2vec fusionado (produccion)
# --------------------------------------------------------------------------- #
def emb_graph2vec(subs):
    import sys
    sys.path.insert(0, str(ROOT))
    from pipeline.features.embedding import Graph2Vec

    graphs = [s["G"] for s in subs]
    g2v = Graph2Vec(dimensions=128, wl_iterations=WL_ITERS, epochs=50,
                    random_state=RANDOM_STATE).fit_transform(graphs)

    def z(X):
        mu, sd = X.mean(0), X.std(0)
        flat = sd <= 1e-12
        sd = np.where(flat, 1.0, sd)
        Y = (X - mu) / sd
        Y[:, flat] = 0.0
        return Y

    struct = np.vstack([s["structural"] for s in subs])
    hier = np.vstack([s["hierarchy"] for s in subs])
    parts = [
        z(struct) * (1.0 / math.sqrt(struct.shape[1])),
        z(g2v) * (1.0 / math.sqrt(g2v.shape[1])),
        z(hier) * (0.5 / math.sqrt(hier.shape[1])),
    ]
    fused = np.hstack(parts)
    return _cosine(fused)


# --------------------------------------------------------------------------- #
# Tecnica 2: Wasserstein Weisfeiler-Lehman  (NeurIPS 2019, arXiv:1906.01277)
# --------------------------------------------------------------------------- #
def _wwl_node_features(s):
    """Rasgos continuos por nodo, refinados con h pasadas WL de promediado.

    El esquema WL de graph2vec compara *histogramas* de etiquetas: dos motivos
    casi iguales caen en cubos distintos y su parecido se pierde en la
    agregacion. WWL mantiene una nube de vectores por nodo y mide el coste de
    transporte optimo (distancia de Wasserstein) entre las dos nubes, que si es
    sensible a diferencias finas (Togninalli et al., 2019, secc. 3).
    """
    g = s["G"]
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    deg = np.array([g.degree(v) for v in nodes], float)
    clus = np.array([nx.clustering(g, v) for v in nodes], float)
    cent = np.vstack([s["coord"][v] for v in nodes])
    rg = np.sqrt((cent ** 2).sum(1).mean()) or 1.0
    dist_c = np.linalg.norm(cent, axis=1) / rg
    road = np.zeros((n, len(HIER_ORDER)))
    elen = np.zeros(n)
    for v in nodes:
        ls = []
        for u in g[v]:
            c = g[v][u]["cls"]
            road[idx[v], HIER_ORDER.index(c)] += 1.0
            ls.append(g[v][u]["length"])
        if ls:
            road[idx[v]] /= road[idx[v]].sum()
            elen[idx[v]] = np.log1p(np.mean(ls))
    base = np.column_stack([
        np.log1p(deg), clus, dist_c,
        (elen - elen.mean()) / (elen.std() or 1.0), road,
    ])
    # propagacion WL: cada iteracion anexa el promedio con los vecinos
    A = nx.to_numpy_array(g, nodelist=nodes)
    Dinv = np.diag(1.0 / np.clip(A.sum(1), 1.0, None))
    P = Dinv @ A
    feats = [base]
    cur = base
    for _ in range(WL_ITERS):
        cur = 0.5 * cur + 0.5 * (P @ cur)
        feats.append(cur)
    return np.hstack(feats)


def _sinkhorn_w1(C, eps=0.05, iters=200):
    """Distancia de Wasserstein-1 (entropica) entre dos nubes con masa uniforme."""
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    K = np.exp(-C / eps)
    u = np.ones(n)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    Pi = u[:, None] * K * v[None, :]
    return float((Pi * C).sum())


def emb_wwl(subs, targets_idx):
    feats = [_wwl_node_features(s) for s in subs]
    # escala global por columna para que ninguna dimension domine el coste
    allf = np.vstack(feats)
    sd = allf.std(0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    feats = [f / sd for f in feats]

    n = len(subs)
    D = np.full((n, n), np.nan)
    log.info("WWL: %d objetivos x %d subgrafos (transporte optimo)...",
             len(targets_idx), n)
    for t in targets_idx:
        ft = feats[t]
        for j in range(n):
            if j == t:
                D[t, j] = 0.0
                continue
            fj = feats[j]
            C = np.linalg.norm(ft[:, None, :] - fj[None, :, :], axis=2)
            D[t, j] = _sinkhorn_w1(C)
    # kernel exp -> parecido en (0, 1]
    finite = D[np.isfinite(D) & (D > 0)]
    gamma = 1.0 / np.median(finite)
    S = np.exp(-gamma * D)
    return S


# --------------------------------------------------------------------------- #
# Tecnica 3: Geometric Scattering Transform  (ICML 2019, PMLR 97:2122)
# --------------------------------------------------------------------------- #
def _scattering_vector(s, J=4, moments=(1, 2, 3, 4)):
    """Cascada de wavelets de difusion + modulo + momentos estadisticos.

    P = 1/2 (I + A D^-1) es el random walk perezoso; los wavelets
    Psi_j = P^(2^(j-1)) - P^(2^j) leen la escala 2^j. Se aplican a varias
    senales de nodo, se toma |.| y se resume el grafo con momentos, lo que da
    un vector de longitud fija e invariante a la numeracion (Gao et al., 2019).
    """
    g = s["G"]
    nodes = list(g.nodes)
    n = len(nodes)
    A = nx.to_numpy_array(g, nodelist=nodes)
    d = np.clip(A.sum(1), 1.0, None)
    P = 0.5 * (np.eye(n) + A / d[None, :])   # 1/2 (I + A D^-1)

    Ppow = [np.eye(n), P]                 # Ppow[0]=I, Ppow[1]=P^(2^0)
    for _ in range(J):
        Ppow.append(Ppow[-1] @ Ppow[-1])  # Ppow[k] = P^(2^(k-1))
    psis = [Ppow[k] - Ppow[k + 1] for k in range(1, J + 1)]
    lowpass = Ppow[J + 1]

    deg = A.sum(1)
    clus = np.array([nx.clustering(g, v) for v in nodes], float)
    cent = np.vstack([s["coord"][v] for v in nodes])
    rg = np.sqrt((cent ** 2).sum(1).mean()) or 1.0
    signals = [
        deg / (deg.max() or 1.0),
        clus,
        np.linalg.norm(cent, axis=1) / rg,
        cent[:, 0] / rg, cent[:, 1] / rg,
    ]

    def mom(x):
        return [float(np.sum(np.abs(x) ** q)) for q in moments]

    out = []
    for x in signals:
        out += mom(lowpass @ x)                      # orden 0 (paso bajo)
        w1 = [psi @ x for psi in psis]
        for j in range(J):
            out += mom(np.abs(w1[j]))                # orden 1
            for jp in range(j + 1, J):
                out += mom(np.abs(psis[jp] @ np.abs(w1[j])))   # orden 2
    return np.asarray(out, float)


def emb_scattering(subs):
    X = np.vstack([_scattering_vector(s) for s in subs])
    X = np.log1p(X)                                  # los momentos crecen mucho
    mu, sd = X.mean(0), X.std(0)
    X = (X - mu) / np.where(sd <= 1e-12, 1.0, sd)
    return _cosine(X)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _cosine(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    U = X / nrm
    S = np.clip(U @ U.T, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


# --------------------------------------------------------------------------- #
# 10 arquetipos de subgrafo
# --------------------------------------------------------------------------- #
def pick_archetypes(subs, struct_dims):
    di = {name: k for k, name in enumerate(struct_dims)}
    md, e1, e2, e3, e4p = di["mean_degree"], di["deg_frac_1"], di["deg_frac_2"], di["deg_frac_3"], di["deg_frac_4plus"]
    elong, dens = di["elongation"], di["density"]

    # nº de subgrafos que comparten al menos un nodo (recurrencia del mismo sitio)
    same_cnt = [0] * len(subs)
    for i in range(len(subs)):
        for j in range(len(subs)):
            if i != j and subs[i]["nodes"] & subs[j]["nodes"]:
                same_cnt[i] += 1

    rows = []
    for i, s in enumerate(subs):
        v = s["structural"]
        h = s["hierarchy"]
        rows.append(dict(
            i=i, id=s["id"], nn=s["n_nodes"], ne=s["n_edges"], same=same_cnt[i],
            md=v[md], f1=v[e1], f2=v[e2], f3=v[e3], f4=v[e4p],
            elong=v[elong], dens=v[dens],
            dom_share=h[5], f_expw=h[0], f_ave=h[1], f_coll=h[2], f_str=h[3], f_srv=h[4],
            tree=(s["n_edges"] == s["n_nodes"] - 1),
            cyc=(abs(s["n_edges"] - s["n_nodes"]) <= 1),
        ))

    def choose(name, pred, key):
        cand = [r for r in rows if pred(r)]
        if not cand:
            return None
        cand.sort(key=key)
        return cand[0]

    specs = [
        ("Diada lineal (2 nodos)",
         lambda r: r["nn"] == 2, lambda r: -r["ne"]),
        ("Cadena corta (3-4 nodos)",
         lambda r: 3 <= r["nn"] <= 4 and r["tree"] and r["elong"] < 0.25,
         lambda r: r["elong"]),
        ("Corredor / avenida (camino largo, elongacion baja)",
         lambda r: r["nn"] >= 9 and r["ne"] <= r["nn"] + 1 and r["elong"] < 0.30,
         lambda r: (r["elong"], -r["nn"])),
        ("Cruce en T / estrella (grado-1 dominante)",
         lambda r: r["f1"] >= 0.45 and r["md"] < 0.45 and 4 <= r["nn"] <= 10,
         lambda r: -r["f1"]),
        ("Anillo / ciclo (grado-2 dominante)",
         lambda r: r["f2"] >= 0.75 and r["cyc"] and r["nn"] >= 6,
         lambda r: -r["f2"]),
        ("Arbol ramificado (aristas = nodos-1, ramas)",
         lambda r: r["tree"] and (r["f3"] + r["f4"]) >= 0.18 and r["nn"] >= 7,
         lambda r: -(r["f3"] + r["f4"])),
        ("Damero compacto (denso, elongacion alta)",
         lambda r: 8 <= r["nn"] <= 28 and r["dens"] >= 0.10 and r["md"] >= 0.45 and r["elong"] >= 0.55,
         lambda r: (-r["dens"], -r["elong"])),
        ("Rejilla grande (>=45 nodos)",
         lambda r: r["nn"] >= 45 and r["md"] >= 0.45,
         lambda r: (-r["nn"], -r["md"])),
        ("Peine / cul-de-sac (espina + dientes)",
         lambda r: r["f1"] >= 0.30 and r["f3"] >= 0.18 and r["nn"] >= 8,
         lambda r: -(r["f1"] * r["f3"])),
        ("Hibrido arterial (clases viales mezcladas)",
         lambda r: r["dom_share"] <= 0.62 and (r["f_expw"] > 0 or r["f_ave"] > 0) and r["nn"] >= 8,
         lambda r: r["dom_share"]),
    ]

    chosen, used = [], set()
    for name, pred, key in specs:
        # se prefiere un objetivo con recurrencia del mismo sitio (>=5, luego >=1)
        # para que ambas matrices salgan pobladas; si no lo hay, se relaja.
        r = (choose(name, lambda x: pred(x) and x["id"] not in used and x["same"] >= 5, key)
             or choose(name, lambda x: pred(x) and x["id"] not in used and x["same"] >= 1, key)
             or choose(name, lambda x: pred(x) and x["id"] not in used, key)
             or choose(name, pred, key))         # permite repetir antes que faltar
        if r is None:                            # ultra-fallback: el mas raro libre
            r = next((x for x in rows if x["id"] not in used), rows[0])
        used.add(r["id"])
        chosen.append((name, r["i"], r["id"]))
        log.info("  %-52s -> %s  (n=%d, m=%d)", name, r["id"], r["nn"], r["ne"])
    return chosen


# --------------------------------------------------------------------------- #
# Top-5: general (con mismo lugar resaltado) y solo-otro-lugar
# --------------------------------------------------------------------------- #
def top5_split(S, subs, ti):
    """Devuelve (general, other):

    - general : top-5 sin filtrar; los que comparten sitio con el objetivo
                (>=1 nodo) quedan marcados con shared_nodes>0 para resaltarlos.
    - other   : top-5 excluyendo el mismo lugar (0 nodos en comun), la lista
                que responde la pregunta de la tesis (comparar zonas distintas).
    """
    order = np.argsort(-S[ti])
    general, other = [], []
    tnodes = subs[ti]["nodes"]
    for j in order:
        if j == ti:
            continue
        shared = len(tnodes & subs[j]["nodes"])
        rec = {"id": subs[j]["id"], "month": subs[j]["month"],
               "score": round(float(S[ti, j]), 4), "shared_nodes": shared}
        if len(general) < 5:
            general.append(rec)
        if shared == 0 and len(other) < 5:
            other.append(rec)
        if len(general) == 5 and len(other) == 5:
            break
    return general, other


# --------------------------------------------------------------------------- #
# Dibujo sobre el mapa real  (calidad de figura de paper)
# --------------------------------------------------------------------------- #
#: grosor de trazo por clase vial
_LW = {"expressway": 3.0, "avenue": 2.2, "collector": 1.5, "street": 1.05, "service": 0.7}

#: nombres cortos para las etiquetas de fila de la figura
SHORT = {
    "Diada lineal (2 nodos)": "Díada\n(2 nodos)",
    "Cadena corta (3-4 nodos)": "Cadena\ncorta",
    "Corredor / avenida (camino largo, elongacion baja)": "Corredor /\navenida",
    "Cruce en T / estrella (grado-1 dominante)": "Cruce en T /\nestrella",
    "Anillo / ciclo (grado-2 dominante)": "Anillo /\nciclo",
    "Arbol ramificado (aristas = nodos-1, ramas)": "Árbol\nramificado",
    "Damero compacto (denso, elongacion alta)": "Damero\ncompacto",
    "Rejilla grande (>=45 nodos)": "Rejilla\ngrande",
    "Peine / cul-de-sac (espina + dientes)": "Peine /\ncul-de-sac",
    "Hibrido arterial (clases viales mezcladas)": "Híbrido\narterial",
}
_INK = "#12355b"        # tinta del subgrafo
_ACCENT = "#e8730c"     # nodos compartidos / marca de mismo lugar
_STREET = "#d3d7de"     # calles del entorno


def _paper_rc():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": "#1a1a1a",
        "axes.edgecolor": "#b9bec7", "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300,
    })


def _edge_polyline(sub, u, v):
    g = sub["G"][u][v].get("geom")
    if g is not None and len(g) >= 2:
        return g
    return np.vstack([sub["geo"][u], sub["geo"][v]])


def _span_m(sub):
    lons = [sub["geo"][n][0] for n in sub["G"].nodes]
    lats = [sub["geo"][n][1] for n in sub["G"].nodes]
    clat = sub["centroid"][0]
    dx = (max(lons) - min(lons)) * 111_320.0 * math.cos(math.radians(clat))
    dy = (max(lats) - min(lats)) * 110_540.0
    return max(dx, dy, 1.0)


def _scalebar(ax, extent_m, clat):
    for cand in (2000, 1000, 500, 200, 100, 50, 25):
        if cand <= extent_m * 0.42:
            L = cand
            break
    else:
        L = 25
    mlon = 1.0 / (111_320.0 * math.cos(math.radians(clat)))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px, py = x0 + (x1 - x0) * 0.07, y0 + (y1 - y0) * 0.085
    ax.plot([px, px + L * mlon], [py, py], "-", color="#1a1a1a", lw=1.6,
            solid_capstyle="butt", zorder=6)
    ax.text(px + L * mlon / 2, py + (y1 - y0) * 0.03,
            f"{L} m" if L < 1000 else f"{L // 1000} km",
            ha="center", va="bottom", fontsize=5.6, zorder=6)


def _draw_panel(ax, sub, netmap, extent_m, *, highlight=None, same=False, score=None):
    from matplotlib.collections import LineCollection

    hi = highlight or set()
    clat, clon = sub["centroid"]
    half_lat = 0.5 * extent_m / 110_540.0
    half_lon = 0.5 * extent_m / (111_320.0 * math.cos(math.radians(clat)))
    x0, x1 = clon - half_lon, clon + half_lon
    y0, y1 = clat - half_lat, clat + half_lat

    EU, EV = netmap["EU"], netmap["EV"]
    inbox = (((EU[:, 0] >= x0) & (EU[:, 0] <= x1) & (EU[:, 1] >= y0) & (EU[:, 1] <= y1))
             | ((EV[:, 0] >= x0) & (EV[:, 0] <= x1) & (EV[:, 1] >= y0) & (EV[:, 1] <= y1)))
    ax.add_collection(LineCollection(np.stack([EU[inbox], EV[inbox]], axis=1),
                                     colors=_STREET, linewidths=0.6, zorder=0))

    n = sub["n_nodes"]
    lwf = 0.45 if n > 80 else (0.65 if n > 30 else 1.0)
    for u, v, d in sub["G"].edges(data=True):
        pl = _edge_polyline(sub, u, v)
        ax.plot(pl[:, 0], pl[:, 1], "-", color=_INK,
                lw=_LW.get(d["cls"], 1.05) * lwf, solid_capstyle="round", zorder=2)
    ns = 4 if n > 80 else (7 if n > 30 else 11)
    base = [x for x in sub["G"].nodes if x not in hi]
    ax.scatter([sub["geo"][x][0] for x in base], [sub["geo"][x][1] for x in base],
               s=ns, c=_INK, zorder=3, linewidths=0)
    if hi:
        ax.scatter([sub["geo"][x][0] for x in hi], [sub["geo"][x][1] for x in hi],
                   s=ns + 26, c=_ACCENT, edgecolors="white", linewidths=0.6, zorder=4)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / math.cos(math.radians(clat)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{clat:.4f}°N, {abs(clon):.4f}°W", fontsize=5.8, labelpad=2)
    if same:
        ax.set_facecolor("#fff5ea")
        for sp in ax.spines.values():
            sp.set_edgecolor(_ACCENT); sp.set_linewidth(1.5)
    if score is not None:
        ax.text(0.05, 0.955, f"s = {score:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7, fontweight="bold",
                color=(_ACCENT if same else _INK),
                bbox=dict(boxstyle="round,pad=0.22",
                          fc=("#fff5ea" if same else "white"),
                          ec=(_ACCENT if same else "#b9bec7"),
                          lw=(0.9 if same else 0.5), alpha=0.95))
    _scalebar(ax, extent_m, clat)


def plot_gallery(disp, split, split_label, rows, table, by_id, netmap, stem,
                 *, panel=1.75):
    """Figura de paper: fila = arquetipo, col 1 = objetivo, col 2-6 = top-5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_rc()

    nrows, ncols = len(rows), 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols + 0.8, panel * nrows + 0.9),
                             squeeze=False)
    heads = ["Objetivo", "Similar #1", "#2", "#3", "#4", "#5"]
    for r, (name, ti, tid) in enumerate(rows):
        tsub = by_id[tid]
        recs = table[tid][split]
        match_spans = [_span_m(by_id[x["id"]]) for x in recs] or [0.0]
        ext = max(200.0, 1.4 * max(_span_m(tsub), float(np.median(match_spans))))
        _draw_panel(axes[r][0], tsub, netmap, ext)
        axes[r][0].set_ylabel(SHORT.get(name, name), rotation=0, ha="right", va="center",
                              fontsize=9, fontweight="bold", labelpad=10, color="#1a1a1a")
        axes[r][0].annotate(f"{tid} · n={tsub['n_nodes']}", xy=(0.5, 1.01),
                            xycoords="axes fraction", ha="center", va="bottom", fontsize=6)
        for c in range(1, ncols):
            ax = axes[r][c]
            if c - 1 >= len(recs):
                ax.axis("off"); continue
            rec = recs[c - 1]
            msub = by_id[rec["id"]]
            same = rec["shared_nodes"] > 0
            shared = tsub["nodes"] & msub["nodes"] if same else set()
            _draw_panel(ax, msub, netmap, ext, highlight=shared, same=same,
                        score=rec["score"])
            ax.annotate(rec["id"], xy=(0.5, 1.01), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=6)
        if r == 0:
            for c in range(ncols):
                axes[0][c].set_title(heads[c], fontsize=11, fontweight="bold", pad=16)

    fig.suptitle(f"{disp}   ·   {split_label}", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.019,
             "Fondo gris: red vial del entorno.   Trazo azul: subgrafo, grosor proporcional a la "
             "jerarquía vial.   s: similitud coseno con el objetivo.",
             ha="center", fontsize=7.5, color="#555555")
    fig.text(0.5, 0.007,
             "Naranja (puntos y recuadro): candidato en el mismo lugar que el objetivo; los puntos "
             "marcan los nodos compartidos.   Cada panel se centra en su centroide (°N, °O) y "
             "comparte la escala con el resto de su fila.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0.02, 0.032, 1, 0.965], w_pad=0.8, h_pad=1.6)
    for ext_ in ("png", "pdf"):
        p = stem.with_suffix(f".{ext_}")
        fig.savefig(p, dpi=300)
        log.info("  escrito %s", p.relative_to(ROOT))
    plt.close(fig)


def main():
    subs, struct_dims, netmap = load_corpus()
    log.info("Corpus: %d subgrafos", len(subs))

    log.info("Eligiendo 10 arquetipos...")
    arche = pick_archetypes(subs, struct_dims)
    targets_idx = [i for _, i, _ in arche]

    techniques = {}
    log.info("Tecnica 1/3: graph2vec fusionado (produccion)...")
    techniques["graph2vec_fusionado"] = emb_graph2vec(subs)
    log.info("Tecnica 2/3: Wasserstein Weisfeiler-Lehman (NeurIPS 2019)...")
    techniques["WWL"] = emb_wwl(subs, targets_idx)
    log.info("Tecnica 3/3: Geometric Scattering (ICML 2019)...")
    techniques["scattering_geometrico"] = emb_scattering(subs)

    by_id = {s["id"]: s for s in subs}
    DISP = {"graph2vec_fusionado": "graph2vec (fusionado, produccion)",
            "WWL": "Wasserstein Weisfeiler-Lehman",
            "scattering_geometrico": "Geometric Scattering"}
    dump = {"archetypes": [{"name": n, "target_id": tid} for n, _, tid in arche],
            "techniques": {}}
    tables = {}
    for tname, S in techniques.items():
        table = {}
        for name, ti, tid in arche:
            general, other = top5_split(S, subs, ti)
            table[tid] = {"general": general, "other": other}
        tables[tname] = table
        dump["techniques"][tname] = table
        disp = DISP[tname]
        plot_gallery(disp, "general",
                     "Top-5 general (mismo lugar permitido y resaltado)",
                     arche, table, by_id, netmap, OUT / f"{tname}__top5_general")
        plot_gallery(disp, "other",
                     "Top-5 en otro lugar (0 nodos compartidos con el objetivo)",
                     arche, table, by_id, netmap, OUT / f"{tname}__otro_lugar")

    # figura principal para el cuerpo del paper: 5 arquetipos donde la forma
    # discrimina, una tecnica, la lista de "otro lugar" (la del diseno comparativo)
    focus_names = {"Corredor / avenida (camino largo, elongacion baja)",
                   "Cruce en T / estrella (grado-1 dominante)",
                   "Anillo / ciclo (grado-2 dominante)",
                   "Arbol ramificado (aristas = nodos-1, ramas)",
                   "Rejilla grande (>=45 nodos)"}
    focus_rows = [a for a in arche if a[0] in focus_names]
    plot_gallery("Wasserstein Weisfeiler-Lehman", "other",
                 "subgrafos estructuralmente analogos en otras zonas de la ciudad",
                 focus_rows, tables["WWL"], by_id, netmap,
                 OUT / "figura_principal_WWL", panel=2.3)

    (OUT / "tables.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False))

    # README con las tablas 10x6 en texto
    lines = ["# Subgrafos similares por tecnica de embedding\n"]
    lines.append("Objetivo (arquetipo) en la columna 1; top-5 en las cinco "
                 "siguientes, con la similitud al objetivo entre parentesis. "
                 "`[=]` marca un candidato que esta en el mismo lugar que el objetivo.\n")
    for tname in techniques:
        lines.append(f"\n## {tname}\n")
        for split, stitle in (("general", "Top-5 general (mismo lugar permitido, marcado con [=])"),
                              ("other", "Top-5 en otro lugar (0 nodos compartidos)")):
            lines.append(f"\n### {stitle}\n")
            lines.append("| Arquetipo (objetivo) | #1 | #2 | #3 | #4 | #5 |")
            lines.append("|---|---|---|---|---|---|")
            for name, _, tid in arche:
                recs = dump["techniques"][tname][tid][split]
                cells = [f'{"[=] " if r["shared_nodes"] else ""}{r["id"]} ({r["score"]:.3f})'
                         for r in recs] + [""] * (5 - len(recs))
                lines.append(f"| {name} / `{tid}` | " + " | ".join(cells) + " |")
    (OUT / "README.md").write_text("\n".join(lines))
    log.info("Tablas en %s", (OUT / "README.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
