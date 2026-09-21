# -*- coding: utf-8 -*-
"""
Runtime Analysis for Flux-Ent and all baselines.

Two benchmarks:
  (A) Real networks with per-network optimal parameters
  (B) Synthetic ER and BA networks with fixed parameters (R=3)

Produces three figures:
  fig_runtime_real.png — Bar chart on six real networks
  fig_runtime_er.png   — Runtime vs |V| on ER graphs
  fig_runtime_ba.png   — Runtime vs |V| on BA graphs

Author: Aman Ullah
"""

import os, time, math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp

OUT_DIR = "runtime_results"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "data"
SEED = 42

BEST_PARAMS = {
    "karate":                dict(R=5, decay="inv_sq",  gamma1=2.0, gamma2=1.0),
    "ca-netscience":         dict(R=3, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
    "ca-GrQ":                dict(R=2, decay="inv_sq",  gamma1=2.0, gamma2=2.0),
    "Air traffic control":   dict(R=2, decay="inv_sq",  gamma1=2.0, gamma2=2.0),
    "moreno_blogs":          dict(R=3, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
    "FB-politician_edges":   dict(R=5, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
}
DEFAULT_PARAMS = dict(R=3, decay="inv_sq", gamma1=2.0, gamma2=2.0)

NETWORK_NAME_MAP = {
    "karate": "Karate", "ca-netscience": "Ca-netscience",
    "ca-GrQ": "Ca-GrQ", "Air traffic control": "Air-traffic",
    "moreno_blogs": "Moreno-blogs", "FB-politician_edges": "FB-politician",
}
NETWORK_ORDER = ["Karate", "Ca-netscience", "Ca-GrQ",
                 "Air-traffic", "Moreno-blogs", "FB-politician"]


# ----------------------------------------------------------------------
# Flux-Ent
# ----------------------------------------------------------------------
def minmax_norm(d):
    v = np.array(list(d.values()), dtype=float)
    mn, mx = v.min(), v.max()
    if mx - mn < 1e-15:
        return {k: 0.0 for k in d}
    return {k: (x - mn) / (mx - mn) for k, x in d.items()}


def flux_ent(G, R=3, decay='inv_sq', gamma1=2.0, gamma2=2.0):
    deg = dict(G.degree())
    ks  = nx.core_number(G)
    if decay == 'inv_sq':
        f = lambda r: 1.0 / (r * r)
    elif decay.startswith('exp_'):
        lam = float(decay.split('_')[1])
        f = lambda r: math.exp(-lam * r)
    else:
        raise ValueError(decay)

    flux_raw = {}
    for i in G.nodes():
        dist = nx.single_source_shortest_path_length(G, i, cutoff=R)
        flux_raw[i] = sum(ks[j] * f(r) for j, r in dist.items()
                          if j != i and r > 0)
    flux_norm = minmax_norm(flux_raw)

    even = {}
    for i in G.nodes():
        di = deg[i]
        if di < 2:
            even[i] = 0.0; continue
        ws = [ks[j] for j in G.neighbors(i)]
        tot = sum(ws)
        if tot <= 1e-15:
            even[i] = 0.0; continue
        H = -sum((w/tot) * math.log(w/tot) for w in ws if w > 0)
        even[i] = H / math.log(di)

    base = {v: flux_norm[v] * even[v] for v in G.nodes()}
    kn = minmax_norm(ks)
    ns = minmax_norm({v: sum(ks[u] for u in G.neighbors(v)) for v in G.nodes()})
    return {v: base[v] * (1 + gamma1 * kn[v] + gamma2 * ns[v]) for v in G.nodes()}


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def degree_centrality(G):      return dict(G.degree())

def betweenness_centrality(G):
    n = G.number_of_nodes()
    k = min(200, n)
    return nx.betweenness_centrality(G, normalized=True, k=k)

def kshell_centrality(G):      return nx.core_number(G)

def mdd_centrality(G, theta=0.7):
    deg = dict(G.degree()); ks = nx.core_number(G)
    return {v: deg[v] + theta*(deg[v]-ks[v]) for v in G.nodes()}

def oved_rank_simplified(G):
    deg = dict(G.degree()); ks = nx.core_number(G)
    tri = nx.triangles(G); clo = nx.closeness_centrality(G)
    return {v: deg[v]*ks[v]*tri[v]*clo[v] for v in G.nodes()}

def eigenvector_centrality(G):
    try:    return nx.eigenvector_centrality_numpy(G)
    except: return nx.eigenvector_centrality(G, max_iter=1000)

def closeness_centrality(G):   return nx.closeness_centrality(G)

def compute_SLCMF_simplified(G, l=2, alpha=0.6,
                             delta_st=0.35, delta_so=0.2, delta_se=0.45,
                             top_k=5):
    """Memory-efficient SLCMF."""
    nodes = list(G.nodes()); n = len(nodes)
    deg = dict(G.degree()); ks = nx.core_number(G); clust = nx.clustering(G)

    F = np.zeros((n, 3))
    for i, v in enumerate(nodes):
        F[i] = [deg[v], ks[v], clust[v]]
    for j in range(3):
        col = F[:, j]; r = col.max() - col.min()
        F[:, j] = (col - col.min()) / r if r > 1e-12 else 0

    nsets = {v: set(G.neighbors(v)) for v in nodes}
    struct, social = {}, {}
    for v in nodes:
        dist = nx.single_source_shortest_path_length(G, v, cutoff=l)
        sub = list(dist.keys()); sg = G.subgraph(sub).copy()

        def asp(g):
            if g.number_of_nodes() < 2:
                return 1.0
            sp = dict(nx.all_pairs_shortest_path_length(g))
            tot, cnt = 0, 0
            for u in g.nodes():
                for w in g.nodes():
                    if u < w:
                        tot += sp[u].get(w, l + 1); cnt += 1
            return tot / cnt if cnt else 1.0

        a1 = asp(sg); sg.remove_node(v); a2 = asp(sg)
        struct[v] = deg[v] * (abs(a2 - a1)/a1 if a1 > 0 else 0)

        js = []
        for u in sub:
            if u == v: continue
            i_ = len(nsets[v] & nsets[u]); u_ = len(nsets[v] | nsets[u])
            if u_: js.append(i_/u_)
        social[v] = np.mean(js) if js else 0

    norms = np.linalg.norm(F, axis=1); norms[norms == 0] = 1.0
    Fn = F / norms[:, None]
    row_i, col_i, data = [], [], []
    for i in range(n):
        s = Fn @ Fn[i]; s[i] = -np.inf
        idx = np.argpartition(-s, min(top_k, n-1))[:min(top_k, n-1)]
        for j in idx:
            row_i.append(i); col_i.append(j); data.append(s[j])
    W = sp.csr_matrix((data, (row_i, col_i)), shape=(n, n))
    mx = W.max() if W.nnz else 0
    if mx > 0: W = W / mx

    A = nx.adjacency_matrix(G, nodelist=nodes).tocsr()
    I_sp = sp.identity(n, format='csr')
    GR = (1.0 / np.mean(list(deg.values()))) * (I_sp + A)
    W_rowsum = np.asarray(W.sum(axis=1)).flatten()
    GR_rowsum = np.asarray(GR.sum(axis=1)).flatten()
    sem = {nodes[i]: float((alpha*W_rowsum + (1-alpha)*GR_rowsum)[i])
           for i in range(n)}

    sn = minmax_norm(struct); on = minmax_norm(social); sm = minmax_norm(sem)
    return {v: delta_st*sn[v] + delta_so*on[v] + delta_se*sm[v] for v in nodes}


METHODS = {
    "Flux-Ent":    flux_ent,
    "Degree":      degree_centrality,
    "Betweenness": betweenness_centrality,
    "K-shell":     kshell_centrality,
    "MDD":         mdd_centrality,
    "OVED-Rank":   oved_rank_simplified,
    "EC":          eigenvector_centrality,
    "CC":          closeness_centrality,
    "SLCMF":       compute_SLCMF_simplified,
}

COLORS = {
    "Flux-Ent": "black", "Degree": "tab:blue", "Betweenness": "tab:orange",
    "K-shell": "tab:green", "MDD": "tab:red", "OVED-Rank": "tab:purple",
    "EC": "tab:brown", "CC": "tab:pink", "SLCMF": "tab:gray",
}
MARKERS = {
    "Flux-Ent": "s", "Degree": "o", "Betweenness": "^", "K-shell": "v",
    "MDD": "D", "OVED-Rank": "p", "EC": "*", "CC": "X", "SLCMF": "h",
}
METHOD_ORDER = list(METHODS.keys())


# ----------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------
def benchmark_real():
    print("\n=== Real-network benchmark ===")
    header = f"{'Network':<24}" + "".join(f"{m:>12}" for m in METHODS)
    print(header); print("-" * len(header))

    results = {m: [] for m in METHODS}
    networks = []
    for net in BEST_PARAMS.keys():
        path = os.path.join(DATA_DIR, net + ".txt")
        if not os.path.exists(path):
            print(f"[skip] {net}"); continue
        G = nx.read_edgelist(path, nodetype=int).to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        networks.append(NETWORK_NAME_MAP[net])
        row = f"{NETWORK_NAME_MAP[net]:<24}"
        for name, func in METHODS.items():
            start = time.perf_counter()
            if name == "Flux-Ent":
                func(G, **BEST_PARAMS[net])
            else:
                func(G)
            elapsed = time.perf_counter() - start
            results[name].append(elapsed)
            row += f"{elapsed:>11.3f}s"
        print(row)
    return networks, results


def benchmark_synthetic(family, sizes, seed=42):
    print(f"\n=== Synthetic {family} benchmark ===")
    results = {m: [] for m in METHODS}
    for n in sizes:
        if family == "BA":
            G = nx.barabasi_albert_graph(n, m=3, seed=seed)
        else:
            G = nx.gnp_random_graph(n, 6.0/n, seed=seed)
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        for name, func in METHODS.items():
            start = time.perf_counter()
            if name == "Flux-Ent":
                func(G, **DEFAULT_PARAMS)
            else:
                func(G)
            results[name].append(time.perf_counter() - start)
    return results


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_real(networks, results, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(networks)); width = 0.09
    for i, (name, times) in enumerate(results.items()):
        offset = (i - len(results)/2) * width + width/2
        lw = 1.8 if name == "Flux-Ent" else 1.0
        ec = 'black' if name == "Flux-Ent" else 'gray'
        z  = 10 if name == "Flux-Ent" else 1
        ax.bar(x + offset, times, width, label=name, color=COLORS[name],
               edgecolor=ec, linewidth=lw, zorder=z)
    ax.set_xticks(x); ax.set_xticklabels(networks, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel("Execution time (s, log scale)", fontsize=13)
    ax.grid(alpha=0.3, axis='y', which='both', linestyle='--')
    ax.legend(fontsize=9, ncol=1, loc='upper left', frameon=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved: {save_path}")


def plot_scalability(results, sizes, family, save_path):
    x = [n/1000 for n in sizes]
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in METHOD_ORDER:
        lw = 3.0 if name == "Flux-Ent" else 1.8
        ms = 10  if name == "Flux-Ent" else 7
        z  = 10  if name == "Flux-Ent" else 1
        ax.plot(x, results[name], marker=MARKERS[name], linewidth=lw,
                markersize=ms, label=name, color=COLORS[name], zorder=z)
    ax.set_xlabel("NodeNumber (k)", fontsize=13)
    ax.set_ylabel("Execution Time (s)", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([f"{n//1000}k" for n in sizes])
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved: {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SIZES = [1000, 5000, 10000, 15000]

    networks, res_real = benchmark_real()
    plot_real(networks, res_real, os.path.join(OUT_DIR, "fig_runtime_real.png"))
    pd.DataFrame(res_real, index=networks).to_csv(
        os.path.join(OUT_DIR, "runtime_real_networks.csv"))

    res_er = benchmark_synthetic("ER", SIZES, SEED)
    plot_scalability(res_er, SIZES, "ER",
                     os.path.join(OUT_DIR, "fig_runtime_er.png"))
    pd.DataFrame(res_er, index=[f"{n//1000}k" for n in SIZES]).to_csv(
        os.path.join(OUT_DIR, "runtime_ER.csv"))

    res_ba = benchmark_synthetic("BA", SIZES, SEED)
    plot_scalability(res_ba, SIZES, "BA",
                     os.path.join(OUT_DIR, "fig_runtime_ba.png"))
    pd.DataFrame(res_ba, index=[f"{n//1000}k" for n in SIZES]).to_csv(
        os.path.join(OUT_DIR, "runtime_BA.csv"))

    print(f"\nAll results saved in: {OUT_DIR}")
