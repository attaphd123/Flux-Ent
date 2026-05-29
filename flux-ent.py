# -*- coding: utf-8 -*-
"""
Flux-Ent: A physics‑informed centrality measure for identifying influential spreaders.

This implementation computes Flux-Ent scores and ranking using grid search
to select the best parameters (by mean score). It is the exact code
that produced the results reported in the paper.

Author: Aman Ullah 
"""

import os
import math
import numpy as np
import networkx as nx
from itertools import product

def minmax_norm(d):
    vals = np.array(list(d.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def rank_nodes_desc(scores):
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

def flux_ent_ranking(edge_file, out_dir=None, verbose=True):
    """
    Compute Flux-Ent ranking for a given edge list file.

    Parameters
    ----------
    edge_file : str
        Path to edge list file (space-separated, node IDs integers).
    out_dir : str or None
        Directory to save ranking TXT. If None, only print.
    verbose : bool
        Print progress and best configuration.

    Returns
    -------
    ranking : list of (node, score)
        Sorted list descending by score.
    best_config : tuple
        (R, charge_type, weight_type, decay_name, gamma1, gamma2)
    """
    # Parameters (same as your original)
    R_values = [2, 3, 5, None]
    charge_types = ['ks', 'deg']
    weight_types = ['ks', 'deg']
    decay_functions = {
        'inv_sq': lambda r: 1.0/(r*r),
        'exp_0.1': lambda r: math.exp(-0.1*r),
        'exp_0.5': lambda r: math.exp(-0.5*r),
        'exp_1.0': lambda r: math.exp(-1.0*r)
    }
    gamma_pairs = [
        (0.1,0.1),(0.1,0.5),(0.5,0.1),(0.5,0.5),
        (1.0,0.5),(0.5,1.0),(1.0,1.0),(2.0,1.0)
    ]

    # Load graph
    G = nx.read_edgelist(edge_file, nodetype=int, data=False).to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    if verbose:
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    nodes = sorted(G.nodes())
    deg = dict(G.degree())
    ks = nx.core_number(G)

    # Precompute helpers
    ks_norm = minmax_norm(ks)
    neighbor_sum = {v: sum(ks[u] for u in G.neighbors(v)) for v in nodes}
    neighbor_sum_norm = minmax_norm(neighbor_sum)

    best_score = None
    best_config = None
    best_mean_score = -1

    for R, charge_type, weight_type, decay_name, (g1, g2) in product(
            R_values, charge_types, weight_types,
            decay_functions.keys(), gamma_pairs):

        decay_func = decay_functions[decay_name]

        # ----- Flux -----
        flux_raw = {}
        for i in nodes:
            if R is None:
                dist = nx.single_source_shortest_path_length(G, i)
            else:
                dist = nx.single_source_shortest_path_length(G, i, cutoff=R)
            s = 0.0
            for j, r in dist.items():
                if j == i or r <= 0:
                    continue
                q = ks[j] if charge_type == 'ks' else deg[j]
                s += q * decay_func(r)
            flux_raw[i] = s
        flux_norm = minmax_norm(flux_raw)

        # ----- Entropy -----
        ent_norm = {}
        for i in nodes:
            di = deg[i]
            if di < 2:
                ent_norm[i] = 0.0
                continue
            weights = [ks[j] if weight_type == 'ks' else deg[j] for j in G.neighbors(i)]
            total = sum(weights)
            if total == 0:
                ent_norm[i] = 0.0
                continue
            H = 0.0
            for w in weights:
                p = w / total
                if p > 0:
                    H -= p * math.log(p)
            ent_norm[i] = H / math.log(di)

        # ----- Combine -----
        base = {v: flux_norm[v] * ent_norm[v] for v in nodes}
        score = {v: base[v] * (1 + g1*ks_norm[v] + g2*neighbor_sum_norm[v]) for v in nodes}
        mean_score = np.mean(list(score.values()))

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_score = score
            best_config = (R, charge_type, weight_type, decay_name, g1, g2)

    if verbose:
        print("Best configuration:", best_config)

    ranking = rank_nodes_desc(best_score)

    # Save to file if out_dir provided
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "FluxEnt_ranking.txt")
        with open(txt_path, "w") as f:
            f.write("Rank\tNode\tScore\n")
            for i, (node, sc) in enumerate(ranking, 1):
                f.write(f"{i}\t{node}\t{sc:.6f}\n")
        if verbose:
            print("Ranking saved to:", txt_path)

    return ranking, best_config

if __name__ == "__main__":
    # Example: use the toy network from the data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edge_file = os.path.join(script_dir, "data", "k-shell14.txt")
    ranking, config = flux_ent_ranking(edge_file, out_dir=os.path.join(script_dir, "ranking_results"))
    print("\nTop 10 nodes:")
    for i, (node, sc) in enumerate(ranking[:10], 1):
        print(f"{i:2d}  Node {node:2d}  Score {sc:.6f}")