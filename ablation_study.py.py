# -*- coding: utf-8 -*-
"""
Ablation Study for Flux-Ent.

Evaluates five variants:
  - Flux only
  - Flux + Evenness
  - Flux + Evenness + Self-boost
  - Flux + Evenness + Neighbour-boost
  - Full Flux-Ent

Each network uses its own optimal parameters (R, decay, γ₁, γ₂) identified
by the sensitivity analysis.

Author: Aman Ullah
"""

import os, math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau
from itertools import product

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR  = "ablation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Per-network optimal parameters (same as in the paper)
BEST_PARAMS = {
    "karate":                dict(R=5, decay="inv_sq",  gamma1=2.0, gamma2=1.0),
    "ca-netscience":         dict(R=3, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
    "ca-GrQ":                dict(R=2, decay="inv_sq",  gamma1=2.0, gamma2=2.0),
    "Air traffic control":   dict(R=2, decay="inv_sq",  gamma1=2.0, gamma2=2.0),
    "moreno_blogs":          dict(R=3, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
    "FB-politician_edges":   dict(R=5, decay="exp_0.5", gamma1=2.0, gamma2=2.0),
}

VARIANTS = {
    "Flux only":                          dict(use_flux=True, use_even=True,
                                               use_self=True, use_neigh=False,
                                               override_gamma1=0.0, override_gamma2=0.0),
    "Flux + Evenness":                    dict(use_flux=True, use_even=True,
                                               use_self=False, use_neigh=False,
                                               override_gamma1=0.0, override_gamma2=0.0),
    "Flux + Evenness + Self-boost":       dict(use_flux=True, use_even=True,
                                               use_self=True, use_neigh=False,
                                               override_gamma1=None, override_gamma2=0.0),
    "Flux + Evenness + Neigh-boost":      dict(use_flux=True, use_even=True,
                                               use_self=False, use_neigh=True,
                                               override_gamma1=0.0, override_gamma2=None),
    "Full Flux-Ent":                      dict(use_flux=True, use_even=True,
                                               use_self=True, use_neigh=True,
                                               override_gamma1=None, override_gamma2=None),
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def minmax_norm(d):
    v = np.array(list(d.values()), dtype=float)
    mn, mx = v.min(), v.max()
    if mx - mn < 1e-15:
        return {k: 0.0 for k in d}
    return {k: (x - mn) / (mx - mn) for k, x in d.items()}


def compute_flux(G, ks, deg, R, decay):
    if decay == 'inv_sq':
        f = lambda d: 1.0 / (d * d)
    elif decay.startswith('exp_'):
        lam = float(decay.split('_')[1])
        f = lambda d: math.exp(-lam * d)
    else:
        raise ValueError(decay)

    flux = {}
    for i in G.nodes():
        dist = nx.single_source_shortest_path_length(G, i) if R is None else \
               nx.single_source_shortest_path_length(G, i, cutoff=R)
        flux[i] = sum(ks[j] * f(r) for j, r in dist.items()
                      if j != i and r > 0)
    return flux


def compute_evenness(G, ks, deg):
    """Normalised Shannon entropy of neighbour k-shell weights."""
    ent = {}
    for i in G.nodes():
        di = deg[i]
        if di < 2:
            ent[i] = 0.0; continue
        ws = [ks[j] for j in G.neighbors(i)]
        tot = sum(ws)
        if tot <= 1e-15:
            ent[i] = 0.0; continue
        H = -sum((w / tot) * math.log(w / tot) for w in ws if w > 0)
        ent[i] = H / math.log(di)
    return ent


def flux_ent_variant(G, params, variant):
    """Compute Flux-Ent scores for a given ablation variant."""
    deg = dict(G.degree())
    ks  = nx.core_number(G)

    flux = compute_flux(G, ks, deg, params['R'], params['decay'])
    flux_norm = minmax_norm(flux)
    even = compute_evenness(G, ks, deg)

    base = {v: flux_norm[v] * even[v] for v in G.nodes()}

    ks_norm = minmax_norm(ks)
    neigh_sum = {v: sum(ks[u] for u in G.neighbors(v)) for v in G.nodes()}
    neigh_norm = minmax_norm(neigh_sum)

    g1 = params['gamma1'] if variant['override_gamma1'] is None else variant['override_gamma1']
    g2 = params['gamma2'] if variant['override_gamma2'] is None else variant['override_gamma2']

    scores = {}
    for v in G.nodes():
        boost = 1.0
        if variant['use_self']:
            boost *= (1 + g1 * ks_norm[v])
        if variant['use_neigh']:
            boost *= (1 + g2 * neigh_norm[v])
        scores[v] = base[v] * boost
    return scores


def load_sir_rank(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def kendall_against_sir(scores, sir_rank):
    pos = {n: i for i, n in enumerate(sir_rank)}
    order = sorted(scores, key=scores.get, reverse=True)
    tau, _ = kendalltau(range(len(order)), [pos[n] for n in order])
    return tau


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rows = []
    for net, params in BEST_PARAMS.items():
        edge_path = os.path.join(DATA_DIR, net + ".txt")
        sir_path  = os.path.join(DATA_DIR, net + "_sir_rank.txt")
        if not (os.path.exists(edge_path) and os.path.exists(sir_path)):
            print(f"[skip] {net} (missing files)")
            continue

        print(f"\n=== {net} ===")
        G = nx.read_edgelist(edge_path, nodetype=int).to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        sir_rank = load_sir_rank(sir_path)

        for variant_name, variant_cfg in VARIANTS.items():
            scores = flux_ent_variant(G, params, variant_cfg)
            tau = kendall_against_sir(scores, sir_rank)
            rows.append({"Network": net, "Variant": variant_name, "Kendall_tau": tau})
            print(f"  {variant_name:35s}: τ = {tau:.4f}")

    df = pd.DataFrame(rows)

    # Pivot to the paper layout
    order = ["Flux only", "Flux + Evenness",
             "Flux + Evenness + Self-boost",
             "Flux + Evenness + Neigh-boost",
             "Full Flux-Ent"]
    pivot = df.pivot(index="Variant", columns="Network",
                     values="Kendall_tau").reindex(order)
    pivot["Average"] = pivot.mean(axis=1).round(4)
    pivot = pivot.round(4)

    print("\n=== Ablation Table (Kendall's τ) ===")
    print(pivot)
    pivot.to_csv(os.path.join(OUT_DIR, "ablation_kendall_table.csv"))

    with open(os.path.join(OUT_DIR, "ablation_kendall_table.tex"),
              "w", encoding="utf-8") as f:
        f.write(pivot.to_latex(
            float_format="%.4f",
            caption="Ablation study: Kendall's $\\tau$ for each variant.",
            label="tab:ablation_kendall"))
    print(f"\nResults saved in {OUT_DIR}/")
