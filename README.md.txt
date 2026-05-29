This repository contains the official implementation of Flux‑Ent, a novel centrality measure for identifying influential spreaders in complex networks. Flux‑Ent uniquely integrates:

- A global flux field inspired by electric flux density (k‑shell charges with distance decay)
- A local entropy term that quantifies neighbourhood diversity
- A self‑importance boost using the node’s own k‑shell and the sum of its neighbours’ k‑shells

The framework automatically selects optimal parameters (radius, decay, boost coefficients) via grid search to maximise the mean score.

## Features

- Works on undirected, unweighted graphs (any size)
- Supports inverse‑square and exponential distance decay
- Includes grid search for adaptive parameter selection
- Outputs node ranking and scores to a text file
- Fully reproducible – produces the same results as reported in the paper

## Requirements

- Python 3.7+
- `networkx`
- `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt