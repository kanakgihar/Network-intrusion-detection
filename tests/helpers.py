"""Synthetic UNSW-NB15-shaped data for fast, offline tests (NOT real traffic)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUMERIC = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "smean", "ct_srv_src"]
CATEGORIES = {"proto": ["tcp", "udp", "icmp"], "service": ["-", "http", "dns"], "state": ["FIN", "CON", "INT"]}
ATTACKS = ["Exploits", "Generic", "DoS"]


def make_synthetic(n=900, seed=0, with_labels=True):
    rng = np.random.RandomState(seed)
    is_attack = rng.rand(n) < 0.55
    df = pd.DataFrame({"id": np.arange(1, n + 1)})
    df["dur"] = np.where(is_attack, rng.exponential(0.5, n), rng.exponential(2.0, n))
    df["spkts"] = rng.poisson(np.where(is_attack, 30, 8))
    df["dpkts"] = rng.poisson(np.where(is_attack, 2, 9))
    df["sbytes"] = df["spkts"] * rng.randint(40, 1500, n)
    df["dbytes"] = df["dpkts"] * rng.randint(40, 1500, n)
    df["rate"] = rng.exponential(np.where(is_attack, 5000, 300))
    df["sttl"] = np.where(is_attack, 254, rng.choice([31, 62, 254], n, p=[.3, .6, .1]))
    df["dttl"] = np.where(is_attack, 0, 252)
    df["smean"] = rng.randint(30, 900, n)
    df["ct_srv_src"] = rng.poisson(np.where(is_attack, 20, 3))
    for c, vals in CATEGORIES.items():
        df[c] = rng.choice(vals, n)
    if with_labels:
        df["attack_cat"] = np.where(is_attack, rng.choice(ATTACKS, n), "Normal")
        df["label"] = is_attack.astype(int)
    return df
