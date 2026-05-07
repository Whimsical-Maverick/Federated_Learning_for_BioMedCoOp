import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def split_by_label(items):
    by_label = defaultdict(list)
    for item in items:
        by_label[item.label].append(item)
    return by_label


def make_iid_splits(items, num_clients, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    splits = [[] for _ in range(num_clients)]
    for idx, item in enumerate(shuffled):
        splits[idx % num_clients].append(item)
    return splits


def make_dirichlet_splits(items, num_clients, alpha, seed, min_size=1):
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    by_label = split_by_label(items)

    for label_items in by_label.values():
        rng.shuffle(label_items)

    splits = None
    for _ in range(100):
        candidate = [[] for _ in range(num_clients)]

        for label in sorted(by_label):
            label_items = by_label[label]
            proportions = np_rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(label_items)).astype(int)

            for client_id, chunk in enumerate(np.split(np.array(label_items, dtype=object), split_points)):
                candidate[client_id].extend(chunk.tolist())

        if min(len(split) for split in candidate) >= min_size:
            splits = candidate
            break

    if splits is None:
        splits = candidate
        empty = [idx for idx, split in enumerate(splits) if not split]
        if empty:
            raise RuntimeError(
                "Dirichlet split produced empty clients. Try increasing "
                "--dirichlet-alpha, reducing --num-clients, or increasing --shots. "
                f"Empty client ids: {empty}"
            )

    for split in splits:
        rng.shuffle(split)

    return splits


def make_client_splits(items, num_clients, partition, seed, dirichlet_alpha):
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    if partition == "iid":
        return make_iid_splits(items, num_clients, seed)
    if partition == "noniid":
        return make_dirichlet_splits(items, num_clients, dirichlet_alpha, seed)

    raise ValueError(f"Unknown partition mode: {partition}")


def save_split_manifest(client_splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for client_id, split in enumerate(client_splits):
        manifest[f"client_{client_id}"] = [
            {
                "impath": item.impath,
                "label": item.label,
                "classname": item.classname,
            }
            for item in split
        ]

    path = output_dir / "client_splits.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
