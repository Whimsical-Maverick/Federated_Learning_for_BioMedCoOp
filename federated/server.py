import random
from pathlib import Path

import torch


def select_clients(num_clients, clients_per_round, seed, round_id):
    if clients_per_round > num_clients:
        raise ValueError("clients_per_round cannot exceed num_clients")
    rng = random.Random(seed + round_id)
    return sorted(rng.sample(range(num_clients), clients_per_round))


def average_prompts(prompt_updates, sample_counts=None):
    if not prompt_updates:
        raise ValueError("No prompt updates were provided")

    if sample_counts is None:
        return torch.stack(prompt_updates, dim=0).mean(dim=0)

    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Total sample count must be positive")

    result = torch.zeros_like(prompt_updates[0])
    for prompt, count in zip(prompt_updates, sample_counts):
        result += prompt * (float(count) / total)
    return result


def save_prompt(prompt, output_dir, round_id):
    round_dir = Path(output_dir) / f"round_{round_id:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "global_prompt.pt"
    torch.save({"round": round_id, "prompt": prompt}, path)
    return path

