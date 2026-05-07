import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as train_entry
from dassl.engine import build_trainer

from federated.client import get_prompt_state, set_prompt_state, train_client
from federated.server import average_prompts, save_prompt, select_clients
from federated.splits import make_client_splits, save_split_manifest


def build_args(parsed, output_dir, extra_opts):
    return SimpleNamespace(
        root=parsed.root,
        output_dir=output_dir,
        resume="",
        seed=parsed.seed,
        source_domains=None,
        target_domains=None,
        transforms=None,
        trainer=parsed.trainer,
        backbone="",
        head="",
        config_file=parsed.config_file,
        dataset_config_file=parsed.dataset_config_file,
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=extra_opts,
    )


def build_base_cfg(parsed):
    dataset_cfg = parsed.dataset_config_file or f"configs/datasets/{parsed.dataset}.yaml"
    method_cfg = parsed.config_file or f"configs/trainers/BiomedCoOp/few_shot/{parsed.dataset}.yaml"
    output_dir = str(Path(parsed.output_dir) / "base")

    opts = [
        "DATASET.NUM_SHOTS",
        str(parsed.shots),
        "DATALOADER.NUM_WORKERS",
        str(parsed.num_workers),
    ]
    opts.extend(parsed.opts)

    args = build_args(parsed, output_dir, opts)
    args.dataset_config_file = dataset_cfg
    args.config_file = method_cfg
    cfg = train_entry.setup_cfg(args)
    return cfg


def init_global_prompt(cfg):
    trainer = build_trainer(cfg)
    prompt = get_prompt_state(trainer)
    train_items = list(trainer.dm.dataset.train_x)
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt, train_items


def evaluate_prompt(base_cfg, prompt, output_dir):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = str(Path(output_dir) / "final_eval")
    cfg.TEST.NO_TEST = False
    cfg.freeze()

    trainer = build_trainer(cfg)
    set_prompt_state(trainer, prompt)
    result = trainer.test()
    return result


def main():
    parser = argparse.ArgumentParser(description="Simulated PromptFL for BiomedCoOp on one Windows PC")
    parser.add_argument("--root", default="data", help="Dataset root")
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. btmri")
    parser.add_argument("--trainer", default="BiomedCoOp_BiomedCLIP")
    parser.add_argument("--dataset-config-file", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--partition", choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="output/promptfl")
    parser.add_argument("--unweighted", action="store_true", help="Use plain average instead of sample-weighted average")
    parser.add_argument("--eval-final", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra Dassl config overrides")
    parsed = parser.parse_args()

    output_dir = Path(parsed.output_dir) / parsed.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = build_base_cfg(parsed)
    global_prompt, train_items = init_global_prompt(base_cfg)

    client_splits = make_client_splits(
        train_items,
        num_clients=parsed.num_clients,
        partition=parsed.partition,
        seed=parsed.seed,
        dirichlet_alpha=parsed.dirichlet_alpha,
    )
    manifest_path = save_split_manifest(client_splits, output_dir)
    print(f"Saved client split manifest to {manifest_path}")

    save_prompt(global_prompt, output_dir, round_id=0)
    print(f"Initialized global prompt with shape {tuple(global_prompt.shape)}")

    round_sccm_history = []
    round_kdsp_history = []
    round_ce_history = []



    for round_id in range(1, parsed.rounds + 1):
        selected = select_clients(parsed.num_clients, parsed.clients_per_round, parsed.seed, round_id)
        print(f"\n=== Communication round {round_id}/{parsed.rounds} | selected clients: {selected} ===")

        prompts = []
        sample_counts = []
        client_sccm_losses = []
        client_kdsp_losses = []
        client_ce_losses = []
        for client_id in selected:
            client_result = train_client(
                base_cfg,
                client_id=client_id,
                round_id=round_id,
                client_items=client_splits[client_id],
                global_prompt=global_prompt,
                local_epochs=parsed.local_epochs,
                output_dir=output_dir,
            )
            prompts.append(client_result["prompt"])
            sample_counts.append(client_result["num_samples"])
            client_sccm_losses.append(client_result["loss_sccm"])
            client_kdsp_losses.append(client_result["loss_kdsp"])
            client_ce_losses.append(client_result["loss_ce"])
            print(f"Client {client_id} finished local update on {client_result['num_samples']} samples" f" | avg_loss_sccm: {client_result['loss_sccm']:.4f} | avg_loss_kdsp: {client_result['loss_kdsp']:.4f} | avg_loss_ce: {client_result['loss_ce']:.4f}    ")

        weights = None if parsed.unweighted else sample_counts
        global_prompt = average_prompts(prompts, sample_counts=weights)
        prompt_path = save_prompt(global_prompt, output_dir, round_id=round_id)
        print(f"Saved aggregated global prompt to {prompt_path}")

        total_samples = sum(sample_counts)
        if parsed.unweighted:
            round_loss_sccm = sum(client_sccm_losses)/len(client_sccm_losses)
            round_loss_kdsp = sum(client_kdsp_losses)/len(client_kdsp_losses)
            round_loss_ce = sum(client_ce_losses)/len(client_ce_losses)
        else:
            round_loss_sccm = sum(
                loss * n for loss,n in zip(client_sccm_losses, sample_counts)
            )/total_samples
            round_loss_kdsp = sum(
                loss * n for loss,n in zip(client_kdsp_losses, sample_counts)
            )/total_samples
            round_loss_ce = sum(
                loss * n for loss,n in zip(client_ce_losses, sample_counts)
            )/total_samples

        round_sccm_history.append(round_loss_sccm)
        round_kdsp_history.append(round_loss_kdsp)
        round_ce_history.append(round_loss_ce)

        print(
            f"Round {round_id} aggregated_metrics: "
            f"avg_loss_sccm: {round_loss_sccm:.4f} | "
            f"avg_loss_kdsp: {round_loss_kdsp:.4f} | "
            f"avg_loss_ce: {round_loss_ce:.4f}"
        )

    final_path = Path(output_dir) / "global_prompt_final.pt"
    torch.save({"round": parsed.rounds, "prompt": global_prompt}, final_path)
    print(f"\nSaved final global prompt to {final_path}")

    if parsed.eval_final:
        result = evaluate_prompt(base_cfg, global_prompt, output_dir)
        print(f"Final test result: {result}")

    



if __name__ == "__main__":
    main()
