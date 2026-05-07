from pathlib import Path

import torch

from dassl.data.data_manager import build_data_loader
from dassl.engine import build_trainer


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_prompt_state(trainer):
    model = _unwrap_model(trainer.model)
    return model.prompt_learner.ctx.detach().cpu().clone()


def set_prompt_state(trainer, prompt):
    model = _unwrap_model(trainer.model)
    target = model.prompt_learner.ctx
    with torch.no_grad():
        target.copy_(prompt.to(device=target.device, dtype=target.dtype))


def replace_train_loader(trainer, client_items):
    cfg = trainer.cfg
    trainer.train_loader_x = build_data_loader(
        cfg,
        sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
        data_source=client_items,
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
        n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
        tfm=trainer.train_loader_x.dataset.transform,
        is_train=True,
    )
    return trainer


def make_client_cfg(base_cfg, client_id, round_id, local_epochs, output_dir):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = local_epochs
    cfg.OUTPUT_DIR = str(Path(output_dir) / f"round_{round_id:03d}" / f"client_{client_id}")
    cfg.RESUME = ""
    cfg.TEST.NO_TEST = True
    cfg.TRAIN.CHECKPOINT_FREQ = 0
    cfg.freeze()
    return cfg


def train_client(base_cfg, client_id, round_id, client_items, global_prompt, local_epochs, output_dir):
    cfg = make_client_cfg(base_cfg, client_id, round_id, local_epochs, output_dir)
    trainer = build_trainer(cfg)
    replace_train_loader(trainer, client_items)

    if global_prompt is not None:
        set_prompt_state(trainer, global_prompt)

    total_loss_ce = 0.0
    total_loss_sccm = 0.0
    total_loss_kdsp = 0.0
    total_batches = 0

    trainer.max_epoch = local_epochs
    for epoch in range(local_epochs):
        trainer.epoch = epoch
        trainer.set_model_mode("train")
        trainer.num_batches = len(trainer.train_loader_x)
        for trainer.batch_idx , batch in enumerate(trainer.train_loader_x):
            loss_summary = trainer.forward_backward(batch)

            total_loss_ce += loss_summary["loss_ce"]
            total_loss_sccm += loss_summary["loss_sccm"]
            total_loss_kdsp += loss_summary["loss_kdsp"]
            total_batches += 1
        
        trainer.update_lr()
        
    avg_loss_ce = total_loss_ce / total_batches if total_batches > 0 else 0.0
    avg_loss_sccm = total_loss_sccm / total_batches if total_batches > 0 else 0.0
    avg_loss_kdsp = total_loss_kdsp / total_batches if total_batches > 0 else 0.0

    # trainer.run_epoch()

    prompt = get_prompt_state(trainer)
    num_samples = len(client_items)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prompt": prompt,
        "num_samples": num_samples,
        "loss_sccm": avg_loss_sccm,
        "loss_kdsp": avg_loss_kdsp,
        "loss_ce": avg_loss_ce,
    }

