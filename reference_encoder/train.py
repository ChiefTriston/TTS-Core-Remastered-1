#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, BatchSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from .cli import add_common_args
from .pad_collate import pad_collate
from .config import RefEncConfig
from .dataset import RefEncDataset, SpeakerBalancedSampler, load_file_list
from .encoder import ReferenceEncoder
from .loss import ArcFaceLoss, GE2ELoss
from .verify_eer import evaluate_eer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--data-dir', required=True, help='Root directory for train/test file lists')
    parser.add_argument('--resume',    type=str, default=None, help='Checkpoint to resume from')
    return parser.parse_args()

def train():
    args = parse_args()
    cfg = RefEncConfig()
    cfg.output_dir = args.output_dir          # override with --output-dir
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, 'logs'))

    # figure out how many micro-batches to accumulate
    accumulation_steps = getattr(cfg, 'accumulation_steps', 1)

    # Load file lists
    train_files = load_file_list(os.path.join(args.data_dir, 'train'))
    test_files  = load_file_list(os.path.join(args.data_dir, 'test'))

    # Prepare balanced sampler wrapped in BatchSampler
    sam = SpeakerBalancedSampler(
        train_files,
        spk_per_batch=cfg.batch_size // cfg.speaker_batch_utterances,
        utts_per_spk=cfg.speaker_batch_utterances,
    )
    train_loader = DataLoader(
        RefEncDataset(train_files, cfg, is_train=True),
        batch_sampler=BatchSampler(sam, batch_size=cfg.batch_size, drop_last=True),
        num_workers=4,
        collate_fn=pad_collate,
    )
    eval_loader = DataLoader(
        RefEncDataset(test_files, cfg, is_train=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate,
    )

    # Model & loss
    # 1) instantiate the whole model on CPU
    model = ReferenceEncoder(cfg)

    # 2) leave the wav2vec2 backbone on CPU so its conv1d layers and weights stay together
    if cfg.backbone == 'wav2vec2':
        model.ssl.to('cpu')

    # 3) move everything *except* 'ssl' onto the GPU
    for name, module in model.named_children():
        if name != 'ssl':
            module.to(device)

    if cfg.loss_type == 'arcface':
        num_classes = len({spk for _, spk in train_files})
        loss_fn = ArcFaceLoss(
            in_features=cfg.speaker_dim,
            num_classes=num_classes,
            margin=cfg.margin,
            scale=cfg.scale,
            margin_schedule=(lambda s: cfg.margin) if cfg.use_margin_schedule else None
        )
    else:
        loss_fn = GE2ELoss()

    optimizer = AdamW(
        list(model.parameters()) + ([loss_fn.weight] if isinstance(loss_fn, ArcFaceLoss) else []),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step+1)/cfg.warmup_steps,
            0.5*(1 + np.cos((step-cfg.warmup_steps)/(cfg.total_steps-cfg.warmup_steps)*np.pi))
        )
    )
    scaler = GradScaler()
    step = 0
    best_eer = float('inf')

    # Resume checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        step = ckpt.get('step', 0)
        best_eer = ckpt.get('best_eer', best_eer)

    # prime gradients for accumulation
    optimizer.zero_grad()

    # Training loop with per-epoch evaluation
    for ep in trange(cfg.max_epochs, desc="Epochs", unit="epoch"):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader,
                                               desc=f"Ep {ep+1}/{cfg.max_epochs}",
                                               unit="batch")):
            with autocast():
                if cfg.mixup:
                    mels, (spk1, spk2, alpha) = batch
                    mels = mels.to(device)
                    emb = model(mels)
                    l1  = loss_fn(emb, spk1.to(device), step)
                    l2  = loss_fn(emb, spk2.to(device), step)
                    loss = alpha * l1 + (1 - alpha) * l2
                else:
                    mels, spks = batch
                    mels, spks = mels.to(device), spks.to(device)
                    emb  = model(mels)
                    loss = loss_fn(emb, spks)

                # scale down for accumulation
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # only step/zero every `accumulation_steps`
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # log true loss
                writer.add_scalar('train/loss', loss.item() * accumulation_steps, step)
                step += 1
                optimizer.zero_grad()

        # ─── End of epoch: evaluate on validation set ───
        model.eval()
        eer = evaluate_eer(model, eval_loader, device)
        print(f"[INFO] Epoch {ep+1} EER = {eer:.4f}")
        writer.add_scalar('eval/eer', eer, step)
        if eer < best_eer:
            best_eer = eer
            save_path = os.path.join(cfg.output_dir, 'best_model.pt')
            torch.save({
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'step':       step,
                'best_eer':   best_eer
            }, save_path)
            print(f"[INFO] Saved best_model.pt (EER={best_eer:.4f})")
        model.train()

    # ─── After all epochs: save final model ───
    final_path = os.path.join(cfg.output_dir, 'final_model.pt')
    torch.save({
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'step':       step,
        'best_eer':   best_eer
    }, final_path)
    print(f"[INFO] Saved final_model.pt (EER={best_eer:.4f})")
    writer.close()

if __name__ == '__main__':
    train()
