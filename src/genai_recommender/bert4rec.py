"""A small BERT4Rec: bidirectional transformer trained by masking items in the sequence
and predicting them, scored at inference by appending a [MASK] after the history.

Sized to train on a laptop CPU in minutes: 2 layers, 64-dim, 30-item window.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    # Median history is ~30 events, so a 30-slot window loses little and halves the work.
    max_len: int = 30
    dim: int = 64
    heads: int = 2
    layers: int = 2
    dropout: float = 0.2
    mask_prob: float = 0.2
    # Share of batches where only the last item is masked: the inference task exactly.
    # 0.5 beat 0.15 on validation NDCG@10 (0.294 vs 0.281 at epoch 16).
    last_only_prob: float = 0.5
    lr: float = 3e-3
    batch_size: int = 512
    epochs: int = 30
    eval_every: int = 2
    patience: int = 3
    seed: int = 7


class BERT4Rec(nn.Module):
    def __init__(self, n_items: int, cfg: Config) -> None:
        super().__init__()
        self.n_items = n_items
        self.cfg = cfg
        self.mask_id = n_items + 1
        self.item_emb = nn.Embedding(n_items + 2, cfg.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        layer = nn.TransformerEncoderLayer(
            cfg.dim,
            cfg.heads,
            4 * cfg.dim,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.out_bias = nn.Parameter(torch.zeros(n_items + 1))
        # Unit-variance embeddings make tied output logits ~8 wide at dim 64 and the loss
        # starts far above log(n_items); BERT-style small init fixes that.
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Hidden states, shape (batch, max_len, dim)."""
        pos = torch.arange(seq.shape[1], device=seq.device)
        h = self.drop(self.item_emb(seq) + self.pos_emb(pos)[None])
        h = self.encoder(h, src_key_padding_mask=seq == 0)
        return self.norm(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Scores over ids 0..n_items with tied output embeddings; id 0 is masked by callers."""
        return h @ self.item_emb.weight[: self.n_items + 1].T + self.out_bias


def _left_pad(seqs: Sequence[np.ndarray], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[i, -len(s) :] = s
    return out


def inference_batch(histories: Sequence[np.ndarray], cfg: Config, mask_id: int) -> torch.Tensor:
    """History truncated to max_len-1 with [MASK] appended in the last slot."""
    seqs = [np.append(h[-(cfg.max_len - 1) :], mask_id) for h in histories]
    return torch.from_numpy(_left_pad(seqs, cfg.max_len))


@torch.inference_mode()
def score(model: nn.Module, histories: Sequence[np.ndarray], batch_size: int = 1024) -> np.ndarray:
    model.eval()
    cfg: Config = model.cfg  # type: ignore[assignment]
    out = []
    for i in range(0, len(histories), batch_size):
        x = inference_batch(histories[i : i + batch_size], cfg, model.mask_id)  # type: ignore[arg-type]
        h = model(x)[:, -1]
        out.append(model.logits(h).numpy())  # type: ignore[operator]
    return np.concatenate(out)


def _training_batch(
    seqs: Sequence[np.ndarray], cfg: Config, mask_id: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    # Long histories contribute a random window each epoch rather than always the tail.
    windows = []
    for s in seqs:
        if len(s) > cfg.max_len:
            start = rng.integers(0, len(s) - cfg.max_len + 1)
            s = s[start : start + cfg.max_len]
        windows.append(s)
    x = _left_pad(windows, cfg.max_len)
    labels = np.zeros_like(x)
    real = x > 0
    if rng.random() < cfg.last_only_prob:
        m = np.zeros_like(real)
        m[:, -1] = real[:, -1]
    else:
        m = real & (rng.random(x.shape) < cfg.mask_prob)
        # Every sequence gets at least one masked position so no row is wasted.
        none = ~m.any(axis=1) & real.any(axis=1)
        m[none, -1] = True
    labels[m] = x[m]
    x[m] = mask_id
    return torch.from_numpy(x), torch.from_numpy(labels)


def train(
    train_seqs: Sequence[np.ndarray],
    n_items: int,
    cfg: Config,
    validate: Callable[[nn.Module], float] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[BERT4Rec, dict[str, object]]:
    """Masked-item training with early stopping on a validation callback (higher is better)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = BERT4Rec(n_items, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    seqs = [s for s in train_seqs if len(s) >= 2]
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(len(seqs))
        total, steps = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            x, y = _training_batch([seqs[j] for j in order[i : i + cfg.batch_size]], cfg, model.mask_id, rng)
            # Project only the masked positions onto the catalogue; the rest carry no label.
            sel = y != 0
            loss = loss_fn(model.logits(model(x)[sel]), y[sel])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            steps += 1
        if validate and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            v = validate(model)
            history.append(dict(epoch=epoch, loss=round(total / steps, 4), valid_ndcg10=round(v, 4)))
            log(f"  epoch {epoch:3d}  loss {total / steps:.3f}  valid NDCG@10 {v:.4f}")
            if v > best:
                best, best_epoch, bad = v, epoch, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    info: dict[str, object] = dict(
        config=asdict(cfg),
        best_epoch=best_epoch,
        best_valid_ndcg10=round(best, 4),
        train_seconds=round(time.perf_counter() - t0, 1),
        parameters=sum(p.numel() for p in model.parameters()),
        history=history,
    )
    return model, info
