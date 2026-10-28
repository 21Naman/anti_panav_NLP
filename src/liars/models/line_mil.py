"""Line-level contradiction checker trained with multiple-instance learning (MIL).

Every chat line is scored on its own against the evidence:
    text_a = "Evidence: " + facts
    text_b = a window of the chat around the line (context_before lines before it,
             context_after after it), with the scored line wrapped in <t> ... </t>
A cross-encoder (DeBERTa, NLI-pretrained) turns each pair into one score: the log-odds
that the marked line contradicts the evidence.

Only the liar is labelled, not the lying line, so the line scores are pooled per speaker
(a soft maximum: tau * logsumexp(score / tau)) and a softmax across the 4 speakers is
trained with cross-entropy against the liar label. The model learns by itself which
line is the lie. When a chat carries meta.lie_msg_idx (our synthetic data), an optional
line-level BCE term (line_mil.line_loss_weight) supervises the lines directly; use it for
stage-1 pretraining on synthetic chats.

head: contradiction (default) starts from the NLI head's contradiction logit, so the
model is already a reasonable zero-shot checker at step 0; head: new uses a fresh
1-output head.

The pure helpers at the top (windows, pooling, labels) need no torch and are unit-tested
on their own.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from liars.augment import map_back, permute_speakers, shuffle_evidence, tta_variants
from liars.config import seed_everything
from liars.device import get_device
from liars.models.base import Model
from liars.preds import make_preds, softmax
from liars.schema import SPEAKERS, Example

META_FILE = "model_meta.json"


# ---------------------------------------------------------------------------
# Pure helpers (no torch)
# ---------------------------------------------------------------------------
@dataclass
class LineItem:
    msg_idx: int        # index of the scored message in ex.messages
    speaker: str
    window: list[int]   # message indices shown, in chat order (msg_idx included)


def line_windows(ex: Example, before: int, after: int) -> list[LineItem]:
    """One item per message: the message plus up to `before` / `after` neighbours."""
    n = len(ex.messages)
    return [LineItem(i, ex.messages[i].speaker, list(range(max(0, i - before), min(n, i + after + 1))))
            for i in range(n)]


def fit_window(window: Sequence[int], target: int, lens: dict[int, int], budget: int) -> tuple[list[int], int]:
    """Drop context lines furthest from `target` until sum(lens) <= budget. The target
    line is never dropped. Returns (kept indices in order, number dropped)."""
    keep = list(window)
    total = sum(lens[i] for i in keep)
    dropped = 0
    for i in sorted((j for j in window if j != target), key=lambda j: (-abs(j - target), j)):
        if total <= budget:
            break
        keep.remove(i)
        total -= lens[i]
        dropped += 1
    return keep, dropped


def render_window(ex: Example, keep: Sequence[int], target: int, line_format: str,
                  mark_open: str, mark_close: str) -> str:
    out = []
    for i in keep:
        line = line_format.format(speaker=ex.messages[i].speaker, text=ex.messages[i].text)
        out.append(f"{mark_open} {line} {mark_close}" if i == target else line)
    return "\n".join(out)


def line_labels(ex: Example) -> np.ndarray | None:
    """1.0 for lie lines, 0.0 otherwise, from meta.lie_msg_idx; None when unknown (real data)."""
    idx = ex.meta.get("lie_msg_idx") if isinstance(ex.meta, dict) else None
    if not idx:
        return None
    y = np.zeros(len(ex.messages), dtype=np.float32)
    y[list(idx)] = 1.0
    return y


def pool_np(scores: np.ndarray, speakers: Sequence[str], method: str, tau: float,
            empty: float) -> np.ndarray:
    """Line scores -> one score per speaker (A..D). Speakers without lines get `empty`."""
    out = np.full(len(SPEAKERS), float(empty))
    spk = np.asarray(speakers)
    for j, s in enumerate(SPEAKERS):
        v = np.asarray(scores, dtype=float)[spk == s]
        if v.size == 0:
            continue
        if method == "max":
            out[j] = v.max()
        elif method == "mean":
            out[j] = v.mean()
        elif method == "logsumexp":
            m = v.max()
            out[j] = m + tau * math.log(np.exp((v - m) / tau).sum())
        else:
            raise ValueError(f"line_mil.pool must be max | mean | logsumexp, got {method!r}")
    return out


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LineMILModel(Model):
    uses_val = True

    # ------------------------------------------------------------------ setup
    def _load(self, source: str, add_marks: bool) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        d = self.dcfg
        tok = AutoTokenizer.from_pretrained(source)
        if add_marks:
            tok.add_special_tokens({"additional_special_tokens": [d["mark_open"], d["mark_close"]]})
        kwargs = {"dtype": torch.float32}  # fp32 master weights (see deberta_mc: fp16 checkpoints)
        if d["head"] == "new":
            kwargs.update(num_labels=1, ignore_mismatched_sizes=True)
        elif d["head"] != "contradiction":
            raise ValueError(f"line_mil.head must be contradiction | new, got {d['head']!r}")
        model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        model.resize_token_embeddings(len(tok))
        self.contra_idx = None
        if d["head"] == "contradiction" or model.config.num_labels > 1:
            names = {i: str(n).lower() for i, n in model.config.id2label.items()}
            hits = [int(i) for i, n in names.items() if n.startswith("contra")]
            if len(hits) != 1 or model.config.num_labels < 2:
                raise ValueError(f"{source}: no single 'contradiction' label in {names}; "
                                 "use line_mil.head=new")
            self.contra_idx = hits[0]
        if d.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.device = get_device(d["device"])
        self.tok, self.model = tok, model.to(self.device)
        self.fp16 = bool(d["fp16"]) and self.device == "cuda"

    def log(self, msg: str) -> None:
        print(f"[line_mil] {msg}", flush=True)

    # ------------------------------------------------------------------ encoding
    def _encode_chat(self, ex: Example) -> tuple[list[list[int]], list[str]]:
        """Token ids of every line's (evidence, window) pair, and the line speakers."""
        d, tok = self.dcfg, self.tok
        text_a = d["evidence_prefix"] + " ".join(ex.evidence)
        a_len = len(tok(text_a, add_special_tokens=False)["input_ids"])
        q = d.get("question") or ""
        q_len = len(tok(q, add_special_tokens=False)["input_ids"]) if q else 0
        lines = [d["line_format"].format(speaker=m.speaker, text=m.text) for m in ex.messages]
        lens = {i: len(x) + 1 for i, x in enumerate(tok(lines, add_special_tokens=False)["input_ids"])}
        budget = d["max_len"] - d["budget_margin"] - d["special_tokens_budget"] - a_len - q_len
        ids, spk = [], []
        for item in line_windows(ex, d["context_before"], d["context_after"]):
            cost = dict(lens)
            cost[item.msg_idx] += d["mark_cost"]
            keep, dropped = fit_window(item.window, item.msg_idx, cost, budget)
            self.n_truncated += int(dropped > 0)
            text_b = render_window(ex, keep, item.msg_idx, d["line_format"], d["mark_open"], d["mark_close"])
            if q:
                text_b += "\n" + q
            ids.append(tok(text_a, text_b, truncation="longest_first", max_length=d["max_len"])["input_ids"])
            spk.append(item.speaker)
        return ids, spk

    def _encoded(self, ex: Example):
        """Encodings are reused across epochs (only speaker letters change under permutation,
        so permuted copies are cached under their own key)."""
        key = (ex.id, tuple(m.speaker for m in ex.messages), tuple(ex.evidence))
        if key not in self._cache:
            self._cache[key] = self._encode_chat(ex)
        return self._cache[key]

    def _line_scores(self, flat_ids: Sequence[list[int]]):
        """(n_lines,) contradiction log-odds for a flat list of encoded lines."""
        torch = self.torch
        pad = self.tok.pad_token_id
        L = max(len(x) for x in flat_ids)
        ids = torch.full((len(flat_ids), L), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, x in enumerate(flat_ids):
            ids[b, : len(x)] = torch.tensor(x)
            mask[b, : len(x)] = 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
        if self.contra_idx is None:
            return logits[:, 0]
        c = logits[:, self.contra_idx]
        others = torch.cat([logits[:, : self.contra_idx], logits[:, self.contra_idx + 1:]], dim=1)
        return c - torch.logsumexp(others, dim=1)  # log p(contra) - log p(not contra)

    def _pool(self, scores, speakers: Sequence[str]):
        """Torch version of pool_np for one chat: (n_lines,) -> (4,)."""
        torch = self.torch
        d = self.dcfg
        out = []
        for s in SPEAKERS:
            idx = [i for i, x in enumerate(speakers) if x == s]
            if not idx:
                out.append(scores.new_tensor(float(d["empty_speaker_score"])))
                continue
            v = scores[idx]
            if d["pool"] == "max":
                out.append(v.max())
            elif d["pool"] == "mean":
                out.append(v.mean())
            elif d["pool"] == "logsumexp":
                out.append(d["pool_tau"] * torch.logsumexp(v / d["pool_tau"], dim=0))
            else:
                raise ValueError(f"line_mil.pool must be max | mean | logsumexp, got {d['pool']!r}")
        return torch.stack(out)

    def _forward_chats(self, examples: Sequence[Example]):
        """Speaker logits (B, 4) and the per-chat line scores (list of (n_lines,) tensors)."""
        torch = self.torch
        enc = [self._encoded(ex) for ex in examples]
        flat = [x for ids, _ in enc for x in ids]
        chunk = self.dcfg["lines_per_forward"]
        scores = torch.cat([self._line_scores(flat[b : b + chunk]) for b in range(0, len(flat), chunk)])
        out, per_chat, start = [], [], 0
        for ids, spk in enc:
            s = scores[start : start + len(ids)]
            start += len(ids)
            per_chat.append(s)
            out.append(self._pool(s, spk))
        return torch.stack(out), per_chat

    # ------------------------------------------------------------------ training
    def _augment(self, ex: Example, rng: random.Random) -> Example:
        aug = self.cfg["augment"]
        if aug["permute_train"]:
            ex = permute_speakers(ex, rng)[0]
        if aug["shuffle_evidence"]:
            ex = shuffle_evidence(ex, rng)
        return ex

    def _loss(self, batch: Sequence[Example]):
        torch = self.torch
        F = torch.nn.functional
        logits, per_chat = self._forward_chats(batch)
        y = torch.tensor([SPEAKERS.index(ex.label) for ex in batch], device=self.device)
        loss = F.cross_entropy(logits, y)
        w = self.dcfg["line_loss_weight"]
        if w > 0:
            parts = []
            for ex, s in zip(batch, per_chat):
                lab = line_labels(ex)
                if lab is not None:
                    parts.append(F.binary_cross_entropy_with_logits(s, torch.tensor(lab, device=self.device)))
            if parts:
                loss = loss + w * torch.stack(parts).mean()
        return loss

    def _speaker_logits(self, examples: Sequence[Example]) -> np.ndarray:
        torch = self.torch
        out = []
        bs = self.dcfg["eval_batch_size"]
        with torch.inference_mode():
            for b in range(0, len(examples), bs):
                out.append(self._forward_chats(examples[b : b + bs])[0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(SPEAKERS)))

    def _eval(self, val):
        self.model.eval()
        p = softmax(self._speaker_logits(val))
        self.model.train()
        y = np.array([SPEAKERS.index(ex.label) for ex in val])
        return float((p.argmax(1) == y).mean()), _nll(p, y)

    def _optimizer(self, opt_lr: float):
        torch = self.torch
        d = self.dcfg
        decay = d["layerwise_lr_decay"]
        if decay >= 1.0:
            return torch.optim.AdamW(self.model.parameters(), lr=opt_lr, weight_decay=d["weight_decay"])
        layers = self.model.config.num_hidden_layers
        groups: dict[float, list] = {}
        for name, p in self.model.named_parameters():
            if "embeddings" in name:
                depth = 0
            elif ".layer." in name:
                depth = int(name.split(".layer.")[1].split(".")[0]) + 1
            else:
                depth = layers + 1
            groups.setdefault(opt_lr * decay ** (layers + 1 - depth), []).append(p)
        return torch.optim.AdamW([{"params": ps, "lr": lr} for lr, ps in groups.items()],
                                 weight_decay=d["weight_decay"])

    def fit(self, train, val, cfg):
        self.cfg, self.dcfg = cfg, cfg["line_mil"]
        d = self.dcfg
        self.tta_n = cfg["augment"]["tta_n"]
        self.n_truncated = 0
        self._cache: dict = {}
        refit = cfg["train"].get("_refit") or {}
        init = cfg["train"].get("_init_from")
        seed_everything(cfg["seed"])
        self._load(init or d["model_id"], add_marks=not init)
        torch = self.torch
        if init:
            self.log(f"initialised from {init}")
        rng = random.Random(cfg["seed"])

        n_lines = [len(ex.messages) for ex in train]
        lens = [max(len(x) for x in self._encoded(ex)[0]) for ex in train[: d["length_probe_n"]]]
        self.log(f"{len(train)} train / {len(val) if val else 0} val chats; lines per chat p50 "
                 f"{np.percentile(n_lines, 50):.0f}, max {max(n_lines)}; tokens per line pair p50 "
                 f"{np.percentile(lens, 50):.0f}, p99 {np.percentile(lens, 99):.0f} (max_len {d['max_len']}); "
                 f"line pairs with dropped context: {self.n_truncated}")
        n_lab = sum(line_labels(ex) is not None for ex in train)
        if d["line_loss_weight"] > 0:
            self.log(f"line-level loss weight {d['line_loss_weight']} on {n_lab}/{len(train)} chats with lie_msg_idx")

        mb, accum = d["micro_batch_size"], d["grad_accum"]
        steps_per_epoch = max(1, math.ceil(len(train) / mb) // accum)
        total_steps = refit.get("steps") or math.ceil(d["epochs"] * steps_per_epoch)
        opt = self._optimizer(d["lr"])
        from transformers import get_cosine_schedule_with_warmup

        sched = get_cosine_schedule_with_warmup(opt, int(d["warmup_ratio"] * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        eval_every = max(1, int(d["eval_every_frac"] * steps_per_epoch))
        self.log(f"steps: {total_steps} total, {steps_per_epoch}/epoch, eval every {eval_every}")
        self.model.train()
        best = {"acc": -1.0, "loss": float("inf"), "step": 0, "state": None}
        self.history, step, micro, bad, t0, done, epoch = [], 0, 0, 0, time.time(), False, 0
        while not done:
            order = list(train)
            rng.shuffle(order)
            for b in range(0, len(order), mb):
                batch = [self._augment(ex, rng) for ex in order[b : b + mb]]
                scaler.scale(self._loss(batch) / accum).backward()
                micro += 1
                if micro % accum:
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), d["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step == d["time_probe_steps"]:
                    sec = (time.time() - t0) / step
                    fits = cfg["train"].get("_fits_left", 1)
                    est = sec * total_steps * fits / 3600
                    self.log(f"timing: {sec:.2f} s/step -> {sec * steps_per_epoch / 60:.1f} min per epoch, "
                             f"~{est:.2f} h for the {fits} fit(s) left (+ eval/predict)")
                    if est > d.get("max_gpu_hours", float("inf")):
                        raise RuntimeError(
                            f"estimated {est:.1f} GPU h > line_mil.max_gpu_hours {d['max_gpu_hours']}: "
                            "stopping (use the base model, holdout mode, fewer epochs, train.subsample, "
                            "smaller context, or raise the limit)")
                if val and (step % eval_every == 0 or step == total_steps):
                    acc, vloss = self._eval(val)
                    self.history.append({"step": step, "val_acc": acc, "val_loss": vloss,
                                         "minutes": round((time.time() - t0) / 60, 1)})
                    self.log(f"step {step}/{total_steps} val acc {acc:.4f} loss {vloss:.4f}")
                    if (acc, -vloss) > (best["acc"], -best["loss"]):
                        best = {"acc": acc, "loss": vloss, "step": step,
                                "state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}
                        bad = 0
                    else:
                        bad += 1
                        done = bad >= d["patience"]
                if step >= total_steps:
                    done = True
                if done:
                    break
            opt.zero_grad(set_to_none=True)
            micro = 0
            epoch += 1
            if epoch >= math.ceil(d["epochs"]) and not refit:
                done = True
        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        self.best_step = best["step"] if best["state"] is not None else step
        self.train_minutes = round((time.time() - t0) / 60, 1)
        self.max_mem_gb = (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                           if self.device == "cuda" else None)
        self.model.eval()
        self.temperature = refit.get("temperature", 1.0)
        self.val_line_hit = None
        if val:
            logits = self._tta_logits(val)
            yv = np.array([SPEAKERS.index(ex.label) for ex in val])
            self.temperature = min(d["temperatures"], key=lambda t: _nll(self._probs(logits, t), yv))
            self.val_line_hit = self.line_hit_rate(val)
        save_dir = cfg["train"].get("_save_dir")
        if d["save_model"] and save_dir:
            self.save(Path(save_dir))

    # ------------------------------------------------------------------ inference
    def _tta_logits(self, examples: Sequence[Example]) -> np.ndarray:
        variants = [tta_variants(ex, self.tta_n, self.cfg["seed"]) for ex in examples]
        flat = [v for vs in variants for v, _ in vs]
        raw = self._speaker_logits(flat).reshape(len(examples), self.tta_n, len(SPEAKERS))
        out = np.zeros_like(raw)
        for i, vs in enumerate(variants):
            for j, (_, inverse) in enumerate(vs):
                out[i, j] = map_back(raw[i, j], inverse)
        return out

    @staticmethod
    def _probs(logits: np.ndarray, t: float) -> np.ndarray:
        n, k, s = logits.shape
        return softmax(logits.reshape(n * k, s), t).reshape(n, k, s).mean(axis=1)

    def predict_proba(self, examples):
        self._last_tta_logits = self._tta_logits(examples)  # for the trainer's TTA check
        return make_preds([ex.id for ex in examples], self._probs(self._last_tta_logits, self.temperature))

    def line_scores(self, examples: Sequence[Example]) -> pd.DataFrame:
        """Long table: id, msg_idx, speaker, score (log-odds), p_contra. For error analysis:
        which line the model thinks is the lie."""
        torch = self.torch
        rows = []
        bs = self.dcfg["eval_batch_size"]
        with torch.inference_mode():
            for b in range(0, len(examples), bs):
                chunk = examples[b : b + bs]
                _, per_chat = self._forward_chats(chunk)
                for ex, s in zip(chunk, per_chat):
                    for i, v in enumerate(s.cpu().numpy().tolist()):
                        rows.append((ex.id, i, ex.messages[i].speaker, v, 1 / (1 + math.exp(-v))))
        return pd.DataFrame(rows, columns=["id", "msg_idx", "speaker", "score", "p_contra"])

    def line_hit_rate(self, examples: Sequence[Example]) -> float | None:
        """Share of chats with meta.lie_msg_idx whose top-scored line is a lie line."""
        lab = [ex for ex in examples if line_labels(ex) is not None]
        if not lab:
            return None
        df = self.line_scores(lab)
        top = df.loc[df.groupby("id")["score"].idxmax()].set_index("id")["msg_idx"]
        return float(np.mean([int(top[ex.id]) in ex.meta["lie_msg_idx"] for ex in lab]))

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))
        (path / META_FILE).write_text(json.dumps(
            {"temperature": self.temperature, "best_step": self.best_step, "history": self.history},
            indent=1))
        self.log(f"saved model -> {path}")

    @classmethod
    def load(cls, path: Path, cfg: dict) -> "LineMILModel":
        m = cls()
        m.cfg, m.dcfg, m.tta_n, m.n_truncated, m._cache = cfg, cfg["line_mil"], cfg["augment"]["tta_n"], 0, {}
        m._load(str(path), add_marks=False)
        meta = json.loads((Path(path) / META_FILE).read_text())
        m.temperature, m.best_step, m.history = meta["temperature"], meta["best_step"], meta["history"]
        m.model.eval()
        return m

    def refit_info(self) -> dict:
        return {"steps": max(1, self.best_step), "temperature": self.temperature}

    def report(self) -> str:
        d = self.dcfg
        rows = [("model", d["model_id"]), ("head", d["head"]), ("pool", f"{d['pool']} (tau {d['pool_tau']})"),
                ("context", f"{d['context_before']} before / {d['context_after']} after"),
                ("line loss weight", d["line_loss_weight"]), ("best step", self.best_step),
                ("train minutes", self.train_minutes), ("temperature", self.temperature),
                ("val lie-line hit rate (chats with lie_msg_idx)", self.val_line_hit),
                ("max GPU memory GiB (training)", self.max_mem_gb), ("tta_n", self.tta_n),
                ("line pairs with dropped context (all encodes)", self.n_truncated)]
        lines = ["# Line MIL report", "", "| item | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        if self.history:
            lines += ["", "| step | val acc | val loss | minutes |", "|---|---|---|---|"]
            lines += [f"| {r['step']} | {r['val_acc']:.4f} | {r['val_loss']:.4f} | {r['minutes']} |"
                      for r in self.history]
        return "\n".join(lines)
