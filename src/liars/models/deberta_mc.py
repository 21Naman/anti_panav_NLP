"""DeBERTa focus-marked multiple choice: 4 choices per chat, one per speaker.

Choice X: text_a = "Evidence: " + facts; text_b = the chat as "S: text" lines with X's lines
wrapped in <t> ... </t> (special tokens added to the tokenizer), then the question.
A softmax across the 4 choices gives the liar probabilities. Truncation keeps all evidence
and every line of X; other speakers' lines furthest from X's lines are dropped first.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from liars.augment import map_back, permute_speakers, shuffle_evidence, tta_variants
from liars.device import get_device
from liars.models.base import Model
from liars.preds import make_preds, softmax
from liars.schema import SPEAKERS, Example

META_FILE = "model_meta.json"


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def keep_for_choice(ex: Example, speaker: str, line_lens: Sequence[int], fixed: int,
                    max_len: int, mark_cost: int) -> tuple[list[int], int]:
    """Message indices kept for the choice `speaker` and how many were dropped.
    `fixed` = tokens outside the chat lines (evidence, question, special tokens)."""
    n = len(ex.messages)
    own = [i for i in range(n) if ex.messages[i].speaker == speaker]
    cost = {i: line_lens[i] + (mark_cost if i in own else 0) for i in range(n)}
    keep = set(range(n))
    total = fixed + sum(cost.values())
    if total <= max_len:
        return sorted(keep), 0

    def dist(i: int) -> float:
        return min((abs(i - j) for j in own), default=float("inf"))

    others = sorted((i for i in range(n) if i not in own), key=lambda i: (-dist(i), i))
    dropped = 0
    for i in others:
        if total <= max_len:
            break
        keep.discard(i)
        total -= cost[i]
        dropped += 1
    return sorted(keep), dropped


class DebertaMCModel(Model):
    uses_val = True

    # ------------------------------------------------------------------ setup
    def _load(self, source: str, add_marks: bool):
        import torch
        from transformers import AutoModelForMultipleChoice, AutoTokenizer

        self.torch = torch
        d = self.dcfg
        tok = AutoTokenizer.from_pretrained(source)
        if add_marks:
            tok.add_special_tokens({"additional_special_tokens": [d["mark_open"], d["mark_close"]]})
        # fp32 master weights: transformers 5 loads a checkpoint in its stored dtype (the NLI
        # DeBERTa checkpoints are fp16), and GradScaler cannot unscale fp16 gradients.
        # Autocast still runs the forward pass in fp16 on CUDA.
        model = AutoModelForMultipleChoice.from_pretrained(source, ignore_mismatched_sizes=True,
                                                           dtype=torch.float32)
        model.resize_token_embeddings(len(tok))
        if d.get("gradient_checkpointing"):  # large on a T4: activations otherwise overflow 15 GB
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.device = get_device(d["device"])
        self.tok, self.model = tok, model.to(self.device)
        self.fp16 = bool(d["fp16"]) and self.device == "cuda"

    def log(self, msg: str) -> None:
        print(f"[deberta] {msg}", flush=True)

    # ------------------------------------------------------------------ encoding
    def _encode_example(self, ex: Example) -> list[list[int]]:
        d = self.dcfg
        tok = self.tok
        text_a = d["evidence_prefix"] + " ".join(ex.evidence)
        lines = [d["line_format"].format(speaker=m.speaker, text=m.text) for m in ex.messages]
        line_lens = [len(x) + 1 for x in tok(lines, add_special_tokens=False)["input_ids"]]  # +1 newline
        a_len = len(tok(text_a, add_special_tokens=False)["input_ids"])
        q_len = len(tok(d["question"], add_special_tokens=False)["input_ids"])
        fixed = a_len + q_len + d["special_tokens_budget"]
        out = []
        for s in SPEAKERS:
            keep, dropped = keep_for_choice(ex, s, line_lens, fixed, d["max_len"] - d["budget_margin"],
                                            d["mark_cost"])
            self.n_truncated += int(dropped > 0)
            body = []
            for i in keep:
                if ex.messages[i].speaker == s:
                    body.append(f"{d['mark_open']} {lines[i]} {d['mark_close']}")
                else:
                    body.append(lines[i])
            text_b = "\n".join(body) + "\n" + d["question"]
            ids = tok(text_a, text_b, truncation="longest_first", max_length=d["max_len"])["input_ids"]
            out.append(ids)
        return out

    def _collate(self, encoded: Sequence[list[list[int]]]):
        torch = self.torch
        pad = self.tok.pad_token_id
        L = max(len(ids) for choices in encoded for ids in choices)
        input_ids = torch.full((len(encoded), len(SPEAKERS), L), pad, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for b, choices in enumerate(encoded):
            for c, ids in enumerate(choices):
                input_ids[b, c, : len(ids)] = torch.tensor(ids)
                mask[b, c, : len(ids)] = 1
        return input_ids.to(self.device), mask.to(self.device)

    def _logits(self, examples: Sequence[Example]):
        torch = self.torch
        ids, mask = self._collate([self._encode_example(ex) for ex in examples])
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            return self.model(input_ids=ids, attention_mask=mask).logits.float()

    # ------------------------------------------------------------------ training
    def _optimizer(self):
        torch = self.torch
        d = self.dcfg
        decay = d["layerwise_lr_decay"]
        if decay >= 1.0:
            return torch.optim.AdamW(self.model.parameters(), lr=d["lr"], weight_decay=d["weight_decay"])
        layers = self.model.config.num_hidden_layers
        groups: dict[float, list] = {}
        for name, p in self.model.named_parameters():
            if "embeddings" in name:
                depth = 0
            elif ".layer." in name:
                depth = int(name.split(".layer.")[1].split(".")[0]) + 1
            else:
                depth = layers + 1  # pooler / classifier head
            lr = d["lr"] * decay ** (layers + 1 - depth)
            groups.setdefault(lr, []).append(p)
        return torch.optim.AdamW([{"params": ps, "lr": lr} for lr, ps in groups.items()],
                                 weight_decay=d["weight_decay"])

    def _augment(self, ex: Example, rng: random.Random) -> Example:
        aug = self.cfg["augment"]
        if aug["permute_train"]:
            ex = permute_speakers(ex, rng)[0]
        if aug["shuffle_evidence"]:
            ex = shuffle_evidence(ex, rng)
        return ex

    def _eval(self, val):
        self.model.eval()
        logits = self._batched_logits(val)
        self.model.train()
        y = np.array([SPEAKERS.index(ex.label) for ex in val])
        p = softmax(logits)
        return float((p.argmax(1) == y).mean()), _nll(p, y)

    def _batched_logits(self, examples: Sequence[Example]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for b in range(0, len(examples), self.dcfg["eval_batch_size"]):
                out.append(self._logits(examples[b : b + self.dcfg["eval_batch_size"]]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(SPEAKERS)))

    def fit(self, train, val, cfg):
        self.cfg, self.dcfg = cfg, cfg["deberta"]
        d = self.dcfg
        self.tta_n = cfg["augment"]["tta_n"]
        self.n_truncated = 0
        refit = cfg["train"].get("_refit") or {}
        init = cfg["train"].get("_init_from")
        self._load(init or d["model_id"], add_marks=not init)
        torch = self.torch
        if init:
            self.log(f"initialised from {init}")
        rng = random.Random(cfg["seed"])

        lens = [max(len(ids) for ids in self._encode_example(ex)) for ex in train]
        a = np.asarray(lens)
        self.log(f"{len(train)} train / {len(val) if val else 0} val. tokens per choice: p50 "
                 f"{np.percentile(a, 50):.0f}, p90 {np.percentile(a, 90):.0f}, p99 "
                 f"{np.percentile(a, 99):.0f}, max {a.max()} (max_len {d['max_len']}); "
                 f"chats with dropped lines: {self.n_truncated}")

        mb, accum = d["micro_batch_size"], d["grad_accum"]
        steps_per_epoch = max(1, math.ceil(len(train) / mb) // accum)
        total_steps = refit.get("steps") or math.ceil(d["epochs"] * steps_per_epoch)
        opt = self._optimizer()
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
                y = torch.tensor([SPEAKERS.index(ex.label) for ex in batch], device=self.device)
                loss = torch.nn.functional.cross_entropy(self._logits(batch), y)
                scaler.scale(loss / accum).backward()
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
                    est_run = sec * total_steps * fits / 3600
                    self.log(f"timing: {sec:.2f} s/step -> {sec * steps_per_epoch / 60:.1f} min per "
                             f"epoch, {sec * total_steps / 60:.1f} min for {total_steps} steps; "
                             f"~{est_run:.2f} h for the {fits} fit(s) left (+ eval/predict)")
                    if est_run > d.get("max_gpu_hours", float("inf")):
                        raise RuntimeError(
                            f"estimated {est_run:.1f} GPU h > deberta.max_gpu_hours "
                            f"{d['max_gpu_hours']}: stopping (use the base model, fewer epochs, or "
                            "raise the limit)")
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
        if val:
            logits = self._tta_logits(val)
            yv = np.array([SPEAKERS.index(ex.label) for ex in val])
            self.temperature = min(d["temperatures"], key=lambda t: _nll(self._probs(logits, t), yv))
        save_dir = cfg["train"].get("_save_dir")
        if d["save_model"] and save_dir:
            self.save(Path(save_dir))

    # ------------------------------------------------------------------ inference
    def _tta_logits(self, examples: Sequence[Example]) -> np.ndarray:
        variants = [tta_variants(ex, self.tta_n, self.cfg["seed"]) for ex in examples]
        flat = [v for vs in variants for v, _ in vs]
        raw = self._batched_logits(flat).reshape(len(examples), self.tta_n, len(SPEAKERS))
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
        return make_preds([ex.id for ex in examples], self._probs(self._tta_logits(examples), self.temperature))

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))
        (path / META_FILE).write_text(json.dumps(
            {"temperature": self.temperature, "best_step": self.best_step, "history": self.history}, indent=1))
        self.log(f"saved model -> {path}")

    @classmethod
    def load(cls, path: Path, cfg: dict) -> "DebertaMCModel":
        m = cls()
        m.cfg, m.dcfg, m.tta_n, m.n_truncated = cfg, cfg["deberta"], cfg["augment"]["tta_n"], 0
        m._load(str(path), add_marks=False)
        meta = json.loads((Path(path) / META_FILE).read_text())
        m.temperature, m.best_step, m.history = meta["temperature"], meta["best_step"], meta["history"]
        m.model.eval()
        return m

    def refit_info(self) -> dict:
        return {"steps": max(1, self.best_step), "temperature": self.temperature}

    def report(self) -> str:
        rows = [("model", self.dcfg["model_id"]), ("best step", self.best_step),
                ("train minutes", self.train_minutes), ("temperature", self.temperature),
                ("gradient checkpointing", bool(self.dcfg.get("gradient_checkpointing"))),
                ("max GPU memory GiB (training)", getattr(self, "max_mem_gb", None)),
                ("tta_n", self.tta_n), ("choices with dropped lines (all encodes)", self.n_truncated)]
        lines = ["# DeBERTa MC report", "", "| item | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        if self.history:
            lines += ["", "| step | val acc | val loss | minutes |", "|---|---|---|---|"]
            lines += [f"| {r['step']} | {r['val_acc']:.4f} | {r['val_loss']:.4f} | {r['minutes']} |"
                      for r in self.history]
        return "\n".join(lines)
