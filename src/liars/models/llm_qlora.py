"""QLoRA open-LLM classifier (main model).

Base model in 4-bit NF4 (fp16 compute; T4 has no bf16), LoRA adapters on the attention +
MLP projections of the LANGUAGE model (vision/audio towers of multimodal checkpoints are
excluded by `qlora.target_regex`). Loss = cross-entropy over the 4 letter logits at the
answer position only. Mixed precision with a gradient scaler, gradient checkpointing,
paged 8-bit AdamW, cosine schedule with warmup, early stopping on the inner split.
Only the LoRA adapters are saved (data/models/<run>/<fold0|full>/).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from liars.augment import permute_speakers, shuffle_evidence
from liars.llm import load_lm, prompt_for, score_prompts, set_answer_prefix
from liars.models.llm_zeroshot import LLMZeroShotModel, _nll
from liars.schema import SPEAKERS, Example

META_FILE = "model_meta.json"


def length_stats(lengths: Sequence[int]) -> str:
    a = np.asarray(lengths)
    return (f"prompt tokens: p50 {np.percentile(a, 50):.0f}, p90 {np.percentile(a, 90):.0f}, "
            f"p99 {np.percentile(a, 99):.0f}, max {a.max()}")


class LLMQLoRAModel(LLMZeroShotModel):
    uses_val = True

    # ------------------------------------------------------------------ setup
    def _build(self, cfg: dict):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        self.torch = torch
        q = cfg["qlora"]
        h = load_lm(self.lcfg, for_training=True)  # fresh copy, never the zero-shot cache
        model = h.model
        model.config.use_cache = False
        quantized = h.device == "cuda" and self.lcfg["load_in_4bit"]
        ckpt_kwargs = {"use_reentrant": False}
        if quantized and q["upcast_non_quantized"]:
            # peft default: frozen non-4-bit weights to fp32 (costs ~4 bytes/param of memory)
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=q["gradient_checkpointing"],
                gradient_checkpointing_kwargs=ckpt_kwargs)
        elif q["gradient_checkpointing"]:
            # frozen weights stay fp16; autocast runs norms/softmax in fp32 anyway
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt_kwargs)
            model.enable_input_require_grads()
        init = cfg["train"].get("_init_from")  # resolved from train.init_from_run by the trainer
        if init:
            model = PeftModel.from_pretrained(model, init, is_trainable=True)
            self.log(f"initialised adapters from {init}")
        else:
            lora = LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                              target_modules=q["target_regex"], bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora)
        for p in model.parameters():  # the gradient scaler needs fp32 trainable weights
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_train == 0:
            raise ValueError("LoRA matched no modules: check qlora.target_regex for this model")
        self.log(f"trainable LoRA parameters: {n_train / 1e6:.1f} M")
        h.model = model
        import inspect

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        # Gemma-style multimodal models require token_type_ids when training (0 = text).
        self.needs_token_types = "token_type_ids" in inspect.signature(base.forward).parameters
        set_answer_prefix(h, self.pcfg.get("answer_prefix", ""))
        return h

    def log(self, msg: str) -> None:
        print(f"[qlora] {msg}", flush=True)

    # ------------------------------------------------------------------ training
    def _batch_loss(self, h, batch: Sequence[Example]):
        torch = self.torch
        prompts = [prompt_for(h, ex, self.pcfg, self.lcfg["max_len"]) for ex in batch]
        enc = h.tokenizer(prompts, add_special_tokens=False, padding=True,
                          return_tensors="pt").to(h.model.device)
        if self.needs_token_types and "token_type_ids" not in enc:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        y = torch.tensor([SPEAKERS.index(ex.label) for ex in batch], device=h.model.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=h.device == "cuda"):
            try:
                out = h.model(**enc, logits_to_keep=1)
            except TypeError:
                out = h.model(**enc)
        logits = out.logits[:, -1, :].float()[:, torch.tensor(h.letter_ids, device=out.logits.device)]
        return torch.nn.functional.cross_entropy(logits, y)

    def _augment(self, ex: Example, rng: random.Random) -> Example:
        aug = self.cfg["augment"]
        if aug["permute_train"]:
            ex = permute_speakers(ex, rng)[0]
        if aug["shuffle_evidence"]:
            ex = shuffle_evidence(ex, rng)
        return ex

    def _eval(self, h, val: Sequence[Example]) -> tuple[float, float]:
        h.model.eval()
        h.memo.clear()  # weights changed since the last scoring
        prompts = [prompt_for(h, ex, self.pcfg, self.lcfg["max_len"]) for ex in val]
        logits = score_prompts(h, prompts, self.lcfg["batch_size"])
        h.memo.clear()
        h.model.train()
        y = np.array([SPEAKERS.index(ex.label) for ex in val])
        from liars.preds import softmax

        p = softmax(logits)
        return float((p.argmax(1) == y).mean()), _nll(p, y)

    def fit(self, train, val, cfg):
        torch = None
        self.cfg = cfg
        self.lcfg, self.pcfg, q = cfg["llm"], cfg["prompt"], cfg["qlora"]
        self.tta_n = cfg["augment"]["tta_n"]
        self.fewshot = []
        refit = cfg["train"].get("_refit") or {}
        h = self._build(cfg)
        torch = self.torch
        model = h.model
        rng = random.Random(cfg["seed"])

        lengths = [len(h.tokenizer(prompt_for(h, ex, self.pcfg, self.lcfg["max_len"]),
                                   add_special_tokens=False)["input_ids"]) for ex in train]
        self.log(f"{len(train)} train / {len(val) if val else 0} val. {length_stats(lengths)}; "
                 f"truncated {h.stats['truncated']}")

        mb, accum = q["micro_batch_size"], q["grad_accum"]
        # whole optimizer steps per epoch; a trailing partial accumulation is dropped
        steps_per_epoch = max(1, math.ceil(len(train) / mb) // accum)
        total_steps = refit.get("steps") or math.ceil(q["epochs"] * steps_per_epoch)
        if q.get("max_steps"):
            total_steps = min(total_steps, q["max_steps"])
        params = [p for p in model.parameters() if p.requires_grad]
        if h.device == "cuda" and q["optimizer"] == "paged_adamw_8bit":
            import bitsandbytes as bnb

            opt = bnb.optim.PagedAdamW8bit(params, lr=q["lr"], weight_decay=q["weight_decay"])
        else:
            opt = torch.optim.AdamW(params, lr=q["lr"], weight_decay=q["weight_decay"])
        from transformers import get_cosine_schedule_with_warmup

        sched = get_cosine_schedule_with_warmup(opt, int(q["warmup_ratio"] * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=h.device == "cuda")
        eval_every = max(1, int(q["eval_every_frac"] * steps_per_epoch))
        self.log(f"steps: {total_steps} total, {steps_per_epoch}/epoch (micro batch {mb} x accum "
                 f"{accum}), eval every {eval_every}" + (f", refit ({refit})" if refit else ""))

        model.train()
        best = {"acc": -1.0, "loss": float("inf"), "step": 0, "state": None}
        bad_evals, step, micro, skipped, t0 = 0, 0, 0, 0, time.time()
        self.history = []
        done = False
        epoch = 0
        while not done:
            order = list(train)
            rng.shuffle(order)
            for b in range(0, len(order), mb):
                batch = [self._augment(ex, rng) for ex in order[b : b + mb]]
                loss = self._batch_loss(h, batch)
                if not torch.isfinite(loss):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / accum).backward()
                micro += 1
                if micro % accum:
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, q["max_grad_norm"])
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped += 1  # inf/NaN gradients: fp16 overflow, step skipped
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step == q["time_probe_steps"]:
                    sec = (time.time() - t0) / step
                    est_epoch = sec * steps_per_epoch / 3600
                    est_total = sec * total_steps / 3600
                    self.log(f"timing: {sec:.1f} s/step -> {est_epoch:.2f} h per epoch, "
                             f"{est_total:.2f} h for {total_steps} steps (+ eval/predict)")
                    if skipped > q["time_probe_steps"] * q["max_skipped_frac"]:
                        raise RuntimeError(
                            f"{skipped} of the first {step} steps had inf/NaN gradients (fp16 "
                            "overflow): this model does not train in fp16 here; try another base")
                    if est_total > q["max_gpu_hours"]:
                        raise RuntimeError(
                            f"estimated {est_total:.1f} GPU h > qlora.max_gpu_hours "
                            f"{q['max_gpu_hours']}: stopping (shrink epochs/max_len or raise the limit)")
                if val and (step % eval_every == 0 or step == total_steps):
                    acc, vloss = self._eval(h, val)
                    self.history.append({"step": step, "val_acc": acc, "val_loss": vloss,
                                         "lr": sched.get_last_lr()[0], "skipped": skipped,
                                         "minutes": round((time.time() - t0) / 60, 1)})
                    self.log(f"step {step}/{total_steps} val acc {acc:.4f} loss {vloss:.4f} "
                             f"(skipped {skipped}, {(time.time() - t0) / 60:.1f} min)")
                    if (acc, -vloss) > (best["acc"], -best["loss"]):
                        best = {"acc": acc, "loss": vloss, "step": step,
                                "state": {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                                          if p.requires_grad}}
                        bad_evals = 0
                    else:
                        bad_evals += 1
                        if bad_evals >= q["patience"]:
                            self.log(f"early stop at step {step} (best step {best['step']})")
                            done = True
                if step >= total_steps:
                    done = True
                if done:
                    break
            opt.zero_grad(set_to_none=True)  # drop a partial accumulation at epoch end
            micro = 0
            epoch += 1
            if epoch >= math.ceil(q["epochs"]) and not refit:
                done = True
        if val and (not self.history or self.history[-1]["step"] != step):
            acc, vloss = self._eval(h, val)  # final weights are a candidate too
            self.history.append({"step": step, "val_acc": acc, "val_loss": vloss,
                                 "lr": sched.get_last_lr()[0], "skipped": skipped,
                                 "minutes": round((time.time() - t0) / 60, 1)})
            self.log(f"step {step} (final) val acc {acc:.4f} loss {vloss:.4f}")
            if (acc, -vloss) > (best["acc"], -best["loss"]):
                best = {"acc": acc, "loss": vloss, "step": step, "state": None}

        if best["state"] is not None:
            named = dict(model.named_parameters())
            with torch.no_grad():
                for n, v in best["state"].items():
                    named[n].copy_(v.to(named[n].device))
        self.best_step = best["step"] if best["state"] is not None else step
        self.train_minutes = round((time.time() - t0) / 60, 1)
        self.skipped_steps = skipped
        model.eval()
        h.memo.clear()
        self.h = h
        self.temperature = refit.get("temperature", 1.0)
        if val:
            logits = self._tta_logits(val)
            y = np.array([SPEAKERS.index(ex.label) for ex in val])
            self.temperature = min(self.lcfg["temperatures"], key=lambda t: _nll(self._probs(logits, t), y))
        save_dir = cfg["train"].get("_save_dir")
        if save_dir:
            self.save(Path(save_dir))

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.h.model.save_pretrained(str(path))
        meta = {"base_model": self.lcfg["model_id"], "temperature": self.temperature,
                "best_step": self.best_step, "answer_prefix": self.pcfg.get("answer_prefix", ""),
                "history": getattr(self, "history", [])}
        (path / META_FILE).write_text(json.dumps(meta, indent=1))
        self.log(f"saved adapters -> {path}")

    @classmethod
    def load(cls, path: Path, cfg: dict) -> "LLMQLoRAModel":
        """Base model + saved adapters, ready for predict_proba."""
        from peft import PeftModel

        m = cls()
        m.cfg, m.lcfg, m.pcfg = cfg, cfg["llm"], cfg["prompt"]
        m.tta_n = cfg["augment"]["tta_n"]
        m.fewshot = []
        meta = json.loads((Path(path) / META_FILE).read_text())
        h = load_lm(m.lcfg)
        h.model = PeftModel.from_pretrained(h.model, str(path)).eval()
        set_answer_prefix(h, m.pcfg.get("answer_prefix", ""))
        m.h, m.temperature, m.best_step = h, meta["temperature"], meta["best_step"]
        return m

    def refit_info(self) -> dict:
        return {"steps": max(1, self.best_step), "temperature": self.temperature}

    def report(self) -> str:
        base = super().report().replace("# LLM zero-shot report", "# QLoRA report")
        extra = [("best step", self.best_step), ("train minutes", getattr(self, "train_minutes", "?")),
                 ("skipped steps (fp16 overflow / NaN)", getattr(self, "skipped_steps", "?"))]
        lines = [base] + [f"| {k} | {v} |" for k, v in extra]
        hist = getattr(self, "history", [])
        if hist:
            lines += ["", "| step | val acc | val loss | lr | skipped | minutes |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['step']} | {r['val_acc']:.4f} | {r['val_loss']:.4f} | {r['lr']:.2e} | "
                      f"{r['skipped']} | {r['minutes']} |" for r in hist]
        return "\n".join(lines)
