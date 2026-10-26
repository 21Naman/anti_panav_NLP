"""Shared open-LLM plumbing: load (4-bit on CUDA), chat-template prompts, letter scoring.

Scoring never parses free text: we read the logits of the 4 letter tokens at the answer
position. The letter token ids are found by tokenising a rendered prompt with and
without the letter appended, so they match exactly what the chat template produces.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from liars.device import get_device
from liars.prompt import fit_to_budget
from liars.schema import SPEAKERS, Example


@dataclass
class LMHandle:
    model_id: str
    model: object
    tokenizer: object
    device: str
    template_kwargs: dict
    letter_ids: list[int] = field(default_factory=list)
    memo: dict = field(default_factory=dict)   # prompt hash -> raw letter logits
    stats: dict = field(default_factory=lambda: {"n_scored": 0, "seconds": 0.0, "nan_rows": 0,
                                                 "truncated": 0, "max_mem_gb": 0.0})


def load_lm(lcfg: dict, for_training: bool = False) -> LMHandle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device(lcfg["device"])
    model_id = lcfg["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict = {}
    if device == "cuda":
        kwargs["dtype"] = torch.float16  # T4: no bf16
        kwargs["device_map"] = {"": 0}
        if lcfg["load_in_4bit"]:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=lcfg.get("double_quant", True),
            )
    else:
        kwargs["dtype"] = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:  # multimodal checkpoints (…ForConditionalGeneration)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    if not for_training:
        model.eval()
    h = LMHandle(model_id, model, tok, device, dict(lcfg.get("chat_template_kwargs") or {}))
    h.letter_ids = letter_token_ids(h, render(h, "Answer with one letter."))
    return h


def render(h: LMHandle, user_text: str, fewshot: Sequence[tuple[str, str]] = (),
           system: str | None = None, answer_prefix: str = "") -> str:
    """Chat-template prompt ending right where the answer letter goes."""
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in fewshot:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user_text})
    try:
        text = h.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               **h.template_kwargs)
    except Exception:  # some multimodal templates want list-of-parts content
        parts = [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in msgs]
        text = h.tokenizer.apply_chat_template(parts, tokenize=False, add_generation_prompt=True,
                                               **h.template_kwargs)
    return text + answer_prefix


def _ids(h: LMHandle, text: str) -> list[int]:
    return h.tokenizer(text, add_special_tokens=False)["input_ids"]


def letter_token_ids(h: LMHandle, prompt: str) -> list[int]:
    base = _ids(h, prompt)
    out = []
    for letter in SPEAKERS:
        ids = _ids(h, prompt + letter)
        if ids[: len(base)] == base and len(ids) == len(base) + 1:
            out.append(ids[-1])
        else:  # boundary merge: fall back to the bare letter token
            alt = _ids(h, letter)
            if len(alt) != 1:
                raise ValueError(f"{h.model_id}: letter {letter!r} is not a single token")
            out.append(alt[0])
    if len(set(out)) != len(SPEAKERS):
        raise ValueError(f"{h.model_id}: letter tokens not distinct: {out}")
    return out


def prompt_for(h: LMHandle, ex: Example, pcfg: dict, max_len: int,
               fewshot: Sequence[tuple[str, str]] = ()) -> str:
    """Rendered prompt for one example, truncated to max_len tokens."""
    overhead = len(_ids(h, render(h, "", fewshot, pcfg.get("system"), pcfg.get("answer_prefix", ""))))
    user, dropped = fit_to_budget(ex, pcfg, lambda t: len(_ids(h, t)) + overhead, max_len)
    if dropped:
        h.stats["truncated"] += 1
    return render(h, user, fewshot, pcfg.get("system"), pcfg.get("answer_prefix", ""))


def score_prompts(h: LMHandle, prompts: Sequence[str], batch_size: int,
                  progress: bool = False) -> np.ndarray:
    """(n, 4) raw letter logits (float32). Non-finite rows become zeros (= uniform)."""
    import torch

    keys = [hashlib.sha1(p.encode("utf-8")).hexdigest() for p in prompts]
    todo = sorted({k: p for k, p in zip(keys, prompts) if k not in h.memo}.items(),
                  key=lambda kv: len(kv[1]))
    if todo:
        from tqdm import tqdm

        t0 = time.time()
        letter_ids = torch.tensor(h.letter_ids)
        for b in tqdm(range(0, len(todo), batch_size), disable=not progress, desc="llm"):
            chunk = todo[b : b + batch_size]
            enc = h.tokenizer([p for _, p in chunk], add_special_tokens=False, padding=True,
                              return_tensors="pt").to(h.model.device)
            with torch.inference_mode():
                try:
                    out = h.model(**enc, logits_to_keep=1)
                except TypeError:
                    out = h.model(**enc)
            logits = out.logits[:, -1, :].float()[:, letter_ids.to(out.logits.device)].cpu().numpy()
            bad = ~np.isfinite(logits).all(axis=1)
            h.stats["nan_rows"] += int(bad.sum())
            logits[bad] = 0.0
            for (k, _), row in zip(chunk, logits):
                h.memo[k] = row
        h.stats["n_scored"] += len(todo)
        h.stats["seconds"] += time.time() - t0
        if h.device == "cuda":
            h.stats["max_mem_gb"] = max(h.stats["max_mem_gb"], torch.cuda.max_memory_allocated() / 2**30)
    return np.stack([h.memo[k] for k in keys]) if keys else np.zeros((0, len(SPEAKERS)))


_LMS: dict[tuple, LMHandle] = {}


def get_lm(lcfg: dict) -> LMHandle:
    """One loaded model per process (folds reuse it)."""
    key = (lcfg["model_id"], lcfg["device"], lcfg["load_in_4bit"])
    if key not in _LMS:
        _LMS.clear()  # free the previous model before loading another
        _LMS[key] = load_lm(lcfg)
    return _LMS[key]
