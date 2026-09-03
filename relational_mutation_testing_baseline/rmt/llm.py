"""Thin Anthropic client for the relational pipeline.

Deliberately small: one blocking call, adaptive thinking, streaming (Stage 2/3
outputs are long), and a content-addressed disk cache so a rerun after a crash
does not re-pay for prompts that already succeeded.

Thread-safe: the driver runs many mutants concurrently, so cache writes go through a
temp file and an atomic rename, and a semaphore caps how many requests are in flight.

The key comes from /nas/secrets.env (JSON, `anthropic_key`) unless ANTHROPIC_API_KEY
is already set.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import anthropic

MODEL = os.environ.get("RMT_MODEL", "claude-opus-5")
CACHE = Path(os.environ.get("RMT_CACHE",
                            Path(__file__).resolve().parent.parent / "results/.llm_cache"))
SECRETS = Path("/nas/secrets.env")

# Cap concurrent requests so a wide fan-out does not just trade queueing for 429s.
MAX_INFLIGHT = int(os.environ.get("RMT_MAX_INFLIGHT", "16"))
_gate = threading.Semaphore(MAX_INFLIGHT)
_lock = threading.Lock()

# USD per million tokens. Override with RMT_PRICE_IN / RMT_PRICE_OUT if the rate card
# moves; the token counts in the ledger are measured, the dollars are only this rate
# applied to them.
PRICE_IN = float(os.environ.get("RMT_PRICE_IN", "5"))
PRICE_OUT = float(os.environ.get("RMT_PRICE_OUT", "25"))

# Which stage a call belongs to, by the tag prefix its caller passes.
STAGE_OF = {"mut": "stage1_mutate", "tst": "stage2_testgen", "tr": "stage3_transpile",
            "rmut": "stage4_rust_mutant"}
_ledger = []


def _record(tag, usage, cached):
    """One line in the per-stage cost ledger. Cache hits are recorded too, with the
    usage the call originally cost: a resumed run should still be able to report what
    the pipeline costs, not merely what this invocation paid for."""
    with _lock:
        _ledger.append({
            "stage": STAGE_OF.get(tag.split("_")[0], "other"),
            "tag": tag, "cached": cached,
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens") or 0,
            "cache_creation": usage.get("cache_creation_input_tokens") or 0,
        })


def cost_ledger():
    """Per-stage totals, plus the whole-run total. `paid_usd` excludes cache hits."""
    with _lock:
        rows = list(_ledger)
    by = {}
    for r in rows:
        s = by.setdefault(r["stage"], {"calls": 0, "cache_hits": 0, "input": 0,
                                       "output": 0, "usd": 0.0, "paid_usd": 0.0})
        s["calls"] += 1
        s["cache_hits"] += r["cached"]
        s["input"] += r["input"] + r["cache_read"] + r["cache_creation"]
        s["output"] += r["output"]
        usd = ((r["input"] + r["cache_read"] + r["cache_creation"]) * PRICE_IN
               + r["output"] * PRICE_OUT) / 1e6
        s["usd"] += usd
        if not r["cached"]:
            s["paid_usd"] += usd
    total = {"calls": sum(s["calls"] for s in by.values()),
             "cache_hits": sum(s["cache_hits"] for s in by.values()),
             "input": sum(s["input"] for s in by.values()),
             "output": sum(s["output"] for s in by.values()),
             "usd": sum(s["usd"] for s in by.values()),
             "paid_usd": sum(s["paid_usd"] for s in by.values())}
    return {"model": MODEL, "price_in": PRICE_IN, "price_out": PRICE_OUT,
            "by_stage": by, "total": total}


def _key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    return json.loads(SECRETS.read_text())["anthropic_key"]


_client = None


def client():
    global _client
    with _lock:
        if _client is None:
            # Pin the base URL: the shell exports ANTHROPIC_BASE_URL for a local proxy
            # that does not accept the sk-ant- key from secrets.env.
            _client = anthropic.Anthropic(
                api_key=_key(),
                base_url=os.environ.get("RMT_BASE_URL", "https://api.anthropic.com"),
                max_retries=4, timeout=1800.0,
            )
    return _client


RETRYABLE = {"overloaded_error", "rate_limit_error", "api_error", "timeout_error"}


def _retryable(e):
    """Whether a failed call is worth trying again.

    The status code alone is not enough: a stream that has already returned 200 reports
    an overload as an SSE error event, so the HTTP status says nothing and only the
    error type in the body does. Reading just the status dropped those calls on the
    floor, and a dropped call kills the mutant it belonged to.
    """
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if (getattr(e, "status_code", 0) or 0) >= 429:
        return True
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and err.get("type") in RETRYABLE


def ask(system, user, effort="high", max_tokens=16000, temperature=None, tag="x"):
    """One turn. Returns the concatenated text blocks (thinking blocks dropped)."""
    payload = {"model": MODEL, "system": system, "user": user, "effort": effort,
               "max_tokens": max_tokens}
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = CACHE / f"{tag}_{h}.json"
    if hit.exists():
        blob = json.loads(hit.read_text())
        _record(tag, blob.get("usage") or {}, cached=True)
        return blob["text"]

    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    )
    last = None
    for attempt in range(5):
        try:
            with _gate:
                with client().messages.stream(**kwargs) as s:
                    msg = s.get_final_message()
            break
        except (anthropic.APIStatusError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            # Overload, rate limit, transport: back off and retry. Anything else --
            # a bad kwarg, a bad key -- is our bug and must surface immediately rather
            # than costing five backoffs per call across the whole run.
            if not _retryable(e):
                raise
            last = e
            time.sleep(5 * (attempt + 1) ** 2)
    else:
        raise RuntimeError(f"LLM call failed after retries: {last}")

    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = msg.usage.model_dump()
    _record(tag, usage, cached=False)
    # Atomic: concurrent workers may race on the same key, and a half-written cache
    # entry would poison every later run.
    tmp = hit.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"payload": payload, "text": text, "usage": usage},
                              indent=1))
    tmp.replace(hit)
    return text


def fenced(text, lang=None):
    """Extract the last fenced code block, preferring one tagged `lang`."""
    blocks, cur, tag = [], None, None
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if cur is None:
                cur, tag = [], line.lstrip()[3:].strip()
            else:
                blocks.append((tag, "\n".join(cur)))
                cur = None
            continue
        if cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append((tag, "\n".join(cur)))
    if not blocks:
        return None
    if lang:
        for t, b in reversed(blocks):
            if t and t.lower().startswith(lang):
                return b
    return blocks[-1][1]


def json_block(text):
    """Extract a JSON object/array from a reply, tolerating prose around it."""
    b = fenced(text, "json")
    for cand in ([b] if b else []) + [text]:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = cand.find(opener), cand.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON in reply: {text[:400]}")
