"""Thin model client for the relational pipeline.

Deliberately small: one blocking call, reasoning at a chosen effort, streaming (Stage 2/3
outputs are long), and a content-addressed disk cache so a rerun after a crash
does not re-pay for prompts that already succeeded.

Two providers, chosen by the model name in RMT_MODEL: Anthropic for `claude-*`, OpenAI
for `gpt-*`. The pipeline is a measurement instrument and the model is the thing being
measured, so everything above this file -- prompts, gates, taxonomy -- is identical
across providers, and only the transport differs. The cache key includes the model, so
two runs never read each other's answers.

Thread-safe: the driver runs many mutants concurrently, so cache writes go through a
temp file and an atomic rename, and a semaphore caps how many requests are in flight.

Keys come from /nas/secrets.env (JSON: `anthropic_key`, `openai-key`) unless the
corresponding environment variable is already set.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import anthropic

MODEL = os.environ.get("RMT_MODEL", "claude-opus-5")
PROVIDER = "openai" if MODEL.startswith("gpt") else "anthropic"
CACHE = Path(os.environ.get("RMT_CACHE",
                            Path(__file__).resolve().parent.parent / "results/.llm_cache"))
SECRETS = Path("/nas/secrets.env")

# Cap concurrent requests so a wide fan-out does not just trade queueing for 429s.
MAX_INFLIGHT = int(os.environ.get("RMT_MAX_INFLIGHT", "16"))
_gate = threading.Semaphore(MAX_INFLIGHT)
_lock = threading.Lock()

# USD per million tokens, per model. Override with RMT_PRICE_IN / RMT_PRICE_OUT if the
# rate card moves; the token counts in the ledger are measured, the dollars are only this
# rate applied to them.
RATES = {"claude-opus-5": (5.0, 25.0), "gpt-5.4-mini": (0.75, 4.5),
         "gpt-5.4": (2.5, 15.0), "gpt-5.5": (5.0, 30.0)}
_rate = RATES.get(MODEL, (5.0, 25.0))
PRICE_IN = float(os.environ.get("RMT_PRICE_IN", _rate[0]))
PRICE_OUT = float(os.environ.get("RMT_PRICE_OUT", _rate[1]))

# Which stage a call belongs to, by the tag prefix its caller passes.
STAGE_OF = {"mut": "stage1_mutate", "tst": "stage2_testgen", "tr": "stage3_transpile",
            "rmut": "stage4_rust_mutant", "nb": "baseline_naive_testgen",
            "prb": "stage5_probe", "tw": "stage5_twin", "twp": "stage5_twin"}
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


def _key(env, field):
    return os.environ.get(env) or json.loads(SECRETS.read_text())[field]


_client = None


def client():
    global _client
    with _lock:
        if _client is None:
            if PROVIDER == "openai":
                import openai
                _client = openai.OpenAI(
                    api_key=_key("OPENAI_API_KEY", "openai-key"),
                    max_retries=4, timeout=1800.0)
            else:
                # Pin the base URL: the shell exports ANTHROPIC_BASE_URL for a local
                # proxy that does not accept the sk-ant- key from secrets.env.
                _client = anthropic.Anthropic(
                    api_key=_key("ANTHROPIC_API_KEY", "anthropic_key"),
                    base_url=os.environ.get("RMT_BASE_URL", "https://api.anthropic.com"),
                    max_retries=4, timeout=1800.0,
                    # Ask for gzip only. The installed httpx2 hands zstd bodies to
                    # `zstandard` with a keyword the installed version does not accept,
                    # so a zstd-encoded reply raises TypeError inside the stream reader
                    # before any of our code sees it -- intermittently, since the server
                    # picks the encoding. Not advertising zstd sidesteps it entirely.
                    default_headers={"accept-encoding": "gzip"},
                )
    return _client


RETRYABLE = {"overloaded_error", "rate_limit_error", "api_error", "timeout_error"}


class Refused(RuntimeError):
    """The request was declined by the API's safety layer (`stop_reason == "refusal"`).

    Not cached and not retried here: the same prompt is refused again far more often
    than not, and the caller knows how to rephrase where a blind retry does not. Seen
    first on Stage 5 probe prompts, where "record the library's behaviour" next to a
    description of an out-of-bounds read looked like exploit-writing to the classifier.
    """


class TransientStreamError(RuntimeError):
    """A stream that ended in `response.failed`, or with no terminal event at all.

    Server-side and worth retrying, unlike a truncated-but-well-formed turn, which is
    returned to the caller as a short reply.
    """


def _retryable(e):
    """Whether a failed call is worth trying again.

    The status code alone is not enough: a stream that has already returned 200 reports
    an overload as an SSE error event, so the HTTP status says nothing and only the
    error type in the body does. Reading just the status dropped those calls on the
    floor, and a dropped call kills the mutant it belonged to.
    """
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError,
                      TransientStreamError)):
        return True
    if type(e).__name__ in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
        return True
    if (getattr(e, "status_code", 0) or 0) >= 429:
        return True
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and err.get("type") in RETRYABLE


def _call_anthropic(system, user, effort, max_tokens, model=None, temperature=None):
    # Adaptive thinking and the effort dial are Claude 5 features; Haiku 4.5 rejects
    # both with a 400. It is in the panel as the weaker migration arm, and asking it to
    # think harder is not what that arm is for, so it simply runs without them.
    model = model or MODEL
    thinks = not model.startswith("claude-haiku")
    extra = ({"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
             if thinks else {})
    # `temperature` CANNOT be forwarded: anthropic 1.5.0 exposes no temperature parameter
    # on either messages.stream() or messages.create(), so passing it raises TypeError.
    # It stays in `ask`'s signature because callers pass it and it enters the cache key
    # (so a sweep still yields distinct draws), but it reaches no API call. Candidate
    # diversity therefore comes from model choice and prompt variation, not temperature.
    # Verified against the installed SDK rather than assumed -- an earlier version of this
    # comment claimed the blocker was adaptive thinking, which was wrong.
    _ = temperature
    with client().messages.stream(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], **extra) as s:
        msg = s.get_final_message()
    return ("".join(b.text for b in msg.content if b.type == "text"),
            msg.usage.model_dump(), msg.stop_reason)


TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def _call_openai(system, user, effort, max_tokens):
    """The same turn through the Responses API.

    `max_output_tokens` there covers reasoning *and* text, as Anthropic's `max_tokens`
    does, so the budgets the stages pass mean the same thing on both sides. Cached input
    is reported as zero in the ledger because OpenAI already counts cached tokens inside
    `input_tokens`, and the ledger adds the two.

    We read the terminal event ourselves instead of calling `get_final_response()`. A turn
    that spends its whole budget on reasoning ends the stream with `response.incomplete`
    (`incomplete_details.reason == 'max_output_tokens'`) and never emits
    `response.completed`, which that helper reports as a bare RuntimeError. That is a
    *truncated reply*, not a transport failure, and it is the same condition the Anthropic
    side already handles: every stage re-asks when a reply will not parse. Raising here
    instead killed the mutant outright -- ~15% of them on gpt-5.4-mini -- so we surface the
    partial text and let the stage decide. Usage is still recorded: those tokens were paid
    for whether or not the reply was usable.
    """
    final = None
    with client().responses.stream(
            model=MODEL, instructions=system, input=user,
            max_output_tokens=max_tokens,
            reasoning={"effort": effort}) as s:
        for ev in s:
            if getattr(ev, "type", "") in TERMINAL_EVENTS:
                final = ev.response
    if final is None:
        raise TransientStreamError("Responses stream ended with no terminal event")
    if getattr(final, "status", None) == "failed":
        raise TransientStreamError(f"response.failed: {getattr(final, 'error', None)}")
    u = getattr(final, "usage", None)
    return (final.output_text or ""), {
        "input_tokens": u.input_tokens if u else 0,
        "output_tokens": u.output_tokens if u else 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }, getattr(final, "status", None)


def ask(system, user, effort="high", max_tokens=16000, temperature=None, tag="x",
        model=None):
    """One turn. Returns the concatenated text blocks (thinking blocks dropped).

    `model` overrides the module default for this call only; `temperature` is forwarded
    for models that run without adaptive thinking. Both enter the cache key ONLY when
    given, so every previously cached call keeps its key and the existing cache stays warm.
    """
    payload = {"model": model or MODEL, "system": system, "user": user, "effort": effort,
               "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = CACHE / f"{tag}_{h}.json"
    if hit.exists():
        blob = json.loads(hit.read_text())
        _record(tag, blob.get("usage") or {}, cached=True)
        return blob["text"]

    call = _call_openai if PROVIDER == "openai" else _call_anthropic
    last = None
    for attempt in range(5):
        try:
            with _gate:
                text, usage, stop = (
                    call(system, user, effort, max_tokens, model, temperature)
                    if PROVIDER != "openai" else
                    call(system, user, effort, max_tokens))
            break
        except Exception as e:
            # Overload, rate limit, transport: back off and retry. Anything else --
            # a bad kwarg, a bad key -- is our bug and must surface immediately rather
            # than costing five backoffs per call across the whole run.
            if not _retryable(e):
                raise
            last = e
            time.sleep(5 * (attempt + 1) ** 2)
    else:
        raise RuntimeError(f"LLM call failed after retries: {last}")

    _record(tag, usage, cached=False)
    if stop == "refusal":
        # Logged with usage so input-side (0 output tokens) and mid-stream refusals
        # can be told apart afterwards; neither is cached.
        with _lock, (CACHE / "refusals.log").open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')}\t{tag}\t{h}\t"
                    f"out={usage.get('output_tokens', 0)}\ttext={len(text)}\n")
        raise Refused(f"{tag}: request refused by the API safety layer "
                      f"(after {usage.get('output_tokens', 0)} output tokens)")
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
