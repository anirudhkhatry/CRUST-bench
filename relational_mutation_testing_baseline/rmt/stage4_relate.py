"""Stage 4 -- replay the C mutation on the Rust side, and ask the study's question.

The reference implementation M is assumed correct. We apply the *same fault* to it,
producing M', and the whole experiment reduces to one question per mutant:

    does RT pass M and fail M' ?

The fault is ported from the mutant's language-agnostic `semantic_change` description,
not from the C diff -- the Rust code is an independent re-implementation and shares no
syntax with the C, so the only transferable thing is the described behaviour.
"""
from . import rbench as R
from .llm import ask, json_block
from .prompts import load as load_prompt
from .stage1_mutate import edits_of

SYSTEM = ("You are a precise Rust programmer introducing a described fault into an "
          "implementation. You make the smallest edit that reproduces it and follow "
          "output formats exactly.")


def apply_patch(impl_dir, rec):
    """Return {relpath: mutated_text} or raise ValueError with the rejection reason.

    A fault with several sites in the C original usually has several here too, so this
    takes a list of edits and applies them all or none.
    """
    files = R.impl_files(impl_dir)
    patch, changed = {}, False
    for e in edits_of(rec):
        rel = e.get("file", "")
        if rel not in files:
            cands = [f for f in files if f.endswith(rel.lstrip("./"))]
            if len(cands) != 1:
                raise ValueError(f"unknown file {rel!r}")
            rel = cands[0]
        text = patch.get(rel) or (impl_dir / rel).read_text(errors="ignore")
        old, new = e.get("original_snippet", ""), e.get("mutated_snippet", "")
        if not old:
            raise ValueError("empty original_snippet")
        n = text.count(old)
        if n == 0:
            raise ValueError(f"original_snippet not found verbatim in {rel}")
        if n > 1:
            raise ValueError(f"original_snippet occurs {n} times in {rel}, not unique")
        changed |= old != new
        patch[rel] = text.replace(old, new)
    if not patch:
        raise ValueError("no edits")
    if not changed:
        raise ValueError("mutated_snippet identical to original")
    return patch


def make_counterpart(impl_dir, verdict, tag, retries=2):
    """Build M' from M for one mutant. Returns (patch|None, record)."""
    user = load_prompt(
        "rust_mutate",
        fault_class=verdict.get("fault_class", "?"),
        function=verdict.get("function", "?"),
        reachable_via=verdict.get("reachable_via", "?"),
        semantic_change=verdict.get("semantic_change", ""),
        observable_effect=verdict.get("observable_effect", ""),
        forbidden=("tests or `src/lib.rs` (the `src/interfaces/` files shown above are "
                   "this crate's implementation, pulled in by `include!`, and may be "
                   "edited)" if R.includes_interfaces(impl_dir)
                   else "tests, `src/interfaces/`, or `src/lib.rs`"),
        impl=R.impl_source(impl_dir))

    trail = []
    for i in range(retries + 1):
        # 32k, not 16k: adaptive thinking on a large implementation can spend the whole
        # budget reasoning and return no text at all, and an empty reply used to take
        # the mutant down as an exception rather than costing one retry.
        reply = ask(SYSTEM, user, effort="high", max_tokens=32000, tag=f"rmut_{tag}_{i}")
        try:
            rec = json_block(reply)
        except ValueError as e:
            trail.append({"attempt": i, "outcome": "no_json", "detail": str(e)[:200]})
            user += ("\n\n---\n\nThat reply contained no JSON object. Answer with the "
                     "JSON object only, and keep it short.")
            continue
        if isinstance(rec, list):
            rec = rec[0] if rec else {}
        if rec.get("applicable") is False:
            trail.append({"attempt": i, "outcome": "not_applicable",
                          "note": rec.get("note", "")[:400]})
            return None, {"applied": False, "reason": "not_applicable",
                          "note": rec.get("note", "")[:400], "trail": trail}
        try:
            patch = apply_patch(impl_dir, rec)
        except ValueError as e:
            trail.append({"attempt": i, "outcome": "apply_failed", "detail": str(e)})
            user += (f"\n\n---\n\nThat patch could not be applied: {e}. "
                     "Re-read the implementation above and copy `original_snippet` "
                     "verbatim from it. Return the JSON object again.")
            continue

        build = R.run_test(impl_dir, None, patch=patch, tag=f"rb{i}")
        if build["status"] != "green":
            trail.append({"attempt": i, "outcome": "build_error"})
            user += ("\n\n---\n\nThat patch does not compile:\n```\n"
                     + build["out"][-2500:] +
                     "\n```\nFix it, keeping the same fault. Return the JSON object again.")
            continue

        trail.append({"attempt": i, "outcome": "ok"})
        return patch, {"applied": True, "reason": "ok",
                       "file": ", ".join(sorted(patch)), "n_sites": len(edits_of(rec)),
                       "note": rec.get("note", "")[:400],
                       "edits": [{k: (v or "")[:600] for k, v in e.items()}
                                 for e in edits_of(rec)],
                       "trail": trail}

    return None, {"applied": False, "reason": trail[-1]["outcome"] if trail else "failed",
                  "trail": trail}


def trial(project, test_rs, verdict, tag, impl_dir=None):
    """The relational trial for one mutant: RT on M, and RT on M'.

    Verdict vocabulary, which is the study's actual result column:
      killed        -- RT passes M and fails M'. The pipeline worked end to end.
      not_killed    -- RT passes both. The test lost its discriminating power.
      unsound       -- RT fails M. The transpiled test is wrong about correct Rust.
      no_counterpart-- the fault could not be reproduced in Rust at all.
      error         -- RT does not build against M, or a run timed out.

    `impl_dir` overrides the reference: the driver hands in an alternate
    ground-truth-passing implementation when RT fails the codex reference but passes
    other correct implementations, i.e. when the evidence says the reference, not the
    test, is the odd one out. The record then carries `reference` so the swap is visible.
    """
    swapped = impl_dir is not None
    impl_dir = impl_dir or R.reference(project)
    if impl_dir is None:
        return {"verdict": "error", "detail": "no reference implementation"}
    on_m = R.run_test(impl_dir, test_rs, tag=f"{tag}_m")
    if on_m["status"] == "build_error":
        return {"verdict": "error", "detail": "RT does not build against M",
                "on_m": on_m["status"], "out": on_m["out"][-1500:]}
    if on_m["status"] == "timeout":
        return {"verdict": "error", "detail": "RT timed out on M", "on_m": "timeout"}
    if on_m["status"] != "green":
        return {"verdict": "unsound", "detail": "RT fails the assumed-correct M",
                "on_m": on_m["status"], "out": on_m["out"][-1500:]}

    patch, mrec = make_counterpart(impl_dir, verdict, tag)
    if patch is None:
        return {"verdict": "no_counterpart", "on_m": "green", "rust_mutant": mrec}

    on_mp = R.run_test(impl_dir, test_rs, patch=patch, tag=f"{tag}_mp")
    if on_mp["status"] == "build_error":
        return {"verdict": "error", "detail": "RT does not build against M'",
                "on_m": "green", "on_mp": "build_error", "rust_mutant": mrec}
    killed = on_mp["status"] in ("test_failed", "timeout")
    # Control: does the RBench ground-truth suite itself detect M'? This separates
    # "RT is weak" from "M' is barely different", and gives the Stage 5 comparison
    # (RT-only / GT-only / both / neither) for free.
    gt = R.run_gt(impl_dir, patch=patch, tag=f"{tag}_gt")
    rec = {"verdict": "killed" if killed else "not_killed",
           "on_m": "green", "on_mp": on_mp["status"],
           "gt_on_mp": gt["status"], "gt_kills_mp": gt["status"] == "test_failed",
           "rust_mutant": mrec, "out": on_mp["out"][-1500:] if killed else ""}
    if swapped:
        rec["reference"] = str(impl_dir)
    return rec
