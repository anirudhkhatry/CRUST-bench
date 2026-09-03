Your test did not satisfy the gate. Fix it and return the complete file again.

## What happened

<<FEEDBACK>>

## Your previous test

```c
<<PREV>>
```

Remember the gate: the test must **exit 0 on the original library** and **exit non-zero
on the mutant**, reaching the fault only through the public API, asserting concrete
literal values derived from the original code's behaviour.

If your expected values were wrong, recompute them from the original source rather than
loosening the assertion — a weakened assertion passes both versions and kills nothing.
If the mutant is genuinely unobservable through the public API on any input, say so in
one line prefixed `EQUIVALENT:` instead of returning code.

Output the complete corrected C file in one fenced block, nothing else.
