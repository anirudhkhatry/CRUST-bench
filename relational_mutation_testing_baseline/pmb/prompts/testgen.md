You are writing a **pytest test that kills a mutant**.

We have a Python module from `<<REPO>>` and a single-fault mutant of it. Write one test
file that:

- **passes on the original module**, and
- **fails on the mutant**.

## The module under test

File: `<<FILE>>`, which uses the library **`<<SOURCE>>`**.

```python
<<SOURCE_CODE>>
```

## The mutant

Fault class: `<<FAULT_CLASS>>` in `<<FUNCTION>>`
Rationale: <<RATIONALE>><<HAZARD_BLOCK>>

```diff
<<DIFF>>
```

## How to import it

The harness places a generated shim next to your test. Import the module under test
**only** through it:

```python
import pmb_target                      # the module object is `pmb_target.module`
from pmb_target import some_function    # every public name is re-exported
```

Do not import the module by its path, and do not add it to `sys.path` yourself: the
post-migration version of this file often lives somewhere else, and the shim is the only
import that works on both sides.

## Hard constraints

1. **Reach the fault through the module's public callables.** Call the module the way a
   normal caller would. Do not copy module code into the test and do not re-implement the
   logic you are checking — a test that recomputes the answer alongside the module passes
   both versions.
2. **Assert concrete values, not properties.** Write `assert n == 7`, not `assert n > 0`.
   Every expected value must be a literal you derived by reasoning about the *original*
   code. Properties both versions satisfy kill nothing.
3. **No real I/O.** No network, no sleeping, no clock, no randomness, no reading or
   writing files outside `tmp_path`. Where the code under test performs I/O, stub it with
   `unittest.mock.patch` or `monkeypatch` **at the boundary the module calls** — patch the
   `<<SOURCE>>` entry point the module uses, and give it a fixed canned result — then
   assert on what the module *did with* that result. That logic, not the transport, is
   what the mutant changed.
4. **Do not assert on the mock.** `mock.assert_called_with(...)` checks how the module
   talks to the library; the library is exactly the thing that is about to be replaced, so
   such an assertion is not a statement about behaviour and will not survive. Assert on
   the module's return value, on its raised exception, or on the state it leaves behind.
5. **No tautologies.** Do not assert something trivially true, do not compare a value to
   itself, and do not `pytest.fail()` unconditionally.
6. **Deterministic and fast.** Fixed inputs, well under a second, no dependence on test
   order.
7. **One `def test_kill()` function** (helpers and fixtures above it are fine). A failed
   `assert`, or an unexpected exception, is the kill signal.

## Output

The complete test file in one fenced block, nothing else:

```python
import pytest
import pmb_target

def test_kill():
    ...
```

If the mutant is genuinely unobservable from outside the module on any input, say so in
one line prefixed `EQUIVALENT:` instead of returning code.
