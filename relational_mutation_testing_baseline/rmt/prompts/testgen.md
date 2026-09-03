You are writing a **C test that kills a mutant**.

We have a C library `<<PROJECT>>` and a single-edit mutant of it. Write one C file with a
`main()` that:

- **exits 0 on the original library**, and
- **exits non-zero on the mutant**.

## The library

<<SOURCES>>

## The mutant

Fault class: `<<FAULT_CLASS>>` in `<<FUNCTION>>` (`<<FILE>>`)
Rationale: <<RATIONALE>>

```diff
<<DIFF>>
```

## Hard constraints

1. **Reach the fault through the public API.** Call the library the way a normal caller
   would, through the entry points below. Do not call internal helpers directly, do not
   copy library code into the test, and do not `#include` a `.c` file. A test that pokes
   at internals is worthless to us.

   Public entry points this library is expected to expose (these are the names of the
   corresponding Rust API, so match them to the C functions of the same meaning):

<<IFACE>>

2. **Assert concrete values, not properties.** Write `assert(len == 7)`, not
   `assert(len > 0)`. Every expected value must be a literal you derived by reasoning
   about the *original* code. Properties that both versions satisfy do not kill anything.
3. **No tautologies.** Do not assert something trivially true, do not compare a value to
   itself, and do not hard-code a `return 1`.
4. **Deterministic.** No randomness, no time, no reading files, no network, no printing
   that matters. Fixed inputs only.
5. **Self-contained and fast.** One file, includes the library's public headers by the
   path shown above (e.g. `#include "src/foo.h"`), runs in well under a second.
6. **Signal failure by exit status.** `return 0;` at the end of `main` for pass; any
   non-zero exit (including an `assert` abort) counts as a kill. You may `printf`
   diagnostics to stderr.

## Output

The complete C file in one fenced block, nothing else:

```c
#include <assert.h>
...
int main(void) { ...; return 0; }
```
