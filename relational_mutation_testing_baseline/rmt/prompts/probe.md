You are writing a **C characterisation test** for a library.

We have a C library `<<PROJECT>>` and several independently written ports of it. A
characterisation test pins down what the original does so the ports can be checked
against it: it runs a fixed workload through the public API and *records* every result
with the recording functions below, instead of asserting. The same test, translated
mechanically, is then run against each port and the recorded values are compared.
Write one C file with a `main()` that runs on the original library and exits 0.

## The library

<<SOURCES>>

## Where to focus

An existing unit test for `<<FUNCTION>>` (it checks one `<<FAULT_CLASS>>` corner case):

```c
<<KILL_TEST>>
```

Use it as the seed: keep its scenario, but broaden it into a workload that exercises
that operation and its neighbours at several scales, and covers the rest of the API as
well. A port that gets *this* right may still be wrong nearby.

## Hard constraints

1. **Public API only.** Call the library the way a normal caller would, through the
   entry points below. No internal helpers, no copied library code, no `#include` of a
   `.c` file.

   The ports implement the **Rust interface below** (bodies are `unimplemented!()` on
   purpose). Match its methods to the C functions of the same meaning, and shape the
   workload around operations whose *results are visible through that interface*: a C
   call whose Rust counterpart returns nothing and has no getter cannot be compared, so
   recording it only produces holes. Where C exposes a struct field the Rust interface
   also makes `pub`, that field is fair game. Opaque handles are recorded as
   `!= NULL` flags only.

<<IFACE>>

2. **Record with the three recording functions**, which are already defined for you
   (do not declare or define them):

   ```c
   void obs_ll (const char *label, long long v);    /* integers, booleans, counts, status */
   void obs_dbl(const char *label, double v);       /* floating point */
   void obs_str(const char *label, const char *s);  /* NUL-terminated strings (NULL is fine) */
   ```

   - **Labels are unique, printable ASCII, and hierarchical**: `scenario/step/what`,
     e.g. `insert_100/len`, `lookup/absent_key_7/found`, `encode/int64/neg_min/byte3`.
     In a loop, put the index in the label. Two records may never share a label.
   - **Record between 40 and 300 values.** Prefer many small facts over a few large
     ones; each is compared independently.
   - **Status and error paths are recorded, not asserted**: `obs_ll("open/missing_file/ok",
     rc == 0)` — record `1` for success and `0` for failure, never the raw error code.
   - **Record behaviour, never representation.** Do NOT record pointers or addresses,
     allocated sizes or capacities, memory footprints, buffer lengths that are not part
     of the contract, hash values, iteration order of hash-based containers (unless the
     API defines the order — sort first if it does not), uninitialised bytes, or
     anything that depends on the platform, the allocator or word size. A port is free
     to differ on all of these; recording them makes the test wrong, not stricter.
   - Floating point: record the value as is; the comparison is tolerant.

3. **Workload shape.** Several scales — a handful of elements, then tens, then a few
   hundred if the library is fast enough — plus boundary inputs (empty, single element,
   largest representable, zero, negative, exact powers of two where sizes matter) and
   at least a few error-path calls. Realistic sequences of operations, not one call per
   function.

4. **Deterministic and fast.** No randomness (a fixed LCG in the test is fine), no
   time, no network, no reading files you did not create. Runs in under 5 seconds.

5. **Exit 0.** `return 0;` from `main`. You may use `assert` for your own setup
   sanity, but a test that aborts recorded nothing useful.

6. **Self-contained.** One C file, includes the library's public headers by the path
   shown above (e.g. `#include "src/foo.h"`).

## Output

The complete C file in one fenced block, nothing else:

```c
#include <string.h>
...
int main(void) { ...; return 0; }
```
