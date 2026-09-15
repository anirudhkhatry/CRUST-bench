Refactor the target file below into smaller, well-structured modules while preserving all
externally observable behaviour.

## Target file

<<TARGET>>

## The source tree

<<SOURCES>>

## Requirements

- Preserve every export the target file currently provides, with the same names and the
  same behaviour. Callers must not have to change.
- Split the target into focused modules; import them back into the target file.
- Do not change behaviour, including edge cases, error messages, ordering and defaults.

## Output

Return one fenced `json` block mapping each file path (relative to `src/`) to its complete
new contents. Include every file you created or modified, and no others.
