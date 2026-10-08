# Project Instructions

## Prefer code parameters
- Configure Python behavior through explicit function or constructor parameters; use directly editable, representative literal values in sample entrypoints.
- Do not add `argparse`, `sys.argv`, environment variables, or CLI flags for configuration unless the user explicitly requests a CLI.
- When removing an unwanted CLI, remove parser helpers, imports, usage text, and parser-specific tests while preserving the callable API (follow the test-edit policy below).
- Point out sample configuration values in the handoff.