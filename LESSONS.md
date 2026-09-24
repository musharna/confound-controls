# Lessons

- 2026-09-24: mutmut guessed source_paths from the cwd basename; a test that spawns a child Python inside mutants/ re-ran the guess there ("mutants"), crashed, and the nightly mutation job checked 0 mutants for 6 nights. `[tool.mutmut] source_paths` is now explicit. Caught by: the nightly gate's 0-checked check.
