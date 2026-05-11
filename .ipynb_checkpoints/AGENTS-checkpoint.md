# Project Rules

- Keep Qualtran resource-estimation notebooks data-free when only data size matters: use shape objects such as `qualtran.symbolics.Shaped` instead of concrete ROM data.
- For headless notebook or plotting validation, set `MPLCONFIGDIR` to a writable temporary directory when the home Matplotlib config directory is not writable.
- Preserve unrelated dirty work in the repository. Do not revert deleted, modified, or untracked files unless the user explicitly requests it.
- Prefer focused validation commands for notebooks: parse the notebook JSON, compile executable cells, and run representative resource-counting paths.
