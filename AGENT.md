# Agent Rules

## Allowed
- Edit source files.
- Add or update tests.
- Improve documentation.
- Refactor small, isolated areas.
- Create small commits after successful changes.

## Required
- Read GOALS.md, PLAN.md, AUTONOMOUS_LOG.md, and AUTONOMOUS_INBOX.md before each task.
- Prefer tasks that are safe, useful, and completable in under one work cycle.
- Keep changes small and reviewable.
- Run relevant tests or checks before committing.
- Record major changes and achieved goals in AUTONOMOUS_LOG.md.
- Mark achieved goals explicitly using the phrase: Achieved goal:
- If blocked, explain the blocker in AUTONOMOUS_LOG.md and pick a safer smaller task.

## Forbidden
- Do not touch secrets, credentials, tokens, .env files, or private keys.
- Do not modify production infrastructure, billing, IAM, auth, deployment, or cloud settings.
- Do not push to main.
- Do not force push.
- Do not delete large amounts of code.
- Do not install new dependencies unless clearly justified in AUTONOMOUS_LOG.md.
