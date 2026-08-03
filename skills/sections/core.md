<!-- INTERNAL: include in every skill -->

## Instruction enforcement

- All instruction bullets in this file are mandatory, blocking, and
  closure-gating for the phase, action, decision, artifact, or response they
  govern.
- Do not proceed with or claim completion for any action, decision, artifact, or
  response when an applicable instruction bullet is unmet, unverifiable, or in
  conflict; report the blocker or conflict instead.
- The `(D)` label marks a deterministic script or executable-helper contract; it
  does not change the mandatory status of labeled or unlabeled rules.

## Core Rules

- When this skill is invoked, follow this `SKILL.md` as the workflow contract
  for the task; if a higher-precedence instruction conflicts with a required
  skill step, report the conflict instead of silently skipping the step.
- Do not claim completion unless this skill's completion gate is satisfied,
  intentionally inapplicable, or reported as a blocker.
- Scope completion, current-state, root-cause, no-fix, unsupported, and
  durable-resolution claims to evidence actually checked, or to fresh same-task
  evidence that still applies.
- Reuse fresh sufficient same-run evidence unless state is uncertain, plausibly
  changed, materially broadened, externally mutable for the decision, or this
  skill explicitly requires a fresh check.
- Prefer direct local evidence and targeted diagnostics for the next skill
  decision; use current official sources only when local evidence leaves a
  concrete ambiguity or the task depends on unstable external behavior.
- When two active sources use different identities for the same required item,
  treat the producer/current source as authoritative, update stale consumers or
  gates to match it, and ask if authority is unclear.
- Do not do generalized best-practice refresh, reference-repo comparison, or
  skill-maintenance work during routine skill runs unless the user explicitly
  asks or a required decision remains ambiguous after targeted evidence.
- Ask before risky, destructive, irreversible, credential-dependent, externally
  mutating, complex, invasive, nonstandard, or high-maintenance steps unless the
  user already explicitly requested that tradeoff.
- Do not update this `SKILL.md` or other skill/control files during a routine
  run unless the user explicitly asked for skill maintenance or the task cannot
  be completed safely without a narrow in-scope fix.
- For skill runtime workflows, invoke shared helpers through installed console
  commands, `python -m <module>` entrypoints, or scripts copied into the
  installed skill folder; do not locate shared helpers by absolute paths or by
  the repo's parent directory.
- When a workflow needs a shared repo-maintenance script, run `scripts/<name>`
  from the active source checkout root when available, otherwise from the
  installed skill folder; when a helper is skill-local, run it from that skill
  folder or the corresponding source skill folder; stop as blocked if neither
  declared location contains it.
- When editing an existing text file, preserve its current line-ending
  convention unless intentional normalization is part of the task.
- Follow this skill's output contract when present; otherwise report only the
  outcome, unresolved blockers, retained state with reasons, and important
  unverified items.

## Credential Handling

- Do not ask for credentials unless they are truly required after local checks.
- If credentials are truly required after local checks, report only:

1. which credential or login is missing
2. why it is needed
3. where it will be stored
4. the exact command the user should run
5. whether it goes into a local credential store, config file, keyring, CI
   secret, registry setting, connector, or another exact target

- If the user refuses a missing permission, credential, login, or scope, stop
  retrying and report the blocked action and exact entities still pending.
