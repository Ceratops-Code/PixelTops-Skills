## Instruction enforcement

- Treat every instruction bullet in this skill as mandatory and closure-gating
  for the action it governs.
- Do not claim completion when required evidence is missing; report the exact
  blocker or retained uncertainty.

## Core rules

- Use the skill-local executable helpers for deterministic image operations
  and treat their JSON results as the operation record.
- Preserve the user's source image by default and require explicit authority
  before overwriting an existing artifact.
- Match every untouched-pixel claim to an explicit allowed-change mask and a
  successful byte-level verification result.
- Ask before widening the requested image, mask, output, or external-service
  scope; ambiguous object selection alone does not authorize a guess.
- Report only output artifacts, the effective selection and allowed masks,
  verification results, blockers, and important retained variability.
