---
name: pixeltops-image-editor
description: Perform precise raster-image selection, mask-bounded erasure, filling, background replacement, and blending, plus whole-image resizing and pixel-boundary verification where applicable.
---

# PixelTops Image Editor

## Goal

Edit raster images through one mask-aware CLI while making the exact
allowed-change region inspectable and enforcing unchanged pixels outside it.

## Inputs

- Source image, intended output, and requested operation.
- For selection or masked edits, a text-described object, polygon lasso, or
  supplied mask.
- For blend-capable masked edits, a hard boundary, feather radius, or seamless
  blend.
- For background replacement, a background image or explicit color.

Infer ordinary output and audit filenames beside the requested result. Ask only
when object identity, overwrite authority, or a material visual choice is
ambiguous.

## Command routing

Run `python scripts/image_edit.py <command>` from this skill folder. The public
commands are:

- `select` for Grounding DINO plus SAM 2.1, polygon lasso, or mask normalization.
- `erase` or `fill` for LaMa inpainting inside the resolved mask.
- `remove-background` or `replace-background` for foreground-aware output.
- `composite` for hard, feathered, or seamless masked blending.
- `resize` for explicit contain, cover, or stretch behavior.
- `preview`, `verify`, and `doctor` for inspection and evidence.

Use `--help` on the selected command for exact arguments. Do not reimplement
these deterministic operations in prompt text or ad hoc scripts.

## Workflow

1. Run `doctor` when runtime or model state is not already fresh and verified.
2. For `select`, `erase`, `fill`, `remove-background`, and
   `replace-background`, resolve a selection mask. For text or polygon
   selection, also create and inspect a preview before editing.
3. If multiple text-grounded candidates exist, use `--box-index` only when the
   request or preview identifies the intended object; otherwise ask.
4. Run the narrowest operation. For mask-based mutating commands, keep the
   generated selection mask and allowed-change mask beside the result.
5. For masked RGB edits, require the operation audit or a separate `verify`
   command to report `outsideChangedPixels: 0`.
6. Inspect the result for task correctness. A successful pixel audit proves
   scope containment, not visual quality inside the mask.

## Constraints

- Prefer PNG for masks, audited intermediates, and final lossless verification;
  JPEG compression invalidates byte-exact comparison.
- Treat white mask pixels as editable and black pixels as protected.
- Never suppress a failed outside-mask audit or replace the allowed mask after
  seeing the result.
- AI selection and LaMa filling may vary inside the allowed region. The
  hard-composite and verification boundary remains deterministic.
- Global operations such as resize intentionally affect the whole output; do
  not claim outside-mask preservation for them.
- Use image generation only when the user requests new visual content. Generate
  that content separately, then use this skill's mask-aware compositing path.

## Completion

Complete only when the requested output exists, every applicable selection and
allowed-change mask is saved, masked edits pass outside-change verification,
and visual inspection supports the requested result. For `resize`, `preview`,
and `verify`, require only the artifacts and evidence produced by that command.
Report ambiguity or visual failure instead of treating a zero-pixel boundary
audit as sufficient.

## Output contract

Report the result path, operation used, every applicable selection- or
allowed-change-mask path, the outside-changed-pixel count when verification
applies, and any important variability or blocker. Omit model-loading logs and
routine dependency details when checks pass.
