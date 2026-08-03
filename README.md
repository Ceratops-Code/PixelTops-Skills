# PixelTops Skills

This repository contains portable, mask-aware image editing skills and their
deterministic runtime helpers.

## Skills

| Skill | Purpose |
| --- | --- |
| `pixeltops-image-editor` | Select, erase, fill, composite, resize, and verify raster images with explicit change boundaries. |

## First install

Run `python scripts/install-skills-bootstrap.py` only for the first skill
installation.

## Deploy

Repository deployment runs `install-runtime`, then `runtime-validation`, before
handing managed skill installation to `ceratops-skill-lifecycle/deploy`. The
runtime installer requires `uv` on `PATH` and stores machine-local environments
and models under `$CODEX_HOME/tools/masked-image-edit`. Regular skill usage
never installs or validates the runtime.
