# Contributing

Keep changes scoped and submit them through a pull request.

Before opening a pull request, run:

```text
python -m json.tool templates/skill-sections.json
python -m compileall -q scripts skills/pixeltops-image-editor/scripts
```

Do not commit generated runtime copies, credentials, model weights, or local
environment files.
