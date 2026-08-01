# Contributing

Keep changes scoped and submit them through a pull request.

Before opening a pull request, run:

```text
python -m pip install --disable-pip-version-check -r requirements-dev.txt
python scripts/validate_repository.py --evidence-file .validation-evidence.jsonl
```

The validator prints only `OK` on success or compact JSON on failure. Full
command output is written to the evidence file.

Do not commit generated runtime copies, credentials, model weights, or local
environment files.
